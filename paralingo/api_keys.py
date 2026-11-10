# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""API key management for the HTTP API.

Keys look like ``spk_<key_id>_<secret>``. Only a SHA-256 hash of each key is stored, so a leaked
keys file does not reveal usable keys. Each key belongs to one *tenant* (an isolated set of
profiles) and carries *scopes*:

* ``profiles:read``   - summaries, guidance, sections, trends, observations
* ``profiles:write``  - sessions, observations, accent, consent
* ``audio:analyze``   - submit user turns (audio and/or transcript) for analysis
* ``admin``           - everything above, plus deleting a speaker's data

    python -m paralingo.api_keys --keys keys.json create --name acme-bot --tenant acme \\
        --scopes profiles:read profiles:write audio:analyze
    python -m paralingo.api_keys --keys keys.json list
    python -m paralingo.api_keys --keys keys.json revoke <key_id>
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .store import utcnow

SCOPES = ("profiles:read", "profiles:write", "audio:analyze", "admin")
PREFIX = "spk_"
_KEY_RE = re.compile(r"^spk_([0-9a-f]{10})_([A-Za-z0-9_-]{20,})$")
_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


@dataclass
class ApiKey:
    key_id: str
    name: str
    tenant: str
    scopes: list[str]
    key_hash: str
    created_at: str
    revoked: bool = False
    rate_per_minute: int = 120
    extra: dict[str, Any] = field(default_factory=dict)

    def allows(self, scope: str) -> bool:
        return not self.revoked and ("admin" in self.scopes or scope in self.scopes)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    """JSON-file key store. Re-reads the file when it changes, so revocations apply without a restart."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._keys: dict[str, ApiKey] = {}

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.path.exists():
            self._keys, self._mtime = {}, None
            return
        mtime = self.path.stat().st_mtime
        if mtime == self._mtime:
            return
        data = json.loads(self.path.read_text("utf-8"))
        self._keys = {k["key_id"]: ApiKey(**k) for k in data.get("keys", [])}
        self._mtime = mtime

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "paralingo-keys", "keys": [asdict(k) for k in self._keys.values()]}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".keys_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        self._mtime = None

    # ------------------------------------------------------------------ operations
    def create(self, name: str, tenant: str, scopes: list[str], rate_per_minute: int = 120) -> tuple[str, ApiKey]:
        """Create a key. Returns (plaintext_key, record); the plaintext is never stored."""
        bad = [s for s in scopes if s not in SCOPES]
        if bad or not scopes:
            raise ValueError(f"scopes must be a non-empty subset of {SCOPES}; got {scopes}")
        if not _TENANT_RE.match(tenant):
            raise ValueError("tenant must be lowercase letters, digits, '-' or '_' (max 63 chars)")
        with self._lock:
            self._load()
            key_id = secrets.token_hex(5)
            while key_id in self._keys:
                key_id = secrets.token_hex(5)
            plain = f"{PREFIX}{key_id}_{secrets.token_urlsafe(32)}"
            rec = ApiKey(key_id=key_id, name=name, tenant=tenant, scopes=sorted(set(scopes)),
                         key_hash=_hash(plain), created_at=utcnow(), rate_per_minute=rate_per_minute)
            self._keys[key_id] = rec
            self._save()
            return plain, rec

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            self._load()
            rec = self._keys.get(key_id)
            if not rec:
                return False
            rec.revoked = True
            self._save()
            return True

    def list(self) -> list[ApiKey]:
        with self._lock:
            self._load()
            return list(self._keys.values())

    def verify(self, presented: str | None) -> ApiKey | None:
        """Return the key record for a valid, non-revoked key, else None (constant-time hash compare)."""
        if not presented:
            return None
        m = _KEY_RE.match(presented.strip())
        if not m:
            return None
        with self._lock:
            self._load()
            rec = self._keys.get(m.group(1))
        if rec is None or rec.revoked:
            return None
        return rec if hmac.compare_digest(rec.key_hash, _hash(presented.strip())) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="paralingo.api_keys", description="Manage Paralingo API keys")
    ap.add_argument("--keys", default="keys.json", help="path to the keys file (keep it out of Git)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--tenant", default="default")
    c.add_argument("--scopes", nargs="+", default=["profiles:read"], choices=SCOPES)
    c.add_argument("--rate", type=int, default=120, help="requests per minute")
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("key_id")
    a = ap.parse_args(argv)

    ks = KeyStore(a.keys)
    if a.cmd == "create":
        plain, rec = ks.create(a.name, a.tenant, a.scopes, a.rate)
        print(f"Created key {rec.key_id} for tenant '{rec.tenant}' with scopes {rec.scopes}.")
        print("Copy it now; it cannot be shown again:\n")
        print(plain)
    elif a.cmd == "list":
        for k in ks.list():
            state = "revoked" if k.revoked else "active"
            print(f"{k.key_id}  {state:8} tenant={k.tenant:12} {k.name:20} {','.join(k.scopes)}  created {k.created_at}")
    else:
        print("revoked" if ks.revoke(a.key_id) else "no such key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
