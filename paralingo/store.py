# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Safe, concurrent read/write access to speech profile files.

One file per speaker: ``<root>/<speaker_id>.speech.json``.

Guarantees
----------
* **Atomic writes** - data is written to a temp file and swapped in with
  ``os.replace`` so a crash never leaves a half-written profile.
* **Cross-process locking** - a ``.lock`` file (O_EXCL) serialises writers,
  so several voice agents / workers can share the same profile safely.
  Stale locks (crashed writer) are broken after ``stale_lock_s``.
* **Optimistic concurrency** - every write bumps ``revision``; callers may pass
  ``expected_revision`` to refuse writing over someone else's change.
* **Schema validation** (when ``jsonschema`` is installed).
* **Audit trail** - who changed what, when.
* **Retention** - consent-driven purge and compaction of old sessions.
"""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

FORMAT_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).parent / "schema" / "paralingo.schema.json"
FILE_SUFFIX = ".speech.json"
MAX_AUDIT_ENTRIES = 500
MAX_OBSERVATIONS = 1000
MAX_TREND_POINTS = 200

try:  # optional
    import jsonschema  # type: ignore

    _VALIDATOR = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text("utf-8")))
except Exception:  # pragma: no cover - jsonschema missing
    _VALIDATOR = None


class ProfileError(Exception):
    """Base error."""


class ProfileNotFound(ProfileError):
    pass


class ProfileExists(ProfileError):
    pass


class ConcurrencyConflict(ProfileError):
    pass


class ConsentError(ProfileError):
    pass


class LockTimeout(ProfileError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _safe_name(speaker_id: str) -> str:
    if not speaker_id or not speaker_id.strip():
        raise ProfileError("speaker_id must be non-empty")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", speaker_id.strip())
    if name in {".", ".."}:
        raise ProfileError("invalid speaker_id")
    return name[:120]


def new_profile(
    speaker_id: str,
    *,
    display_name: str | None = None,
    languages: list[str] | None = None,
    consent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an empty, schema-valid profile document."""
    now = utcnow()
    default_consent = {
        "recording": False,
        "voice_analysis": True,
        "health_indicators": False,  # opt-in only
        "store_transcripts": True,
        "retention_days": 365,
        "granted_at": None,
        "granted_via": None,
    }
    default_consent.update(consent or {})
    return {
        "format": "paralingo",
        "version": FORMAT_VERSION,
        "profile_id": new_id("prof"),
        "created_at": now,
        "updated_at": now,
        "revision": 0,
        "speaker": {
            "speaker_id": speaker_id,
            "display_name": display_name,
            "languages": languages or [],
            "self_reported": {},
            "notes": None,
        },
        "consent": default_consent,
        "settings": {"max_detailed_sessions": 20, "max_utterances_per_session": 300},
        "baseline": {
            "status": "insufficient_data",
            "sessions_used": 0,
            "utterances_used": 0,
            "speech_seconds_used": 0.0,
            "metrics": {},
        },
        "accent": {"current_estimate": None, "estimates": [], "phonetic_markers": [], "intelligibility": None},
        "sessions": [],
        "observations": [],
        "assessments": {
            "emotional_state": {"current": None, "recent_labels": [], "trend": "unknown"},
            "communication_skills": {"scores": {}, "strengths": [], "areas_to_improve": [], "updated_at": None},
            "speaking_style": {
                "pace": "unknown",
                "pitch_expressiveness": "unknown",
                "loudness": "unknown",
                "pausing": "unknown",
                "verbosity": "unknown",
                "directness": "unknown",
                "mean_words_per_turn": None,
                "typical_response_latency_s": None,
            },
            "speech_indicators": [],
        },
        "trends": {},
        "agent_guidance": {
            "recommended_speaking_rate_wpm": None,
            "end_of_turn_silence_ms": None,
            "adaptations": [],
            "updated_at": None,
        },
        "audit_log": [],
        "extensions": {},
    }


def validate(profile: dict[str, Any]) -> list[str]:
    """Return a list of schema errors (empty if valid or jsonschema unavailable)."""
    if _VALIDATOR is None:
        return []
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in _VALIDATOR.iter_errors(_clean_json(profile))
    ]


def _clean_json(obj: Any) -> Any:
    """Round-trip through JSON so numpy scalars / tuples are normalised."""
    return json.loads(json.dumps(obj, default=_json_default))


def _json_default(o: Any) -> Any:
    # numpy scalars, sets, Paths...
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o).__name__}")


class SpeechProfileStore:
    """Directory-backed store of speech profiles."""

    def __init__(self, root: str | os.PathLike[str], *, lock_timeout_s: float = 10.0, stale_lock_s: float = 30.0,
                 strict_validation: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_s = lock_timeout_s
        self.stale_lock_s = stale_lock_s
        self.strict_validation = strict_validation

    # ------------------------------------------------------------------ paths
    def path_for(self, speaker_id: str) -> Path:
        return self.root / f"{_safe_name(speaker_id)}{FILE_SUFFIX}"

    def exists(self, speaker_id: str) -> bool:
        return self.path_for(speaker_id).exists()

    def list_speakers(self) -> list[str]:
        out = []
        for p in sorted(self.root.glob(f"*{FILE_SUFFIX}")):
            try:
                out.append(json.loads(p.read_text("utf-8"))["speaker"]["speaker_id"])
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------ CRUD
    def create(self, speaker_id: str, *, agent_id: str = "system", exist_ok: bool = False, **kwargs: Any) -> dict[str, Any]:
        path = self.path_for(speaker_id)
        with self._lock(path):
            if path.exists():
                if exist_ok:
                    return self._read(path)
                raise ProfileExists(f"profile for '{speaker_id}' already exists")
            profile = new_profile(speaker_id, **kwargs)
            self._append_audit(profile, agent_id, "create_profile", None)
            self._write(path, profile)
            return copy.deepcopy(profile)

    def load(self, speaker_id: str) -> dict[str, Any]:
        path = self.path_for(speaker_id)
        if not path.exists():
            raise ProfileNotFound(f"no profile for '{speaker_id}'")
        return self._read(path)

    def delete(self, speaker_id: str) -> bool:
        """Right-to-erasure: remove the speaker's file entirely."""
        path = self.path_for(speaker_id)
        with self._lock(path):
            if path.exists():
                path.unlink()
                return True
            return False

    @contextmanager
    def edit(self, speaker_id: str, *, agent_id: str, action: str, detail: str | None = None,
             expected_revision: int | None = None, create_if_missing: bool = False) -> Iterator[dict[str, Any]]:
        """Lock -> load -> yield mutable profile -> validate -> atomic write -> unlock.

        If the body raises, nothing is written.
        """
        path = self.path_for(speaker_id)
        with self._lock(path):
            if path.exists():
                profile = self._read(path)
            elif create_if_missing:
                profile = new_profile(speaker_id)
                self._append_audit(profile, agent_id, "create_profile", None)
            else:
                raise ProfileNotFound(f"no profile for '{speaker_id}'")
            if expected_revision is not None and profile["revision"] != expected_revision:
                raise ConcurrencyConflict(
                    f"expected revision {expected_revision}, file is at {profile['revision']}")
            yield profile
            profile["revision"] += 1
            profile["updated_at"] = utcnow()
            self._append_audit(profile, agent_id, action, detail)
            apply_retention(profile)
            self._write(path, profile)

    # ------------------------------------------------------------------ internals
    def _read(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, path: Path, profile: dict[str, Any]) -> None:
        clean = _clean_json(profile)
        errors = validate(clean)
        if errors and self.strict_validation:
            raise ProfileError("schema validation failed: " + "; ".join(errors[:5]))
        fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(clean, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            for attempt in range(20):  # Windows: target may be briefly held by a reader / AV / OneDrive
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.05)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        profile.clear()
        profile.update(clean)

    @staticmethod
    def _append_audit(profile: dict[str, Any], agent_id: str, action: str, detail: str | None) -> None:
        profile["audit_log"].append({
            "timestamp": utcnow(), "agent_id": agent_id, "action": action,
            "detail": detail, "revision": profile["revision"],
        })
        del profile["audit_log"][:-MAX_AUDIT_ENTRIES]

    @contextmanager
    def _lock(self, path: Path) -> Iterator[None]:
        lock = path.with_name(path.name + ".lock")
        deadline = time.monotonic() + self.lock_timeout_s
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}".encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > self.stale_lock_s:
                        lock.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise LockTimeout(f"could not lock {path.name} within {self.lock_timeout_s}s")
                time.sleep(0.01)
            except PermissionError:  # Windows: lock file being deleted concurrently
                if time.monotonic() > deadline:
                    raise LockTimeout(f"could not lock {path.name}")
                time.sleep(0.01)
        try:
            yield
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def apply_retention(profile: dict[str, Any], now: datetime | None = None) -> None:
    """Purge expired sessions and compact old detailed sessions."""
    now = now or datetime.now(timezone.utc)
    days = profile["consent"].get("retention_days")
    if days:
        cutoff = now - timedelta(days=days)
        profile["sessions"] = [s for s in profile["sessions"] if parse_ts(s["started_at"]) >= cutoff]
        profile["observations"] = [o for o in profile["observations"] if parse_ts(o["timestamp"]) >= cutoff]
        keep_ids = {s["session_id"] for s in profile["sessions"]}
        for metric, points in profile["trends"].items():
            profile["trends"][metric] = [p for p in points if parse_ts(p["date"]) >= cutoff or p["session_id"] in keep_ids]

    max_detailed = profile.get("settings", {}).get("max_detailed_sessions", 20)
    closed = [s for s in profile["sessions"] if s["status"] == "closed"]
    for s in closed[:-max_detailed] if len(closed) > max_detailed else []:
        if s["utterances"]:
            s["utterances"] = []
            s["utterances_compacted"] = True

    if not profile["consent"].get("store_transcripts", True):
        for s in profile["sessions"]:
            for u in s["utterances"]:
                u["transcript"] = None

    del profile["observations"][:-MAX_OBSERVATIONS]
    for metric in profile["trends"]:
        del profile["trends"][metric][:-MAX_TREND_POINTS]
