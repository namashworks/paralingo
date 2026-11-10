# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
import io
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from paralingo.api_keys import KeyStore  # noqa: E402
from paralingo.audio import save_wav  # noqa: E402
from paralingo.http_api import create_app  # noqa: E402

SR = 16000


def _wav_bytes(tmp_path, f0=170.0, dur=1.5):
    t = np.arange(int(dur * SR)) / SR
    x = 0.3 * sum((0.6 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 6))
    x = np.concatenate([np.random.default_rng(0).normal(0, 5e-4, 3200), x, np.random.default_rng(1).normal(0, 5e-4, 3200)])
    p = tmp_path / "turn.wav"
    save_wav(p, x, SR)
    return p.read_bytes()


@pytest.fixture()
def env(tmp_path):
    ks = KeyStore(tmp_path / "keys.json")
    full, _ = ks.create("bot", "acme", ["profiles:read", "profiles:write", "audio:analyze"])
    ro, _ = ks.create("dashboard", "acme", ["profiles:read"])
    other, _ = ks.create("other-co", "globex", ["admin"])
    admin, _ = ks.create("ops", "acme", ["admin"])
    client = TestClient(create_app(tmp_path / "profiles", tmp_path / "keys.json"))
    return client, {"full": full, "ro": ro, "other": other, "admin": admin}, ks, tmp_path


def H(k):
    return {"Authorization": f"Bearer {k}"}


def test_keys_are_hashed_and_verified(env):
    _, keys, ks, tmp = env
    raw = (tmp / "keys.json").read_text()
    assert keys["full"] not in raw and "key_hash" in raw
    assert ks.verify(keys["full"]).name == "bot"
    assert ks.verify(keys["full"][:-2] + "xx") is None
    assert ks.verify("nonsense") is None


def test_auth_scopes_and_attribution(env):
    c, keys, ks, _ = env
    r = c.get("/v1/health")
    assert r.status_code == 200 and "paralingo" in r.headers["X-Powered-By"]
    assert c.get("/v1/speakers").status_code == 401
    assert c.get("/v1/speakers", headers={"X-API-Key": keys["ro"]}).status_code == 200
    assert c.post("/v1/speakers/u1/sessions", headers=H(keys["ro"])).status_code == 403
    names = [t["name"] for t in c.get("/v1/tools", headers=H(keys["ro"])).json()]
    assert "paralingo_get_summary" in names and "paralingo_add_observation" not in names
    ks.revoke(ks.verify(keys["ro"]).key_id)
    assert c.get("/v1/speakers", headers=H(keys["ro"])).status_code == 401


def test_full_flow_with_upload_and_tenant_isolation(env, tmp_path):
    c, keys, _, _ = env
    sid = c.post("/v1/speakers/caller-1/sessions", headers=H(keys["full"]), json={"channel": "phone"}).json()["session_id"]
    r = c.post("/v1/speakers/caller-1/utterances", headers=H(keys["full"]),
               data={"session_id": sid, "transcript": "um, hi, I need to change my plan please",
                     "external": json.dumps({"asr_confidence": 0.9})},
               files={"audio": ("turn.wav", io.BytesIO(_wav_bytes(tmp_path)), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["key_metrics"]["f0_mean_hz"] > 160
    bad = c.post("/v1/speakers/caller-1/utterances", headers=H(keys["full"]),
                 files={"audio": ("x.wav", io.BytesIO(b"not a wav file at all"), "audio/wav")})
    assert bad.status_code == 415
    r = c.post("/v1/tools/paralingo_add_observation", headers=H(keys["full"]),
               json={"speaker_id": "caller-1", "category": "preference", "confidence": 0.6,
                     "observation": "Prefers short answers."})
    assert r.status_code == 200 and r.json()["ok"]
    assert c.post("/v1/tools/paralingo_analyze_utterance", headers=H(keys["full"]),
                  json={"speaker_id": "caller-1", "audio_path": "/etc/passwd"}).status_code == 400
    assert c.post(f"/v1/speakers/caller-1/sessions/{sid}/end", headers=H(keys["full"])).status_code == 200
    s = c.get("/v1/speakers/caller-1/summary", headers=H(keys["ro"])).json()
    assert s["sessions_total"] == 1
    # audit log names the key that wrote
    sec = c.post("/v1/tools/paralingo_read_section", headers=H(keys["ro"]),
                 json={"speaker_id": "caller-1", "section": "audit_log"}).json()["result"]
    assert any(e["agent_id"] == "api:bot" for e in sec)
    # another tenant cannot see it, even with admin scope
    assert c.get("/v1/speakers/caller-1/summary", headers=H(keys["other"])).status_code == 404
    assert c.get("/v1/speakers", headers=H(keys["other"])).json() == []
    # erasure needs admin
    assert c.delete("/v1/speakers/caller-1", headers=H(keys["full"])).status_code == 403
    assert c.delete("/v1/speakers/caller-1", headers=H(keys["admin"])).json() == {"deleted": True}
    assert c.get("/v1/speakers/caller-1/summary", headers=H(keys["admin"])).status_code == 404


def test_rate_limit(tmp_path):
    ks = KeyStore(tmp_path / "keys.json")
    k, _ = ks.create("tiny", "t", ["profiles:read"], rate_per_minute=2)
    c = TestClient(create_app(tmp_path / "p", tmp_path / "keys.json"))
    codes = [c.get("/v1/speakers", headers=H(k)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
