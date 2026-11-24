# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
import json
from types import SimpleNamespace

import numpy as np
import pytest

from paralingo import SpeechProfileEngine, validate
from paralingo.realtime import TurnDetector, analyze_turn, is_backchannel, lexical_cues

SR = 16000
RNG = np.random.default_rng(0)


def voice(dur, f0=170.0, glide=0.0, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * 2 ** (glide * t / dur / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return amp * sum((0.6 / k) * np.sin(k * ph) for k in range(1, 6)) * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * t) ** 2)


def sil(d):
    return RNG.normal(0, 0.002, int(d * SR))


def feed(det, segments, chunk=320, encoding="float"):
    """segments: (audio, transcript pushed after the audio or None, agent_speaking)."""
    events = []
    for audio, text, agent in segments:
        det.set_agent_speaking(agent)
        for i in range(0, len(audio), chunk):
            part = audio[i:i + chunk]
            if encoding == "pcm16":
                part = (np.clip(part, -1, 1) * 32767).astype("<i2").tobytes()
            events += det.push_audio(part, encoding="pcm16")
        if text is not None:
            det.push_transcript(text)
    return events


# --------------------------------------------------------------------------- language cues
@pytest.mark.parametrize("text,lang,cont,complete,question", [
    ("I wanted to ask about my order and", "en", True, False, False),
    ("It never arrived.", "en", False, True, False),
    ("Where is it?", "en", False, True, True),
    ("um", "en", True, False, False),
    ("我想问一下然后", "zh", True, False, False),
    ("什么时候到吗？", "zh", False, True, True),
    ("お届けが遅れているんですけど", "ja", True, False, False),
    ("届きましたか？", "ja", False, True, True),
    ("주문했는데", "ko", True, False, False),
    ("배송이 언제 와요?", "ko", False, True, True),
])
def test_lexical_cues_four_languages(text, lang, cont, complete, question):
    c = lexical_cues(text)
    assert (c["continuation"], c["complete"], c["question"]) == (cont, complete, question)


def test_backchannels():
    assert all(is_backchannel(t) for t in ("Mm-hmm.", "yeah", "嗯", "はい", "네"))
    assert not any(is_backchannel(t) for t in ("I disagree", "wait stop", "그게 아니라"))


# --------------------------------------------------------------------------- end of turn
def test_holds_through_hesitation_and_ends_complete_turn_fast():
    det = TurnDetector(SR)
    ev = feed(det, [
        (sil(0.5), None, False),
        (voice(1.2, glide=-1), "I wanted to ask about my order and", False),
        (sil(0.9), None, False),                                          # "and..." hesitation
        (voice(1.0, glide=-6), "I wanted to ask about my order and it never arrived.", False),
        (sil(1.5), None, False),
    ])
    eots = [e for e in ev if e.type == "end_of_turn"]
    assert len(eots) == 1, [e.type for e in ev]
    d = eots[0].data
    assert d["transcript"].endswith("arrived.") and d["silence_ms"] <= 450
    assert len(d["pauses_s"]) == 1 and 0.8 <= d["pauses_s"][0] <= 1.0
    assert d["audio"].size > 2.0 * SR and d["sample_rate"] == SR
    assert d["language"] == "en" and d["prosody"]["contour"] == "falling"


def test_no_transcript_still_ends_by_timing():
    det = TurnDetector(SR)
    ev = feed(det, [(sil(0.5), None, False), (voice(1.0, glide=0.5), None, False), (sil(2.5), None, False)])
    eots = [e for e in ev if e.type == "end_of_turn"]
    assert len(eots) == 1 and 0.5 <= eots[0].data["silence_ms"] / 1000 <= 1.7


def test_personal_pause_calibration_waits_longer(tmp_path):
    eng = SpeechProfileEngine(tmp_path)
    eng.ensure_profile("slow")
    with eng.store.edit("slow", agent_id="t", action="set_guidance") as p:
        p["agent_guidance"]["end_of_turn_silence_ms"] = 1750
    det = TurnDetector.from_profile(eng, "slow", SR)
    assert abs(det.pause_s - 1.5) < 1e-9
    fast = TurnDetector(SR)
    segs = [(sil(0.5), None, False), (voice(1.0, glide=0.5), None, False), (sil(3.0), None, False)]
    t_slow = [e for e in feed(det, segs) if e.type == "end_of_turn"][0].data["silence_ms"]
    t_fast = [e for e in feed(fast, segs) if e.type == "end_of_turn"][0].data["silence_ms"]
    assert t_slow > t_fast + 400


def test_chunk_size_and_encoding_do_not_change_decisions():
    segs = [(sil(0.4), None, False), (voice(1.0, glide=-6), "That is all.", False), (sil(1.2), None, False)]
    times = []
    for chunk, enc in ((160, "float"), (320, "pcm16"), (1600, "float"), (7, "pcm16")):
        RNG_state = np.random.default_rng(1)
        globals()["RNG"] = RNG_state
        ev = feed(TurnDetector(SR), segs, chunk=chunk, encoding=enc)
        times.append([(e.type, e.t) for e in ev])
    assert all(t == times[0] for t in times[1:]), times


def test_mulaw_8khz_phone_audio():
    sr = 8000
    t = np.arange(int(1.0 * sr)) / sr
    x = np.concatenate([np.zeros(4000), 0.3 * np.sin(2 * np.pi * 140 * t * (1 - 0.1 * t)), np.zeros(12000)])
    det = TurnDetector(sr)
    ulaw = np.array([_lin2ulaw(v) for v in (x * 32767).astype(np.int16)], dtype=np.uint8).tobytes()
    ev = det.push_audio(ulaw, encoding="mulaw")
    assert [e.type for e in ev] == ["speech_start", "end_of_turn"]


def _lin2ulaw(sample: int) -> int:
    """Reference G.711 mu-law encoder (ITU-T)."""
    BIAS, CLIP = 0x84, 32635
    sign = 0x80 if sample < 0 else 0
    s = min(abs(int(sample)), CLIP) + BIAS
    exponent, mask = 7, 0x4000
    while exponent > 0 and not (s & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (s >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


# --------------------------------------------------------------------------- interruptions
def test_backchannel_vs_barge_in():
    det = TurnDetector(SR)
    ev = feed(det, [
        (sil(0.5), None, True),
        (voice(0.35, f0=200), "yeah", True),            # "yeah" while the agent talks
        (sil(0.8), None, True),
        (voice(1.2, f0=210, glide=2)[:int(0.25 * SR)], "wait that's not", True),
        (voice(1.0, f0=210), "wait that's not what I said", True),
        (sil(1.8), None, False),
    ])
    types = [e.type for e in ev]
    assert "backchannel" in types
    bc = types.index("backchannel")
    assert "barge_in" in types[bc:] and "end_of_turn" in types[bc:]
    barge = next(e for e in ev if e.type == "barge_in")
    assert barge.data["speech_s"] <= 0.45                # real words -> fast reaction
    assert types.count("end_of_turn") == 1               # the backchannel didn't create a turn
    eot = next(e for e in ev if e.type == "end_of_turn")
    assert eot.data["interrupted_agent"] is True


def test_latency_tracking_and_speed():
    det = TurnDetector(SR)
    feed(det, [(sil(0.5), None, False), (voice(1.0, glide=-6), "Thanks.", False), (sil(0.6), None, False)])
    det.set_agent_speaking(True)
    s = det.latency.summary()
    assert s["end_of_turn_delay_ms"]["n"] == 1 and s["response_ms"]["n"] == 1
    assert s["processing"]["real_time_factor"] < 0.25


def test_analyze_turn_and_store_metrics(tmp_path):
    eng = SpeechProfileEngine(tmp_path, agent_id="rt")
    sid = eng.start_session("u")
    det = TurnDetector(SR)
    ev = feed(det, [(sil(0.5), None, False), (voice(2.0, glide=-6), "Um, my order never arrived.", False),
                    (sil(1.0), None, False)])
    eot = next(e for e in ev if e.type == "end_of_turn")
    res = analyze_turn(eng, "u", eot, session_id=sid)
    assert 130 < res["key_metrics"]["f0_mean_hz"] < 160     # 170 Hz gliding down to 120 Hz
    eng.record_realtime_metrics("u", sid, det.latency.summary())
    p = eng.get_profile("u")
    assert p["sessions"][-1]["realtime"]["end_of_turn_delay_ms"]["n"] == 1
    assert validate(p) == []


# --------------------------------------------------------------------------- Azure adapter
class _Signal:
    def __init__(self):
        self.handlers = []

    def connect(self, fn):
        self.handlers.append(fn)

    def fire(self, evt):
        for h in self.handlers:
            h(evt)


def _azure_result(text, words=(), language=None, confidence=0.91):
    detail = {"DisplayText": text, "NBest": [{"Confidence": confidence, "Display": text, "Words": [
        {"Word": w, "Offset": int(s * 1e7), "Duration": int((e - s) * 1e7), "Confidence": 0.9} for w, s, e in words]}]}
    if language:
        detail["PrimaryLanguage"] = {"Language": language, "Confidence": "High"}
    return SimpleNamespace(result=SimpleNamespace(text=text, json=json.dumps(detail), properties=None))


def test_azure_bridge_feeds_detector_and_stream():
    from paralingo.integrations.azure import AzureTurnBridge, result_language, result_words

    r = _azure_result("届きましたか？", [("届きましたか", 0.6, 1.4)], language="ja-JP")
    assert result_language(r.result) == "ja-JP"
    assert result_words(r.result) == [{"word": "届きましたか", "start": 0.6, "end": 1.4, "confidence": 0.9}]

    written, got = [], []
    stream = SimpleNamespace(write=lambda b: written.append(len(b)))
    recognizer = SimpleNamespace(recognizing=_Signal(), recognized=_Signal())
    bridge = AzureTurnBridge(TurnDetector(SR), stream=stream, on_event=got.append).attach(recognizer)

    def push(audio):
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), 640):
            bridge.push_audio(pcm[i:i + 640])

    push(sil(0.5))
    push(voice(1.0, glide=-6))
    recognizer.recognizing.fire(_azure_result("届きました", language="ja-JP"))
    recognizer.recognized.fire(_azure_result("届きましたか？", [("届きましたか", 0.55, 1.45)], language="ja-JP"))
    push(sil(1.2))
    eot = [e for e in got if e.type == "end_of_turn"]
    assert len(eot) == 1 and sum(written) == int(2.7 * SR) * 2
    d = eot[0].data
    assert d["transcript"] == "届きましたか？" and d["language"] == "ja" and d["lexical"]["question"]
    assert d["asr_confidence"] == 0.91 and d["words"][0]["word"] == "届きましたか"
    # word at 0.55 s in the stream; turn audio starts 0.2 s (pre-roll) before speech at 0.5 s
    assert abs(d["words"][0]["start"] - 0.25) < 0.03


# --------------------------------------------------------------------------- WebSocket endpoint
def test_websocket_realtime_endpoint(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from paralingo.api_keys import KeyStore
    from paralingo.http_api import create_app

    ks = KeyStore(tmp_path / "keys.json")
    key, _ = ks.create("yuzu", "acme", ["audio:analyze", "profiles:read", "profiles:write"])
    ro, _ = ks.create("ro", "acme", ["profiles:read"])
    client = TestClient(create_app(tmp_path / "profiles", tmp_path / "keys.json"))
    sid = client.post("/v1/speakers/caller-1/sessions", headers={"Authorization": f"Bearer {key}"}).json()["session_id"]

    with client.websocket_connect(f"/v1/realtime?speaker_id=caller-1&sample_rate={SR}&session_id={sid}") as ws:
        ws.send_text(json.dumps({"type": "auth", "api_key": key}))
        assert json.loads(ws.receive_text())["type"] == "ready"
        audio = np.concatenate([sil(0.5), voice(1.5, glide=-6), sil(1.2)])
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        msgs = []
        for i in range(0, len(pcm), 640):
            if i == int(1.6 * SR) * 2:
                ws.send_text(json.dumps({"type": "transcript", "text": "Where is my order?", "final": True}))
            ws.send_bytes(pcm[i:i + 640])
        ws.send_text(json.dumps({"type": "stop"}))
        while True:
            m = json.loads(ws.receive_text())
            msgs.append(m)
            if m["type"] == "summary":
                break
    types = [m["type"] for m in msgs]
    assert types[:3] == ["speech_start", "end_of_turn", "analysis"] and types[-1] == "summary"
    eot = msgs[1]["data"]
    assert eot["transcript"] == "Where is my order?" and "audio" not in eot
    assert 130 < msgs[2]["data"]["key_metrics"]["f0_mean_hz"] < 160
    prof = client.get("/v1/speakers/caller-1/summary", headers={"Authorization": f"Bearer {key}"}).json()
    assert prof["sessions_total"] == 1

    with client.websocket_connect(f"/v1/realtime?speaker_id=caller-1") as ws:
        ws.send_text(json.dumps({"type": "auth", "api_key": ro}))
        assert "lacks scope" in json.loads(ws.receive_text())["error"]


def test_fixed_timeout_baseline_mode():
    from paralingo.realtime import TurnConfig

    det = TurnDetector(SR, TurnConfig(fixed_timeout_s=0.7))
    ev = feed(det, [(sil(0.5), None, False), (voice(1.0), "and", False), (sil(1.5), None, False)])
    eot = [e for e in ev if e.type == "end_of_turn"]
    assert len(eot) == 1 and 690 <= eot[0].data["silence_ms"] <= 720   # ignores the "and" cue entirely


def test_soft_sentence_start_resumes_the_turn():
    """Hysteresis: a quieter next sentence still counts as the same person continuing."""
    det = TurnDetector(SR)
    ev = feed(det, [(sil(0.5), None, False), (voice(1.0, amp=0.3), "I need help with my order and", False),
                    (sil(0.6), None, False), (voice(1.0, amp=0.02), None, False), (sil(2.5), None, False)])
    eots = [e for e in ev if e.type == "end_of_turn"]
    assert len(eots) == 1 and eots[0].data["turn_duration_s"] > 2.4
