# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
import json
import threading

import numpy as np
import pytest

from paralingo import (ConcurrencyConflict, ConsentError, SpeechProfileEngine, SpeechProfileStore,
                            SpeechProfileTools, analyze_audio, analyze_text, pauses_from_word_timestamps, validate)
from paralingo.audio import save_wav

SR = 16000


def voice(f0, dur, jit=0.0, shim=0.0, glide=0.0, amp=0.3, seed=0):
    """Glottal-like harmonic signal with controllable jitter / shimmer / pitch glide (semitones)."""
    rng = np.random.default_rng(seed)
    out, t = [], 0.0
    while t < dur:
        f = f0 * 2 ** (glide * t / dur / 12)
        T = (1 / f) * (1 + rng.normal(0, jit))
        a = 1 + rng.normal(0, shim)
        tt = np.arange(int(T * SR)) / SR
        out.append(a * sum((0.6 / k) * np.sin(2 * np.pi * k * tt / T) for k in range(1, 6)))
        t += T
    return amp * np.concatenate(out)


def sil(d, seed=1):
    return np.random.default_rng(seed).normal(0, 0.0005, int(d * SR))


# ----------------------------------------------------------------------- audio
def test_pitch_pauses_and_quality_on_clean_voice():
    x = np.concatenate([sil(.3), voice(150, 1.0), sil(.6), voice(150, 1.2), sil(1.3), voice(150, .8), sil(.3)])
    r = analyze_audio(x, SR)
    assert abs(r["pitch"]["f0_median_hz"] - 150) < 4
    assert r["pauses"]["count"] == 2
    assert r["pauses"]["long_count"] == 1
    assert 2.7 < r["speech_s"] < 3.4
    assert r["voice_quality"]["hnr_db"] > 25
    assert r["voice_quality"]["jitter_local_pct"] < 1.04
    assert r["audio_quality"]["quality"] == "good"


def test_jitter_shimmer_increase_with_perturbation():
    clean = analyze_audio(np.concatenate([sil(.2), voice(150, 2), sil(.2)]), SR)["voice_quality"]
    rough = analyze_audio(np.concatenate([sil(.2), voice(150, 2, jit=0.02, shim=0.1), sil(.2)]), SR)["voice_quality"]
    assert rough["jitter_local_pct"] > clean["jitter_local_pct"] + 1
    assert rough["shimmer_local_pct"] > clean["shimmer_local_pct"] + 3
    assert rough["hnr_db"] < clean["hnr_db"]


def test_intonation_rising_vs_level():
    rise = analyze_audio(np.concatenate([sil(.2), voice(140, 1.0, glide=8), sil(.2)]), SR)["pitch"]
    flat = analyze_audio(np.concatenate([sil(.2), voice(140, 1.0), sil(.2)]), SR)["pitch"]
    assert rise["terminal_contour"] == "rising"
    assert rise["f0_range_semitones"] > flat["f0_range_semitones"] + 3


def test_silence_only_does_not_crash():
    r = analyze_audio(sil(1.0), SR)
    assert r["speech_s"] == 0 and r["pitch"]["f0_mean_hz"] is None


def test_word_timestamp_pauses():
    w = [{"word": "hi", "start": 0.0, "end": 0.3}, {"word": "there", "start": 0.35, "end": 0.7},
         {"word": "so", "start": 1.9, "end": 2.1}]
    p = pauses_from_word_timestamps(w)
    assert p["count"] == 1 and p["long_count"] == 1 and abs(p["max_s"] - 1.2) < 1e-6


# ----------------------------------------------------------------------- language
def test_disfluency_and_lexical_features():
    t = analyze_text("Um, I I want to b-b-buy the, uh, thing. I mean, you know, maybe the stuff?")
    f = t["fluency"]
    assert f["fillers"]["um"] == 1 and f["fillers"]["uh"] == 1 and f["fillers"]["you know"] == 1
    assert f["word_repetitions"] == 1
    assert f["part_word_repetitions"] == 1
    assert f["revisions"] >= 1
    assert t["lexical"]["hedges"].get("maybe") == 1
    assert t["lexical"]["vague_word_count"] == 2
    assert t["lexical"]["question_count"] == 1


def test_sentiment_negation():
    assert analyze_text("This is great, thank you so much")["sentiment"]["score"] > 0.3
    assert analyze_text("I am really frustrated, nothing works and it is broken")["sentiment"]["score"] < -0.3
    assert analyze_text("it is not good")["sentiment"]["score"] < 0


# ----------------------------------------------------------------------- store
def test_profile_valid_and_atomic_revisions(tmp_path):
    st = SpeechProfileStore(tmp_path)
    p = st.create("Caller 42/abc")
    assert validate(p) == []
    assert st.path_for("Caller 42/abc").name == "Caller_42_abc.speech.json"
    with st.edit("Caller 42/abc", agent_id="a", action="x") as q:
        q["speaker"]["notes"] = "hello"
    assert st.load("Caller 42/abc")["revision"] == 1
    with pytest.raises(ConcurrencyConflict):
        with st.edit("Caller 42/abc", agent_id="a", action="x", expected_revision=0):
            pass
    # failed edit body -> nothing written
    with pytest.raises(RuntimeError):
        with st.edit("Caller 42/abc", agent_id="a", action="x") as q:
            q["speaker"]["notes"] = "lost"
            raise RuntimeError
    assert st.load("Caller 42/abc")["speaker"]["notes"] == "hello"


def test_concurrent_writers_do_not_lose_updates(tmp_path):
    eng_store = SpeechProfileStore(tmp_path, lock_timeout_s=30)
    eng_store.create("s")
    eng = [SpeechProfileEngine(eng_store, agent_id=f"agent{i}") for i in range(4)]

    def work(e):
        for k in range(5):
            e.add_observation("s", category="other", observation=f"{e.agent_id}-{k}", confidence=.5)

    th = [threading.Thread(target=work, args=(e,)) for e in eng]
    [t.start() for t in th]
    [t.join() for t in th]
    p = eng_store.load("s")
    assert len(p["observations"]) == 20 and p["revision"] == 20


# ----------------------------------------------------------------------- engine end-to-end
def _session(eng, spk, n=4, f0=150, rate_words=12, jit=0.0, text=None, seed=0):
    sid = eng.start_session(spk, channel="test")
    for i in range(n):
        x = np.concatenate([sil(.3), voice(f0, 1.5, jit=jit, glide=-2 if i % 2 else 3, seed=seed + i), sil(.5),
                            voice(f0 * 1.05, 1.5, jit=jit, seed=seed + 50 + i), sil(.3)])
        words = text or " ".join(["hello", "I", "would", "like", "to", "check", "my", "order", "status", "please",
                                  "thank", "you", "very", "much", "today"][:rate_words])
        eng.analyze_utterance(spk, session_id=sid, samples=x, sample_rate=SR, transcript=words,
                              response_latency_s=0.6)
    return sid, eng.end_session(spk, sid)


def test_full_lifecycle_baseline_trends_guidance(tmp_path):
    eng = SpeechProfileEngine(tmp_path, agent_id="t")
    eng.ensure_profile("u1", display_name="Asha")
    eng.set_consent("u1", health_indicators=True, recording=True)
    for k in range(3):
        _, summ = _session(eng, "u1", n=5, seed=k * 10)
        assert summ["utterance_count"] == 5
    p = eng.get_profile("u1")
    assert validate(p) == []
    assert p["baseline"]["status"] == "established"
    assert p["baseline"]["metrics"]["f0_mean_hz"]["n"] == 15
    assert len(p["trends"]["f0_mean_hz"]) == 3
    assert p["assessments"]["communication_skills"]["scores"]["overall"] is not None
    s = eng.get_summary("u1")
    assert "SPEECH PROFILE for Asha" in s["prompt_context"]
    tr = eng.get_trend("u1", "f0_mean_hz")
    assert tr["direction"] in {"stable", "increasing", "decreasing"}

    # a much higher, rougher voice afterwards should deviate from baseline and raise arousal
    sid = eng.start_session("u1")
    x = np.concatenate([sil(.3), voice(230, 2.5, jit=0.03, amp=0.9, glide=6), sil(.3)])
    r = eng.analyze_utterance("u1", session_id=sid, samples=x, sample_rate=SR,
                              transcript="This is ridiculous, I am so angry, nothing works!")
    assert "f0_mean_hz" in r["baseline_deviations"]
    assert r["emotion"]["arousal"] > 0.3 and r["emotion"]["valence"] < 0
    assert r["emotion"]["label"] in {"angry", "frustrated"}
    assert any(a["priority"] == "high" for a in r["guidance"]["adaptations"])


def test_indicators_need_consent_and_persist(tmp_path):
    eng = SpeechProfileEngine(tmp_path, agent_id="t")
    stutter = "I w-w-want to to go to the the st-st-store and b-b-buy some some bread for for my my family today"
    _session(eng, "u2", text=stutter)
    assert eng.get_profile("u2")["assessments"]["speech_indicators"] == []  # no consent -> nothing
    eng.set_consent("u2", health_indicators=True)
    for k in range(3):
        _session(eng, "u2", text=stutter, seed=100 + k)
    ind = {i["indicator"]: i for i in eng.get_profile("u2")["assessments"]["speech_indicators"]}
    assert ind["stuttering_like_disfluencies"]["status"] == "persistent"
    assert "non-diagnostic" in ind["stuttering_like_disfluencies"]["disclaimer"]
    g = eng.get_profile("u2")["agent_guidance"]["adaptations"]
    assert any("finish the person's words" in a["instruction"] for a in g)


def test_consent_blocks_analysis_and_transcripts(tmp_path):
    eng = SpeechProfileEngine(tmp_path)
    eng.set_consent("u3", voice_analysis=False)
    with pytest.raises(ConsentError):
        eng.analyze_utterance("u3", transcript="hello there")
    eng.set_consent("u3", voice_analysis=True, store_transcripts=False)
    eng.analyze_utterance("u3", transcript="my secret account is 1234")
    assert "secret" not in json.dumps(eng.get_profile("u3"))


# ----------------------------------------------------------------------- tools
def test_tool_dispatch_and_formats(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    save_wav(audio_dir / "t.wav", np.concatenate([sil(.2), voice(180, 1.5), sil(.2)]), SR)
    tools = SpeechProfileTools(tmp_path / "profiles", agent_id="llm", audio_root=audio_dir)
    names = [t["name"] for t in tools.definitions("anthropic")]
    assert "paralingo_get_summary" in names and all("input_schema" in t for t in tools.definitions())
    assert tools.definitions("openai")[0]["type"] == "function"

    sid = json.loads(tools.execute("paralingo_start_session", {"speaker_id": "p"}))["result"]["session_id"]
    r = json.loads(tools.execute("paralingo_analyze_utterance",
                                 {"speaker_id": "p", "session_id": sid, "audio_path": "t.wav",
                                  "transcript": "hi, I need to change my address please",
                                  "external": {"asr_confidence": 0.9}}))
    assert r["ok"] and r["result"]["key_metrics"]["f0_mean_hz"] > 170
    bad = json.loads(tools.execute("paralingo_analyze_utterance",
                                   {"speaker_id": "p", "audio_path": str(tmp_path.parent / "x.wav")}))
    assert not bad["ok"] and "outside" in bad["error"]
    r = json.loads(tools.execute("paralingo_update_accent",
                                 {"speaker_id": "p", "label": "Indian English", "confidence": .7,
                                  "markers": [{"marker": "retroflex /t/"}]}))
    assert r["result"]["current_estimate"]["label"] == "Indian English"
    r = json.loads(tools.execute("paralingo_add_observation",
                                 {"speaker_id": "p", "category": "communication", "confidence": .6,
                                  "observation": "Prefers short, direct answers."}))
    assert r["ok"]
    assert json.loads(tools.execute("paralingo_end_session", {"speaker_id": "p"}))["ok"]
    sec = json.loads(tools.execute("paralingo_read_section", {"speaker_id": "p", "section": "sessions"}))
    assert sec["result"][0]["utterances"][0]["acoustics"]["pitch"]["f0_mean_hz"] > 170
    assert not json.loads(tools.execute("nope", {}))["ok"]

    llm_only = SpeechProfileTools(tmp_path / "profiles", include_pipeline_tools=False)
    assert "paralingo_analyze_utterance" not in [t["name"] for t in llm_only.definitions()]


# ----------------------------------------------------------------------- regressions found by validation on real speech
def test_parabolic_never_jumps_off_a_non_peak():
    from paralingo.audio import _parabolic
    # rising edge (window boundary, not a true peak): the old formula returned an offset of ~-5000 samples
    loc, val = _parabolic(np.array([0.0, 0.5, 1.0001]), 1)
    assert loc == 1.0 and val == 0.5
    loc, _ = _parabolic(np.array([0.2, 1.0, 0.6]), 1)  # true peak: bounded sub-sample shift
    assert 0.5 <= loc <= 1.5


def test_48khz_noisy_breathy_voice_terminates():
    import time as _t
    sr = 48000
    rng = np.random.default_rng(5)
    t = np.arange(int(2.0 * sr)) / sr
    f0 = 220 * (1 + 0.15 * np.sin(2 * np.pi * 3 * t))          # wide vibrato-like F0 movement
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = 0.3 * (np.sin(ph) + 0.4 * np.sin(2 * ph)) * (1 + 0.5 * rng.standard_normal(len(t)) ** 2 * 0.1)
    x += 0.05 * rng.standard_normal(len(t))                      # breathy noise -> flat, ragged peaks
    x = np.concatenate([np.zeros(sr // 4), x, np.zeros(sr // 4)])
    start = _t.perf_counter()
    r = analyze_audio(x, sr)
    assert _t.perf_counter() - start < 10
    assert r["voice_quality"]["jitter_local_pct"] is None or np.isfinite(r["voice_quality"]["jitter_local_pct"])


def test_voice_strain_is_judged_against_own_baseline(tmp_path):
    eng = SpeechProfileEngine(tmp_path, agent_id="t")
    eng.set_consent("v", health_indicators=True)
    _session(eng, "v", n=5, seed=1)               # one session of the speaker's normal voice
    ind = {i["indicator"] for i in eng.get_profile("v")["assessments"]["speech_indicators"]}
    assert "voice_quality_strain" not in ind      # no absolute cut-offs: a first session can't trigger it
    for k in range(2):
        _session(eng, "v", n=5, seed=10 + k)
    # now a clearly rougher, noisier voice than usual
    sid = eng.start_session("v")
    for i in range(5):
        x = np.concatenate([sil(.3), voice(150, 2.0, jit=0.03, shim=0.25, seed=200 + i), sil(.3)])
        x = x + np.random.default_rng(i).normal(0, 0.02, len(x))
        eng.analyze_utterance("v", session_id=sid, samples=x, sample_rate=SR,
                              transcript="hello I would like to check my order status please thank you")
    eng.end_session("v", sid)
    ind = {i["indicator"]: i for i in eng.get_profile("v")["assessments"]["speech_indicators"]}
    assert "voice_quality_strain" in ind
    assert all("baseline" in e for e in ind["voice_quality_strain"]["evidence"])
