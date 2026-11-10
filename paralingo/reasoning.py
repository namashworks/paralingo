# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Reasoning layer: personal baseline, emotion, communication skills, speaking style,
speech-health indicators and agent guidance.

Design principles
-----------------
* **Relative, not absolute.** People differ hugely (a "loud" or "high" voice for one
  person is normal for another), so arousal / change detection uses z-scores against
  the speaker's own baseline whenever it exists, with wide population priors otherwise.
* **Explainable.** Every inference returns the cues that produced it.
* **Humble.** Confidence is capped, poor audio lowers it, and health indicators are
  screening *signals* that require persistence across sessions - never diagnoses.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any

DISCLAIMER = ("Automated, non-diagnostic screening signal derived from speech acoustics/transcripts. "
              "It can be caused by recording conditions, language background, mood or context. "
              "Do not tell the speaker they have a condition; suggest a qualified professional only if appropriate.")

# --------------------------------------------------------------------------- metrics
# Flat metric name -> path inside an utterance record
METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "f0_mean_hz": ("acoustics", "pitch", "f0_mean_hz"),
    "f0_std_semitones": ("acoustics", "pitch", "f0_std_semitones"),
    "f0_range_semitones": ("acoustics", "pitch", "f0_range_semitones"),
    "rms_dbfs_mean": ("acoustics", "energy", "rms_dbfs_mean"),
    "hf_energy_ratio_db": ("acoustics", "spectral", "hf_energy_ratio_db"),
    "jitter_local_pct": ("acoustics", "voice_quality", "jitter_local_pct"),
    "shimmer_local_pct": ("acoustics", "voice_quality", "shimmer_local_pct"),
    "hnr_db": ("acoustics", "voice_quality", "hnr_db"),
    "words_per_minute": ("timing", "words_per_minute"),
    "articulation_rate_sps": ("timing", "articulation_rate_sps"),
    "pause_ratio": ("timing", "pauses", "pause_ratio"),
    "mean_pause_s": ("timing", "pauses", "mean_s"),
    "response_latency_s": ("timing", "response_latency_s"),
    "filler_rate_per_100w": ("fluency", "filler_rate_per_100w"),
    "stuttering_like_per_100_syllables": ("fluency", "stuttering_like_per_100_syllables"),
    "total_disfluency_per_100w": ("fluency", "total_disfluency_per_100w"),
    "type_token_ratio": ("lexical", "type_token_ratio"),
    "hedge_rate_per_100w": ("lexical", "hedge_rate_per_100w"),
    "words_per_turn": ("lexical", "word_count"),
    "sentiment": ("sentiment", "score"),
    "valence": ("emotion", "valence"),
    "arousal": ("emotion", "arousal"),
}
ACOUSTIC_METRICS = {"f0_mean_hz", "f0_std_semitones", "f0_range_semitones", "rms_dbfs_mean", "hf_energy_ratio_db",
                    "jitter_local_pct", "shimmer_local_pct", "hnr_db"}
# metrics that are meaningless for very short utterances
MIN_WORDS_FOR = {"type_token_ratio": 15, "words_per_minute": 3, "articulation_rate_sps": 3,
                 "filler_rate_per_100w": 5, "total_disfluency_per_100w": 5, "hedge_rate_per_100w": 5,
                 "stuttering_like_per_100_syllables": 5}
# minimum std used for z-scores (prevents tiny baseline variance from exploding z)
STD_FLOOR = {"f0_mean_hz": 6.0, "f0_std_semitones": 0.4, "f0_range_semitones": 1.0, "rms_dbfs_mean": 1.5,
             "hf_energy_ratio_db": 1.5, "jitter_local_pct": 0.2, "shimmer_local_pct": 0.8, "hnr_db": 1.5,
             "words_per_minute": 12.0, "articulation_rate_sps": 0.4, "pause_ratio": 0.05, "mean_pause_s": 0.1,
             "response_latency_s": 0.3, "filler_rate_per_100w": 1.0, "stuttering_like_per_100_syllables": 0.7,
             "total_disfluency_per_100w": 1.5, "type_token_ratio": 0.05, "hedge_rate_per_100w": 1.0,
             "words_per_turn": 4.0, "sentiment": 0.1, "valence": 0.1, "arousal": 0.1}
# wide population priors (mean, std) used before a personal baseline exists; pitch & loudness
# are deliberately absent - they depend on the person / microphone.
POPULATION = {"words_per_minute": (150.0, 30.0), "articulation_rate_sps": (4.8, 1.0), "f0_std_semitones": (2.5, 1.0),
              "pause_ratio": (0.2, 0.1), "filler_rate_per_100w": (3.0, 3.0), "f0_range_semitones": (7.0, 3.0)}

TREND_METRICS = ["f0_mean_hz", "f0_std_semitones", "rms_dbfs_mean", "jitter_local_pct", "shimmer_local_pct", "hnr_db",
                 "words_per_minute", "articulation_rate_sps", "pause_ratio", "filler_rate_per_100w",
                 "stuttering_like_per_100_syllables", "type_token_ratio", "hedge_rate_per_100w", "valence", "arousal",
                 "response_latency_s"]


def get_path(d: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def utterance_metrics(u: dict[str, Any]) -> dict[str, float]:
    """Flatten an utterance record into {metric: value}, dropping unreliable values."""
    words = get_path(u, ("lexical", "word_count")) or 0
    poor_audio = get_path(u, ("acoustics", "audio_quality", "quality")) == "poor"
    out: dict[str, float] = {}
    for m, p in METRIC_PATHS.items():
        v = get_path(u, p)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        if m in MIN_WORDS_FOR and words < MIN_WORDS_FOR[m]:
            continue
        if m in ACOUSTIC_METRICS and poor_audio:
            continue
        if m == "words_per_turn" and not words:
            continue
        out[m] = float(v)
    return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = {k for r in rows for k in r}
    return {k: round(sum(r[k] for r in rows if k in r) / sum(1 for r in rows if k in r), 4) for k in sorted(keys)}


# --------------------------------------------------------------------------- baseline
def baseline_update(baseline: dict[str, Any], metrics: dict[str, float]) -> None:
    """Welford online update of mean / variance per metric."""
    for k, x in metrics.items():
        s = baseline["metrics"].setdefault(k, {"n": 0, "mean": None, "std": None, "m2": 0.0, "min": None, "max": None})
        n = s["n"] + 1
        mean = x if s["mean"] is None else s["mean"]
        delta = x - mean
        mean = mean + delta / n
        m2 = (s.get("m2") or 0.0) + delta * (x - mean)
        s.update(n=n, mean=round(mean, 5), m2=round(m2, 6),
                 std=round(math.sqrt(m2 / (n - 1)), 5) if n > 1 else 0.0,
                 min=x if s["min"] is None else min(s["min"], x),
                 max=x if s["max"] is None else max(s["max"], x))


def baseline_status(baseline: dict[str, Any]) -> str:
    if baseline["sessions_used"] >= 3 and baseline["utterances_used"] >= 15:
        return "established"
    if baseline["utterances_used"] >= 5:
        return "provisional"
    return "insufficient_data"


def zscore(metric: str, value: float | None, baseline: dict[str, Any]) -> tuple[float | None, str]:
    """z of value vs personal baseline (if >=5 samples) else population prior. Returns (z, source)."""
    if value is None:
        return None, "none"
    s = baseline.get("metrics", {}).get(metric)
    if s and s["n"] >= 5 and s["mean"] is not None:
        sd = max(s["std"] or 0.0, STD_FLOOR.get(metric, 1e-6))
        return (value - s["mean"]) / sd, "personal"
    if metric in POPULATION:
        mu, sd = POPULATION[metric]
        return (value - mu) / sd, "population"
    return None, "none"


def deviations(metrics: dict[str, float], baseline: dict[str, Any], threshold: float = 2.0) -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        s = baseline.get("metrics", {}).get(k)
        if not s or s["n"] < 5:
            continue
        z, _ = zscore(k, v, baseline)
        if z is not None and abs(z) >= threshold:
            out[k] = round(z, 2)
    return out


# --------------------------------------------------------------------------- emotion
_WORRY = {"worried", "anxious", "scared", "afraid", "nervous", "stressed", "panic", "panicking"}
_TIRED = {"tired", "exhausted", "sleepy", "drained", "worn"}


def estimate_emotion(u: dict[str, Any], baseline: dict[str, Any], tokens: list[str] | None = None) -> dict[str, Any]:
    """Dimensional (arousal / valence) emotion estimate from prosody + words, mapped to a label."""
    m = utterance_metrics(u)
    cues: list[str] = []
    num = den = 0.0
    for metric, w, need_personal in [("f0_mean_hz", 1.0, True), ("rms_dbfs_mean", 1.0, True),
                                     ("hf_energy_ratio_db", 0.6, True), ("f0_std_semitones", 0.6, False),
                                     ("articulation_rate_sps", 0.6, False), ("pause_ratio", -0.4, False)]:
        z, src = zscore(metric, m.get(metric), baseline)
        if z is None or (need_personal and src != "personal"):
            continue
        z = max(-4.0, min(4.0, z))
        num += w * z
        den += abs(w)
        if abs(z) >= 1.5:
            cues.append(f"{metric} {'high' if z > 0 else 'low'} (z={z:+.1f}, {src})")
    arousal = math.tanh(0.6 * num / den) if den else 0.0

    sent = get_path(u, ("sentiment", "score"))
    valence = 0.0
    if sent is not None:
        valence = float(sent)
        if abs(sent) >= 0.2:
            cues.append(f"word sentiment {sent:+.2f}")
    zf, _ = zscore("f0_std_semitones", m.get("f0_std_semitones"), baseline)
    ze, src_e = zscore("rms_dbfs_mean", m.get("rms_dbfs_mean"), baseline)
    if zf is not None and zf < -1 and src_e == "personal" and ze is not None and ze < -1:
        valence -= 0.15
        cues.append("flat, quiet delivery")

    method = "heuristic-prosody+lexicon-v1"
    ext = get_path(u, ("external", "emotion"))
    if isinstance(ext, dict) and ext.get("label"):
        # agent supplied a speech-emotion-recognition model result -> blend
        ec = float(ext.get("confidence", 0.5))
        if "arousal" in ext:
            arousal = (1 - ec) * arousal + ec * float(ext["arousal"])
        if "valence" in ext:
            valence = (1 - ec) * valence + ec * float(ext["valence"])
        method += f"+external:{ext.get('model', 'ser')}"
        cues.append(f"external model says '{ext['label']}' ({ec:.2f})")

    arousal = max(-1.0, min(1.0, arousal))
    valence = max(-1.0, min(1.0, valence))
    toks = set(tokens or [])
    if isinstance(ext, dict) and ext.get("label") and float(ext.get("confidence", 0)) >= 0.7:
        label = str(ext["label"]).lower()
    elif abs(arousal) < 0.25 and abs(valence) < 0.2:
        label = "neutral"
    elif valence <= -0.2:
        if toks & _WORRY:
            label = "anxious"
        elif toks & _TIRED and arousal < 0.25:
            label = "tired"
        elif arousal >= 0.55:
            label = "angry"
        elif arousal >= 0.25:
            label = "frustrated"
        elif arousal <= -0.25:
            label = "sad"
        else:
            label = "displeased"
    elif valence >= 0.2:
        label = "excited" if arousal >= 0.55 else "happy" if arousal >= 0.25 else "calm" if arousal <= -0.25 else "pleased"
    else:
        label = "energetic" if arousal >= 0.25 else "subdued"

    conf = 0.3
    conf += 0.15 if baseline.get("status") == "established" else 0.05 if baseline.get("status") == "provisional" else 0
    conf += 0.1 if den and sent is not None else 0
    conf += min(0.1, 0.03 * len(cues))
    if get_path(u, ("acoustics", "audio_quality", "quality")) == "poor":
        conf *= 0.6
    if isinstance(ext, dict) and ext.get("label"):
        conf = max(conf, min(0.9, float(ext.get("confidence", 0.5))))
    return {"label": label, "arousal": round(arousal, 3), "valence": round(valence, 3),
            "confidence": round(min(conf, 0.9), 2), "method": method, "cues": cues}


# --------------------------------------------------------------------------- skills
def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _band(v: float, lo: float, hi: float, slope: float) -> float:
    """100 inside [lo, hi], decreasing by `slope` points per unit outside."""
    if v < lo:
        return _clamp(100 - (lo - v) * slope)
    if v > hi:
        return _clamp(100 - (v - hi) * slope)
    return 100.0


def score_skills(u: dict[str, Any]) -> dict[str, float | None]:
    """0-100 communication-skill scores for one utterance (None = not enough evidence)."""
    m = utterance_metrics(u)
    s: dict[str, float | None] = {}
    words = get_path(u, ("lexical", "word_count")) or 0

    if "total_disfluency_per_100w" in m:
        f = 100 - 6 * m["total_disfluency_per_100w"]
        f -= 60 * max(0.0, m.get("pause_ratio", 0) - 0.3)
        f -= 8 * (get_path(u, ("timing", "pauses", "long_count")) or 0)
        s["fluency"] = round(_clamp(f), 1)
    s["pace"] = round(_band(m["words_per_minute"], 120, 175, 1.5), 1) if "words_per_minute" in m else None

    clar = []
    if "hnr_db" in m:
        clar.append(_clamp((m["hnr_db"] - 5) / 15 * 100))
    if "articulation_rate_sps" in m:
        clar.append(_band(m["articulation_rate_sps"], 3.5, 6.0, 30))
    asr = get_path(u, ("external", "asr_confidence"))
    if asr is not None:
        clar.extend([float(asr) * 100] * 2)  # strongest clarity signal when available
    incompl = get_path(u, ("fluency", "incomplete_words")) or 0
    s["clarity"] = round(_clamp(sum(clar) / len(clar) - 5 * incompl), 1) if clar else None

    s["expressiveness"] = round(_band(m["f0_std_semitones"], 2.0, 5.5, 45), 1) if "f0_std_semitones" in m else None

    if words >= 5:
        c = 100 - 6 * m.get("hedge_rate_per_100w", 0) - 3 * m.get("filler_rate_per_100w", 0)
        contour = get_path(u, ("acoustics", "pitch", "terminal_contour"))
        if contour == "falling" and "?" not in (u.get("transcript") or ""):
            c += 5
        s["confidence"] = round(_clamp(c), 1)
    else:
        s["confidence"] = None

    s["vocabulary"] = round(_clamp((m["type_token_ratio"] - 0.35) / 0.45 * 100), 1) if "type_token_ratio" in m else None
    s["responsiveness"] = (round(_band(m["response_latency_s"], 0.2, 1.5, 30), 1)
                           if "response_latency_s" in m else None)
    vals = [v for v in s.values() if v is not None]
    s["overall"] = round(sum(vals) / len(vals), 1) if vals else None
    return s


# --------------------------------------------------------------------------- style
def speaking_style(agg: dict[str, float], baseline: dict[str, Any], latency: float | None) -> dict[str, Any]:
    def cat(v, edges, labels):
        if v is None:
            return "unknown"
        for e, lab in zip(edges, labels):
            if v < e:
                return lab
        return labels[-1]

    z_loud, src = zscore("rms_dbfs_mean", agg.get("rms_dbfs_mean"), baseline)
    loud = "unknown" if src != "personal" else "soft" if z_loud < -1 else "loud" if z_loud > 1 else "normal"
    return {
        "pace": cat(agg.get("words_per_minute"), [100, 125, 170, 200], ["very_slow", "slow", "moderate", "fast", "very_fast"]),
        "pitch_expressiveness": cat(agg.get("f0_std_semitones"), [1.0, 2.0, 4.0], ["monotone", "low", "moderate", "expressive"]),
        "loudness": loud,
        "pausing": cat(agg.get("pause_ratio"), [0.1, 0.3], ["sparse", "typical", "frequent"]),
        "verbosity": cat(agg.get("words_per_turn"), [8, 35], ["terse", "moderate", "elaborate"]),
        "directness": cat(agg.get("hedge_rate_per_100w"), [1.0, 4.0], ["direct", "balanced", "hedging"]),
        "mean_words_per_turn": round(agg["words_per_turn"], 1) if "words_per_turn" in agg else None,
        "typical_response_latency_s": round(latency, 2) if latency is not None else None,
    }


# --------------------------------------------------------------------------- indicators
def evaluate_indicators(session: dict[str, Any], agg: dict[str, float], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Session-level, non-diagnostic screening checks. Returns triggered indicators with evidence."""
    utts = session["utterances"]
    speech_s = sum(get_path(u, ("acoustics", "speech_s")) or get_path(u, ("timing", "speech_s")) or 0 for u in utts)
    syllables = sum(get_path(u, ("lexical", "syllable_count")) or 0 for u in utts)
    words = sum(get_path(u, ("lexical", "word_count")) or 0 for u in utts)
    qualities = [get_path(u, ("acoustics", "audio_quality", "quality")) for u in utts]
    good_audio = qualities.count("poor") <= len(qualities) * 0.3
    enough = len(utts) >= 3 and (speech_s >= 15 or words >= 60)
    hits: list[dict[str, Any]] = []
    if not enough:
        return hits

    def add(name, desc, evidence, confounders):
        hits.append({"indicator": name, "description": desc, "evidence": evidence, "confounders": confounders})

    sld = agg.get("stuttering_like_per_100_syllables")
    if sld is not None and sld >= 3.0 and syllables >= 80:
        add("stuttering_like_disfluencies",
            "Part-word / whole-word repetitions and prolongations at or above ~3 per 100 syllables (common screening cut-off).",
            [f"SLD = {sld:.1f} per 100 syllables over {syllables} syllables"],
            ["ASR may insert or drop repetitions", "situational nervousness", "speaking a second language"])

    # Voice strain is judged ONLY against the person's own baseline. Absolute clinical cut-offs
    # (jitter > 1.04 %, shimmer > 3.81 %, HNR < 20 dB) come from sustained vowels; on connected,
    # expressive speech Praat itself exceeds all three for 100 % of healthy RAVDESS speakers (validation/).
    # 3 SD (not 2) because emotional speech alone shifts these measures: 2 SD fired on 23.5 % of healthy
    # single clips, 3 SD on 8 % (calm speech 2 %).
    if good_audio:
        vq = []
        for metric, sign, label in [("jitter_local_pct", 1, "jitter"), ("shimmer_local_pct", 1, "shimmer"),
                                    ("hnr_db", -1, "HNR")]:
            z, src = zscore(metric, agg.get(metric), baseline)
            if src == "personal" and z is not None and sign * z >= 3.0:
                vq.append(f"{label} {agg[metric]:.2f} ({'+' if z > 0 else ''}{z:.1f} SD vs this person's baseline)")
        if len(vq) >= 2:
            add("voice_quality_strain",
                "Voice perturbation and noise clearly worse than this person's own usual voice "
                "(possible hoarseness, breathiness or strain).",
                vq, ["background noise, a different microphone or phone/VoIP codec",
                     "cold, fatigue or voice overuse", "emotional or shouted speech"])

    ar = agg.get("articulation_rate_sps")
    z_wpm, src = zscore("words_per_minute", agg.get("words_per_minute"), baseline)
    if (ar is not None and ar < 3.0) or (src == "personal" and z_wpm is not None and z_wpm < -2):
        add("slowed_speech", "Articulation / speaking rate well below typical or below this person's own baseline.",
            [f"articulation rate {ar} syll/s" if ar is not None else "", f"wpm z={z_wpm:+.1f} ({src})" if z_wpm is not None else ""],
            ["careful speech for clarity", "second-language speaker", "thinking while talking", "fatigue"])

    td = agg.get("total_disfluency_per_100w")
    if ar is not None and ar > 6.5 and td is not None and td > 8:
        add("rapid_cluttered_speech", "Very fast articulation combined with many disfluencies.",
            [f"articulation rate {ar:.1f} syll/s", f"disfluencies {td:.1f}/100 words"],
            ["excitement or time pressure"])

    fst = agg.get("f0_std_semitones")
    z_f, src_f = zscore("f0_std_semitones", fst, baseline)
    if good_audio and fst is not None and (fst < 1.0 or (src_f == "personal" and z_f is not None and z_f < -2)):
        add("reduced_prosody", "Unusually flat intonation (little pitch variation).",
            [f"pitch variability {fst:.2f} semitones" + (f" (z={z_f:+.1f} vs baseline)" if src_f == "personal" else "")],
            ["reading aloud / scripted speech", "tiredness", "cultural or personal speaking style"])

    filler = agg.get("filler_rate_per_100w", 0)
    vague = sum(get_path(u, ("lexical", "vague_word_count")) or 0 for u in utts)
    long_p = sum(get_path(u, ("timing", "pauses", "long_count")) or 0 for u in utts)
    mins = max(speech_s / 60.0, 0.25)
    if filler > 6 and long_p / mins >= 1.5 and vague >= 3:
        add("word_finding_difficulty", "Frequent fillers, long mid-utterance pauses and vague placeholder words.",
            [f"fillers {filler:.1f}/100 words", f"{long_p} long pauses (~{long_p / mins:.1f}/min)", f"{vague} vague words"],
            ["unfamiliar topic", "second-language speaker", "multitasking during the call"])

    z_l, src_l = zscore("rms_dbfs_mean", agg.get("rms_dbfs_mean"), baseline)
    if src_l == "personal" and z_l is not None and z_l < -2:
        add("reduced_loudness", "Speaking markedly more quietly than usual for this person.",
            [f"loudness z={z_l:+.1f} vs baseline"], ["microphone distance / device change", "quiet environment"])

    if baseline.get("status") == "established":
        dev = deviations({k: v for k, v in agg.items() if k in TREND_METRICS}, baseline, 2.5)
        if len(dev) >= 2:
            add("significant_change_from_baseline", "Several speech measures differ strongly from this person's norm.",
                [f"{k} z={v:+.1f}" for k, v in dev.items()], ["different device / channel", "different context of call"])
    for h in hits:
        h["evidence"] = [e for e in h["evidence"] if e]
    return hits


def merge_indicators(profile: dict[str, Any], session_id: str, hits: list[dict[str, Any]], when: str) -> None:
    """Update the profile's indicator list with this session's hits and recompute status."""
    store = {i["indicator"]: i for i in profile["assessments"]["speech_indicators"]}
    hit_names = {h["indicator"] for h in hits}
    for h in hits:
        cur = store.get(h["indicator"]) or {
            "indicator": h["indicator"], "status": "watch", "occurrences": 0, "sessions_flagged": [],
            "first_seen": when, "disclaimer": DISCLAIMER}
        cur.update(description=h["description"], evidence=h["evidence"], confounders=h["confounders"], last_seen=when)
        cur["occurrences"] += 1
        cur["sessions_flagged"] = (cur["sessions_flagged"] + [session_id])[-20:]
        store[h["indicator"]] = cur
    evaluated = [s["session_id"] for s in profile["sessions"]
                 if s.get("summary") and s["summary"].get("indicators_evaluated")][-5:]
    for name, cur in store.items():
        recent = [sid for sid in evaluated if sid in cur["sessions_flagged"]]
        if name in hit_names:
            cur["status"] = "persistent" if len(recent) >= 3 else "watch"
        elif evaluated[-3:] and not any(sid in cur["sessions_flagged"] for sid in evaluated[-3:]):
            cur["status"] = "resolved"
    profile["assessments"]["speech_indicators"] = list(store.values())


# --------------------------------------------------------------------------- guidance
def build_guidance(profile: dict[str, Any], recent_utts: list[dict[str, Any]]) -> dict[str, Any]:
    a = profile["assessments"]
    style = a["speaking_style"]
    adapt: list[dict[str, str]] = []

    def add(instr, reason, prio="medium"):
        adapt.append({"instruction": instr, "reason": reason, "priority": prio})

    rows = [utterance_metrics(u) for u in recent_utts]
    wpms = [r["words_per_minute"] for r in rows if "words_per_minute" in r]
    rate = round(max(110.0, min(170.0, median(wpms) * 0.95)), 0) if wpms else None

    pauses = sorted(p[1] - p[0] for u in recent_utts for p in (get_path(u, ("timing", "pauses", "intervals_s")) or [])
                    if p and p[0] is not None and p[1] is not None)
    eot = None
    if len(pauses) >= 3:
        p90 = pauses[min(len(pauses) - 1, int(0.9 * len(pauses)))]
        eot = int(max(600, min(2500, p90 * 1000 + 250)))

    if style["pace"] in ("very_slow", "slow"):
        add("Speak slowly, use short sentences and one idea at a time.", f"Speaker's own pace is {style['pace']}.")
    elif style["pace"] in ("fast", "very_fast"):
        add("Keep responses brisk and to the point.", f"Speaker talks at a {style['pace']} pace.", "low")
    if style["pausing"] == "frequent" or (eot and eot >= 1200):
        add("Wait longer before taking the turn; never cut the speaker off mid-thought.",
            "Speaker pauses often / for long stretches within turns.", "high")
    neg_set = {"angry", "frustrated", "anxious", "sad", "displeased", "tired"}
    emo = a["emotional_state"].get("current") or {}
    recent_neg = [lab for lab in a["emotional_state"].get("recent_labels", [])[-5:] if lab in neg_set]
    if emo.get("label") in neg_set and emo.get("confidence", 0) >= 0.35:
        add("Acknowledge the person's feelings, lower your tempo and use a warm, calm tone.",
            f"Current emotion estimate: {emo['label']} (confidence {emo['confidence']}).", "high")
    elif len(recent_neg) >= 2:
        add("Open by acknowledging any earlier frustration before moving to the task; keep a calm tone.",
            f"{len(recent_neg)} of the last 5 turns read as {', '.join(sorted(set(recent_neg)))}.", "high")
    if style["directness"] == "hedging":
        add("Offer reassurance and ask one clear question at a time.", "Speaker hedges a lot (uncertainty).")
    if style["verbosity"] == "terse":
        add("Ask open-ended questions to draw out detail.", "Speaker gives short answers.", "low")
    elif style["verbosity"] == "elaborate":
        add("Summarise back periodically to keep the conversation focused.", "Speaker gives long turns.", "low")
    acc = profile.get("accent", {})
    intel = acc.get("intelligibility")
    if (intel is not None and intel < 0.8) or len(acc.get("phonetic_markers", [])) >= 3:
        add("Read back critical details (names, numbers, dates) to confirm them.",
            "Accent features / lower ASR intelligibility noted - prevents costly misunderstandings.", "medium")
    names = {i["indicator"]: i for i in a["speech_indicators"] if i["status"] != "resolved"}
    if "stuttering_like_disfluencies" in names:
        add("Give extra time; never finish the person's words or comment on their fluency.",
            "Stuttering-like disfluencies observed.", "high")
    if "voice_quality_strain" in names or "reduced_loudness" in names:
        add("Keep the exchange efficient and offer a text/SMS alternative.", "Voice may be strained or quiet.")
    if "word_finding_difficulty" in names:
        add("Be patient with searching for words; offer gentle options rather than guessing.",
            "Frequent word-finding pauses observed.", "high")
    qualities = [get_path(u, ("acoustics", "audio_quality", "quality")) for u in recent_utts]
    if qualities and qualities.count("poor") >= max(1, len(qualities) // 2):
        add("Ask the person to repeat instead of guessing; suggest a quieter spot if possible.",
            "Recent audio quality is poor.", "medium")

    prio = {"high": 0, "medium": 1, "low": 2}
    adapt.sort(key=lambda x: prio[x["priority"]])
    return {"recommended_speaking_rate_wpm": rate, "end_of_turn_silence_ms": eot, "adaptations": adapt}
