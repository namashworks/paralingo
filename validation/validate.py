# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Validate paralingo on real recordings.

What it measures
----------------
1. **Agreement with Praat** (the reference tool in speech science, via parselmouth) on every file:
   frame-level pitch (gross pitch error, voicing agreement), and utterance-level F0, jitter, shimmer,
   HNR and pauses. Praat's CPPS is also recorded for reference.
2. **Arousal accuracy** on labelled emotional speech (RAVDESS file naming), using a per-speaker
   baseline built from that speaker's neutral and calm clips, exactly as the product does.
3. **Phone-line robustness**: each clip is passed through a simulated telephone channel
   (300-3400 Hz, 8 kHz, G.711 mu-law) and re-analysed.
4. **Indicator false alarms**: how often the voice-strain and flat-prosody rules fire on clips from
   speakers with presumably healthy voices.

    pip install -e ".[dev,eval]"
    python validation/validate.py --data path/to/Audio_Speech_Actors_01-24 --out validation/results

Works on any folder of WAV files; emotion scoring runs only when files use RAVDESS naming.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paralingo import __version__  # noqa: E402
from paralingo import reasoning as R  # noqa: E402
from paralingo.audio import analyze_audio, load_wav  # noqa: E402
from paralingo.language import analyze_text, tokenize  # noqa: E402

F0_MIN, F0_MAX = 60.0, 500.0
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
            "07": "disgust", "08": "surprised"}
HIGH_AROUSAL = {"happy", "angry", "fearful", "surprised"}
LOW_AROUSAL = {"neutral", "calm", "sad"}
STATEMENTS = {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"}


# --------------------------------------------------------------------------- helpers
def parse_ravdess(path: Path) -> dict[str, Any] | None:
    p = path.stem.split("-")
    if len(p) != 7 or p[2] not in EMOTIONS:
        return None
    actor = int(p[6])
    return {"emotion": EMOTIONS[p[2]], "intensity": "normal" if p[3] == "01" else "strong",
            "statement": p[4], "repetition": p[5], "actor": actor, "sex": "female" if actor % 2 == 0 else "male"}


def telephone(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Simulate a narrowband phone call: 300-3400 Hz band, 8 kHz, G.711 mu-law (8 bit)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    spec[(f < 300) | (f > 3400)] = 0
    n_new = int(round(len(x) * 8000 / sr))
    y = np.fft.irfft(spec[: n_new // 2 + 1], n_new) * (n_new / len(x))
    mu = 255.0
    y = np.clip(y, -1, 1)
    enc = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((enc + 1) / 2 * 255) / 255 * 2 - 1
    return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu, 8000


def praat_reference(x: np.ndarray, sr: int) -> dict[str, Any]:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    voiced = f0 > 0
    out: dict[str, Any] = {"t": times, "f0": f0}
    fv = f0[voiced]
    out["f0_median_hz"] = float(np.median(fv)) if fv.size else None
    out["f0_std_semitones"] = float(np.std(12 * np.log2(fv / np.median(fv)))) if fv.size > 2 else None
    pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
    out["jitter_local_pct"] = _nan_none(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100)
    out["shimmer_local_pct"] = _nan_none(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
    harm = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
    out["hnr_db"] = _nan_none(call(harm, "Get mean", 0, 0))
    try:
        pc = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        out["cpps_db"] = _nan_none(call(pc, "Get CPPS", "no", 0.01, 0.001, 60, 330, 0.05, "Parabolic",
                                        0.001, 0, "Straight", "Robust"))
    except Exception:
        out["cpps_db"] = None
    tg = call(snd, "To TextGrid (silences)", 100, 0, -25, 0.25, 0.1, "silent", "sounding")
    n_int = call(tg, "Get number of intervals", 1)
    sil = []
    for i in range(2, n_int):  # internal intervals only (skip leading/trailing silence)
        if call(tg, "Get label of interval", 1, i) == "silent":
            sil.append(call(tg, "Get end time of interval", 1, i) - call(tg, "Get start time of interval", 1, i))
    out["pause_count"] = len(sil)
    out["pause_total_s"] = float(sum(sil))
    return out


def _nan_none(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else float(v)


def utterance_record(ac: dict[str, Any], transcript: str | None) -> dict[str, Any]:
    timing = {"pauses": ac["pauses"], "speech_s": ac["speech_s"], "words_per_minute": None,
              "articulation_rate_sps": None, "response_latency_s": None}
    u: dict[str, Any] = {"acoustics": {k: v for k, v in ac.items() if k not in ("pauses", "speech_span_s", "tracks")},
                         "timing": timing}
    if transcript:
        t = analyze_text(transcript)
        u.update(fluency=t["fluency"], lexical=t["lexical"], sentiment=t["sentiment"])
        if ac["speech_span_s"] and ac["speech_span_s"] > 0.3:
            timing["words_per_minute"] = round(t["lexical"]["content_word_count"] / ac["speech_span_s"] * 60, 1)
        if ac["speech_s"] and ac["speech_s"] > 0.3:
            timing["articulation_rate_sps"] = round(t["lexical"]["syllable_count"] / ac["speech_s"], 2)
    return u


def process(path_str: str, do_phone: bool) -> dict[str, Any]:
    path = Path(path_str)
    meta = parse_ravdess(path) or {}
    x, sr = load_wav(path)
    t0 = time.perf_counter()
    ours = analyze_audio(x, sr, f0_min=F0_MIN, f0_max=F0_MAX, return_tracks=True)
    ms = (time.perf_counter() - t0) * 1000
    pr = praat_reference(x, sr)

    # frame-level pitch comparison on Praat's time grid
    t_ours = np.array(ours["tracks"]["t"])
    f_ours = np.array([np.nan if v is None else v for v in ours["tracks"]["f0_hz"]])
    idx = np.clip(np.round((pr["t"] - t_ours[0]) / 0.01).astype(int), 0, len(t_ours) - 1)
    fo, fp = f_ours[idx], pr["f0"]
    v_o, v_p = ~np.isnan(fo), fp > 0
    both = v_o & v_p
    rel = np.abs(fo[both] - fp[both]) / fp[both]
    rec: dict[str, Any] = {
        "file": path.name, **meta, "sr": sr, "duration_s": ours["duration_s"], "ours_ms": round(ms, 1),
        "frames_both_voiced": int(both.sum()),
        "gpe": float(np.mean(rel > 0.2)) if both.any() else None,
        "fine_err_pct": float(np.mean(rel[rel <= 0.2]) * 100) if (rel <= 0.2).any() else None,
        "vde": float(np.mean(v_o != v_p)),
        "voicing_recall": float(both.sum() / v_p.sum()) if v_p.any() else None,
        "ours_f0_median_hz": ours["pitch"]["f0_median_hz"], "praat_f0_median_hz": pr["f0_median_hz"],
        "ours_f0_std_st": ours["pitch"]["f0_std_semitones"], "praat_f0_std_st": pr["f0_std_semitones"],
        "ours_jitter_pct": ours["voice_quality"]["jitter_local_pct"], "praat_jitter_pct": pr["jitter_local_pct"],
        "ours_shimmer_pct": ours["voice_quality"]["shimmer_local_pct"], "praat_shimmer_pct": pr["shimmer_local_pct"],
        "ours_hnr_db": ours["voice_quality"]["hnr_db"], "praat_hnr_db": pr["hnr_db"], "praat_cpps_db": pr["cpps_db"],
        "ours_pause_count": ours["pauses"]["count"], "praat_pause_count": pr["pause_count"],
        "ours_pause_total_s": ours["pauses"]["total_s"], "praat_pause_total_s": pr["pause_total_s"],
        "quality": ours["audio_quality"]["quality"], "snr_db": ours["audio_quality"]["snr_db"],
    }
    transcript = STATEMENTS.get(meta.get("statement", ""))
    rec["utterance"] = utterance_record(ours, transcript)
    rec["transcript"] = transcript
    if do_phone:
        xp, srp = telephone(x, sr)
        ph = analyze_audio(xp, srp, f0_min=F0_MIN, f0_max=F0_MAX)
        prp = praat_reference(xp, srp)
        rec.update(phone_f0_median_hz=ph["pitch"]["f0_median_hz"], phone_jitter_pct=ph["voice_quality"]["jitter_local_pct"],
                   phone_shimmer_pct=ph["voice_quality"]["shimmer_local_pct"], phone_hnr_db=ph["voice_quality"]["hnr_db"],
                   phone_pause_count=ph["pauses"]["count"], phone_quality=ph["audio_quality"]["quality"],
                   phone_praat_jitter_pct=prp["jitter_local_pct"], phone_praat_shimmer_pct=prp["shimmer_local_pct"],
                   phone_praat_hnr_db=prp["hnr_db"], phone_praat_f0_median_hz=prp["f0_median_hz"])
    return rec


# --------------------------------------------------------------------------- statistics
def agree(a: list, b: list, tol: float | None = None, rel: bool = False) -> dict[str, Any]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    x, y = np.array(pairs, dtype=float).T
    d = x - y
    out = {"n": len(pairs), "mean_ours": round(float(x.mean()), 3), "mean_praat": round(float(y.mean()), 3),
           "bias": round(float(d.mean()), 3), "mae": round(float(np.abs(d).mean()), 3),
           "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 3) if x.std() > 0 and y.std() > 0 else None}
    if tol is not None:
        within = np.abs(d) / np.abs(y) <= tol if rel else np.abs(d) <= tol
        out["within_tol_pct"] = round(float(within.mean() * 100), 1)
        out["tol"] = f"{tol * 100:.0f}%" if rel else tol
    return out


def _rnd(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(v, nd)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    allv = np.array(pos + neg)
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def emotion_eval(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
    recs = [r for r in recs if r.get("emotion")]
    if not recs:
        return None
    by_actor: dict[int, list] = {}
    for r in recs:
        by_actor.setdefault(r["actor"], []).append(r)
    scored = []
    for actor, rs in by_actor.items():
        base_set = [r for r in rs if r["emotion"] == "neutral" or (r["emotion"] == "calm" and r["intensity"] == "normal")]
        base = {"status": "established", "sessions_used": 3, "utterances_used": len(base_set), "metrics": {}}
        for r in base_set:
            R.baseline_update(base, R.utterance_metrics(r["utterance"]))
        empty = {"status": "insufficient_data", "sessions_used": 0, "utterances_used": 0, "metrics": {}}
        for r in rs:
            if r in base_set or r["emotion"] == "disgust":
                continue
            toks = tokenize(r["transcript"] or "")
            e = R.estimate_emotion(r["utterance"], base, toks)
            e0 = R.estimate_emotion(r["utterance"], empty, toks)
            scored.append({**{k: r[k] for k in ("emotion", "intensity", "actor", "sex")},
                           "arousal": e["arousal"], "label": e["label"], "arousal_no_baseline": e0["arousal"]})

    def summarize(key: str) -> dict[str, Any]:
        hi = [s[key] for s in scored if s["emotion"] in HIGH_AROUSAL]
        lo = [s[key] for s in scored if s["emotion"] in LOW_AROUSAL]
        if not hi or not lo:
            return {"n_high": len(hi), "n_low": len(lo)}
        tp = sum(v >= 0.25 for v in hi) / len(hi)
        tn = sum(v < 0.25 for v in lo) / len(lo)
        return {"auc": _rnd(auc(hi, lo)), "balanced_accuracy": round((tp + tn) / 2, 3),
                "high_detected_pct": round(tp * 100, 1), "low_correct_pct": round(tn * 100, 1),
                "n_high": len(hi), "n_low": len(lo)}

    per_emotion = {}
    for emo in ["calm", "sad", "happy", "surprised", "fearful", "angry"]:
        for inten in ["normal", "strong"]:
            v = [s["arousal"] for s in scored if s["emotion"] == emo and s["intensity"] == inten]
            if v:
                per_emotion[f"{emo}/{inten}"] = {"n": len(v), "mean_arousal": round(float(np.mean(v)), 3)}
    strong_only = [s for s in scored if s["intensity"] == "strong" or s["emotion"] in LOW_AROUSAL]
    hi_s = [s["arousal"] for s in strong_only if s["emotion"] in HIGH_AROUSAL]
    lo_s = [s["arousal"] for s in strong_only if s["emotion"] in LOW_AROUSAL]
    labels: dict[str, int] = {}
    for s in scored:
        labels[s["label"]] = labels.get(s["label"], 0) + 1
    by_sex = {sx: _rnd(auc([s["arousal"] for s in scored if s["sex"] == sx and s["emotion"] in HIGH_AROUSAL],
                            [s["arousal"] for s in scored if s["sex"] == sx and s["emotion"] in LOW_AROUSAL]))
              for sx in ("female", "male")}
    return {"with_personal_baseline": summarize("arousal"), "without_baseline": summarize("arousal_no_baseline"),
            "auc_strong_intensity_only": _rnd(auc(hi_s, lo_s)), "auc_by_sex": by_sex,
            "per_emotion_mean_arousal": per_emotion, "label_counts": labels, "n_scored": len(scored)}


def indicator_rates(recs: list[dict[str, Any]], prefix: str = "ours") -> dict[str, Any]:
    strain = flat = 0
    n = 0
    for r in recs:
        j, s, h = r.get(f"{prefix}_jitter_pct"), r.get(f"{prefix}_shimmer_pct"), r.get(f"{prefix}_hnr_db")
        if None in (j, s, h):
            continue
        n += 1
        strain += ((j > 1.04) + (s > 3.81) + (h < 20)) >= 2
        st = r.get(f"{prefix}_f0_std_st")
        flat += st is not None and st < 1.0
    return {"n": n, "old_absolute_strain_rule_fires_pct": round(100 * strain / n, 1) if n else None,
            "flat_prosody_rule_fires_pct": round(100 * flat / n, 1) if n else None}


def strain_relative_rate(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """False-alarm rate of the shipped voice-strain rule (>= 2 of jitter/shimmer up, HNR down by >= 3 SD vs the
    person's own baseline), per clip, on healthy speakers. Baseline = each actor's neutral + calm/normal clips."""
    recs = [r for r in recs if r.get("emotion")]
    if not recs:
        return None
    fired: dict[str, list[int]] = {}
    keys = {"jitter_local_pct": "ours_jitter_pct", "shimmer_local_pct": "ours_shimmer_pct", "hnr_db": "ours_hnr_db"}
    for actor in {r["actor"] for r in recs}:
        rs = [r for r in recs if r["actor"] == actor]
        base_set = [r for r in rs if r["emotion"] == "neutral" or (r["emotion"] == "calm" and r["intensity"] == "normal")]
        base = {"metrics": {}}
        for r in base_set:
            R.baseline_update(base, {m: r[c] for m, c in keys.items() if r.get(c) is not None})
        for r in rs:
            if r in base_set or r.get("quality") == "poor":
                continue
            hits = 0
            for m, sign in (("jitter_local_pct", 1), ("shimmer_local_pct", 1), ("hnr_db", -1)):
                z, src = R.zscore(m, r.get(keys[m]), base)
                hits += src == "personal" and z is not None and sign * z >= 3.0
            fired.setdefault(r["emotion"], []).append(int(hits >= 2))
    allv = [v for vs in fired.values() for v in vs]
    return {"clips": len(allv), "fires_pct": round(100 * float(np.mean(allv)), 1),
            "by_emotion_pct": {k: round(100 * float(np.mean(v)), 1) for k, v in sorted(fired.items())}}


def summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
    col = lambda k: [r.get(k) for r in recs]  # noqa: E731
    gpe = [r["gpe"] for r in recs if r["gpe"] is not None]
    s: dict[str, Any] = {
        "files": len(recs),
        "paralingo_version": __version__,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "runtime_ms_per_file": {"median": round(float(np.median(col("ours_ms"))), 1),
                                "p95": round(float(np.percentile(col("ours_ms"), 95)), 1)},
        "pitch_frames": {
            "gross_pitch_error_pct": round(float(np.mean(gpe) * 100), 2),
            "files_with_gpe_over_10pct": round(float(np.mean(np.array(gpe) > 0.10) * 100), 1),
            "fine_error_pct": round(float(np.nanmean([r["fine_err_pct"] for r in recs if r["fine_err_pct"] is not None])), 2),
            "voicing_decision_error_pct": round(float(np.mean(col("vde")) * 100), 2),
            "voicing_recall_pct": round(float(np.nanmean([v for v in col("voicing_recall") if v is not None]) * 100), 1),
        },
        "vs_praat": {
            "f0_median_hz": agree(col("ours_f0_median_hz"), col("praat_f0_median_hz"), 0.05, rel=True),
            "f0_std_semitones": agree(col("ours_f0_std_st"), col("praat_f0_std_st"), 0.5),
            "jitter_local_pct": agree(col("ours_jitter_pct"), col("praat_jitter_pct"), 0.3),
            "shimmer_local_pct": agree(col("ours_shimmer_pct"), col("praat_shimmer_pct"), 1.5),
            "hnr_db": agree(col("ours_hnr_db"), col("praat_hnr_db"), 3.0),
            "pause_count": agree(col("ours_pause_count"), col("praat_pause_count"), 0),
            "pause_total_s": agree(col("ours_pause_total_s"), col("praat_pause_total_s"), 0.15),
        },
        "praat_cpps_db": {"mean": round(float(np.nanmean([v for v in col("praat_cpps_db") if v is not None])), 2)}
        if any(v is not None for v in col("praat_cpps_db")) else None,
        "audio_quality_labels": {q: col("quality").count(q) for q in sorted(set(col("quality")))},
        "indicator_false_alarms": {"ours": indicator_rates(recs, "ours"), "praat_values": indicator_rates(recs, "praat"),
                                   "baseline_relative_strain_rule": strain_relative_rate(recs)},
    }
    ph = [r for r in recs if "phone_f0_median_hz" in r]
    if ph:
        pc = lambda k: [r.get(k) for r in ph]  # noqa: E731
        s["phone_channel"] = {
            "files": len(ph),
            "f0_median_vs_clean": agree(pc("phone_f0_median_hz"), pc("ours_f0_median_hz"), 0.05, rel=True),
            "jitter_vs_clean": agree(pc("phone_jitter_pct"), pc("ours_jitter_pct"), 0.3),
            "shimmer_vs_clean": agree(pc("phone_shimmer_pct"), pc("ours_shimmer_pct"), 1.5),
            "hnr_vs_clean": agree(pc("phone_hnr_db"), pc("ours_hnr_db"), 3.0),
            "pause_count_vs_clean": agree(pc("phone_pause_count"), pc("ours_pause_count"), 0),
            "praat_jitter_phone_vs_clean": agree(pc("phone_praat_jitter_pct"), pc("praat_jitter_pct"), 0.3),
            "praat_hnr_phone_vs_clean": agree(pc("phone_praat_hnr_db"), pc("praat_hnr_db"), 3.0),
            "quality_labels": {q: pc("phone_quality").count(q) for q in sorted(set(pc("phone_quality")))},
            "indicator_false_alarms": indicator_rates(
                [{"ours_jitter_pct": r["phone_jitter_pct"], "ours_shimmer_pct": r["phone_shimmer_pct"],
                  "ours_hnr_db": r["phone_hnr_db"], "ours_f0_std_st": None} for r in ph], "ours"),
        }
    s["emotion"] = emotion_eval(recs)
    return s


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing WAV files (searched recursively)")
    ap.add_argument("--out", default=str(ROOT / "validation" / "results"))
    ap.add_argument("--limit", type=int, default=0, help="analyse only the first N files (0 = all)")
    ap.add_argument("--phone-every", type=int, default=3, help="also run the phone simulation on every Nth file (0 = never)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    files = sorted(Path(a.data).rglob("*.wav"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        print("no WAV files found")
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, str(f), bool(a.phone_every) and i % a.phone_every == 0) for i, f in enumerate(files)]
        recs = []
        for i, fu in enumerate(futs, 1):
            try:
                recs.append(fu.result())
            except Exception as exc:  # keep going; report failures
                print(f"failed: {files[i - 1].name}: {exc}")
            if i % 50 == 0 or i == len(files):
                print(f"{i}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    summary = summarize(recs)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = [k for k in recs[0] if k not in ("utterance",)]
    with open(out / "per_file.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in recs for k in r if k != "utterance"}, key=lambda k: (k not in keys, k)))
        w.writeheader()
        for r in recs:
            w.writerow({k: v for k, v in r.items() if k != "utterance"})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
