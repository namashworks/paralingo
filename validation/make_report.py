# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Write validation/RESULTS.md from a validation run (and an optional earlier run for comparison).

    python validation/make_report.py --after results/summary.json --per-file results/per_file.csv \\
        --before validation/results/ravdess_before_fixes.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _f(v):
    return None if v in (None, "", "None") else float(v)


def held_out(per_file: Path) -> dict:
    """Jitter/shimmer agreement on clips not used for tuning (tuning used every 10th clip, offsets 0 and 5)."""
    rows = list(csv.DictReader(open(per_file, encoding="utf-8")))
    rows.sort(key=lambda r: (int(r["actor"]), r["file"]))
    tuning = {i for i in range(len(rows)) if i % 10 in (0, 5)}
    out = {}
    for name, a, b in [("jitter", "ours_jitter_pct", "praat_jitter_pct"), ("shimmer", "ours_shimmer_pct", "praat_shimmer_pct")]:
        pairs = np.array([(_f(r[a]), _f(r[b])) for i, r in enumerate(rows)
                          if i not in tuning and _f(r[a]) is not None and _f(r[b]) is not None])
        out[name] = {"n": len(pairs), "median_ratio": round(float(np.median(pairs[:, 0] / pairs[:, 1])), 2),
                     "pearson_r": round(float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]), 3)}
    return out


def row(label, before, after, fmt="{}"):
    b = fmt.format(before) if before is not None else "–"
    a = fmt.format(after) if after is not None else "–"
    return f"| {label} | {b} | {a} |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", required=True)
    ap.add_argument("--per-file", required=True)
    ap.add_argument("--before", default=str(HERE / "results" / "ravdess_before_fixes.json"))
    ap.add_argument("--out", default=str(HERE / "RESULTS.md"))
    a = ap.parse_args()
    A = json.loads(Path(a.after).read_text("utf-8"))
    B = json.loads(Path(a.before).read_text("utf-8")) if Path(a.before).exists() else None
    ho = held_out(Path(a.per_file))
    g = lambda d, *ks: None if d is None else _dig(d, ks)  # noqa: E731

    vp = lambda s, m, k: g(s, "vs_praat", m, k)  # noqa: E731
    ph = lambda s, m, k: g(s, "phone_channel", m, k)  # noqa: E731
    emo = A["emotion"]
    fa = A["indicator_false_alarms"]
    lines = [
        "# Validation results",
        "",
        f"Dataset: **RAVDESS** speech, {A['files']} clips from 24 professional actors (12 female, 12 male), 8 emotions, "
        "48 kHz. Reference: **Praat 6.1** via parselmouth. Produced by `validation/validate.py`; "
        f"paralingo {A['paralingo_version']}, {A['machine']}.",
        "",
        "RAVDESS is acted speech of two short sentences, so it tests acoustic accuracy and arousal well, "
        "but says little about pauses, fluency, valence (the sentences are emotionally neutral) or spontaneous conversation.",
        "",
        "## Agreement with Praat",
        "",
        "| Measure | Before fixes | After fixes |",
        "|---|---|---|",
        row("Pitch: gross errors (frames off by > 20 %)", g(B, "pitch_frames", "gross_pitch_error_pct"),
            g(A, "pitch_frames", "gross_pitch_error_pct"), "{:.1f} %"),
        row("Pitch: fine error on matching frames", g(B, "pitch_frames", "fine_error_pct"), g(A, "pitch_frames", "fine_error_pct"), "{:.2f} %"),
        row("Pitch: voiced frames found (recall)", g(B, "pitch_frames", "voicing_recall_pct"), g(A, "pitch_frames", "voicing_recall_pct"), "{:.1f} %"),
        row("Median pitch: correlation", vp(B, "f0_median_hz", "pearson_r"), vp(A, "f0_median_hz", "pearson_r"), "{:.3f}"),
        row("Median pitch: clips within 5 %", vp(B, "f0_median_hz", "within_tol_pct"), vp(A, "f0_median_hz", "within_tol_pct"), "{:.1f} %"),
        row("Pitch variability: correlation", vp(B, "f0_std_semitones", "pearson_r"), vp(A, "f0_std_semitones", "pearson_r"), "{:.3f}"),
        row("Jitter: mean ours / Praat", _ratio(B, "jitter_local_pct"), _ratio(A, "jitter_local_pct"), "{:.2f}×"),
        row("Jitter: correlation", vp(B, "jitter_local_pct", "pearson_r"), vp(A, "jitter_local_pct", "pearson_r"), "{:.3f}"),
        row("Shimmer: mean ours / Praat", _ratio(B, "shimmer_local_pct"), _ratio(A, "shimmer_local_pct"), "{:.2f}×"),
        row("Shimmer: correlation", vp(B, "shimmer_local_pct", "pearson_r"), vp(A, "shimmer_local_pct", "pearson_r"), "{:.3f}"),
        row("HNR: mean difference", vp(B, "hnr_db", "bias"), vp(A, "hnr_db", "bias"), "{:+.2f} dB"),
        row("HNR: correlation", vp(B, "hnr_db", "pearson_r"), vp(A, "hnr_db", "pearson_r"), "{:.3f}"),
        row("HNR: clips within 3 dB", vp(B, "hnr_db", "within_tol_pct"), vp(A, "hnr_db", "within_tol_pct"), "{:.1f} %"),
        row("Pause count: clips with the same count", vp(B, "pause_count", "within_tol_pct"), vp(A, "pause_count", "within_tol_pct"), "{:.1f} %"),
        "",
        f"Held-out check: jitter and shimmer settings were tuned on 20 % of clips. On the other {ho['jitter']['n']} clips, "
        f"jitter is {ho['jitter']['median_ratio']}× Praat (median, correlation {ho['jitter']['pearson_r']}) and shimmer "
        f"{ho['shimmer']['median_ratio']}× (correlation {ho['shimmer']['pearson_r']}).",
        "",
        "## Phone line",
        "",
        f"{A['phone_channel']['files']} clips passed through a simulated phone channel (300–3400 Hz, 8 kHz, G.711 µ-law) "
        "and compared with the same clip at full quality.",
        "",
        "| Measure | Before fixes | After fixes |",
        "|---|---|---|",
        row("Median pitch within 5 % of clean", ph(B, "f0_median_vs_clean", "within_tol_pct"), ph(A, "f0_median_vs_clean", "within_tol_pct"), "{:.1f} %"),
        row("Median pitch: mean shift", ph(B, "f0_median_vs_clean", "bias"), ph(A, "f0_median_vs_clean", "bias"), "{:+.1f} Hz"),
        row("HNR: mean shift", ph(B, "hnr_vs_clean", "bias"), ph(A, "hnr_vs_clean", "bias"), "{:+.2f} dB"),
        row("Jitter: mean shift", ph(B, "jitter_vs_clean", "bias"), ph(A, "jitter_vs_clean", "bias"), "{:+.2f} pp"),
        row("Shimmer: mean shift", ph(B, "shimmer_vs_clean", "bias"), ph(A, "shimmer_vs_clean", "bias"), "{:+.2f} pp"),
        "",
        "Phone audio shifts voice-quality measures, which is why indicators are judged against the person's own "
        "baseline and poor-quality audio is excluded.",
        "",
        "## Arousal (calm vs energised)",
        "",
        "Per speaker, a baseline is built from their neutral and calm clips; the other clips are scored. "
        "High arousal = happy, angry, fearful, surprised; low = calm, sad. Disgust is excluded.",
        "",
        "| | With personal baseline | Without baseline |",
        "|---|---|---|",
        f"| AUC (0.5 = chance, 1.0 = perfect) | {emo['with_personal_baseline']['auc']} | {emo['without_baseline']['auc']} |",
        f"| Balanced accuracy | {emo['with_personal_baseline']['balanced_accuracy']} | {emo['without_baseline']['balanced_accuracy']} |",
        f"| High-arousal clips detected | {emo['with_personal_baseline']['high_detected_pct']} % | {emo['without_baseline']['high_detected_pct']} % |",
        f"| Low-arousal clips correct | {emo['with_personal_baseline']['low_correct_pct']} % | {emo['without_baseline']['low_correct_pct']} % |",
        "",
        f"Strong-intensity clips only: AUC {emo['auc_strong_intensity_only']}. "
        f"By sex: female {emo['auc_by_sex']['female']}, male {emo['auc_by_sex']['male']}. "
        "Most misses are 'sad' delivered at strong intensity, which actors perform loudly (crying), so it measures as high arousal.",
        "",
        "## Health indicator false alarms (healthy speakers)",
        "",
        "| Rule | Clips where it fires |",
        "|---|---|",
        f"| Old rule: absolute cut-offs (jitter > 1.04 %, shimmer > 3.81 %, HNR < 20 dB), using Praat's own values | "
        f"{fa['praat_values']['old_absolute_strain_rule_fires_pct']} % |",
        f"| Current rule: ≥ 2 of jitter, shimmer, HNR worse by ≥ 3 SD than the person's own baseline | "
        f"{fa['baseline_relative_strain_rule']['fires_pct']} % of single clips |",
        f"| Flat prosody (pitch variability < 1 semitone) | {fa['ours']['flat_prosody_rule_fires_pct']} % |",
        "",
        "The absolute cut-offs come from sustained vowels and do not apply to connected speech, so they were removed. "
        "Per emotion, the current rule fires on: "
        + ", ".join(f"{k} {v} %" for k, v in fa["baseline_relative_strain_rule"]["by_emotion_pct"].items())
        + ". In the product it is evaluated over whole sessions (3+ turns) and must recur in 3 of 5 sessions before "
          "it is called persistent, so these single-clip rates are a worst case. RAVDESS has no voice disorders, so how many "
          "real problems the rule catches is not measured yet.",
        "",
        "## Speed",
        "",
        f"Median {A['runtime_ms_per_file']['median']} ms per clip (p95 {A['runtime_ms_per_file']['p95']} ms) for the full "
        "acoustic analysis of a ~3.7 s, 48 kHz clip.",
        "",
        "## Not yet validated",
        "",
        "* Accuracy on spontaneous conversation, background noise and non-English speech.",
        "* Valence (positive vs negative), fluency and stuttering measures, speaking rate, and pause detection in longer turns.",
        "* Whether the indicators detect real voice or speech disorders (needs clinical datasets).",
        "",
        "Reproduce: download RAVDESS from Zenodo (record 1188976), then run "
        "`python validation/validate.py --data <folder>` and `python validation/make_report.py ...`.",
        "",
    ]
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


def _dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _ratio(s, m):
    if s is None:
        return None
    d = s["vs_praat"][m]
    return d["mean_ours"] / d["mean_praat"] if d.get("mean_praat") else None


if __name__ == "__main__":
    raise SystemExit(main())
