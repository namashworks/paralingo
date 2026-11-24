# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Write validation/TURN_TAKING.md from a bench_turntaking.py result file.

    python validation/make_turntaking_report.py validation/results/turntaking.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def row(name: str, d: dict) -> str:
    bt = d.get("cut_in_by_pause_type_pct", {})
    rd = d["response_delay_ms"]
    return (f"| {name} | {d['cut_in_rate_pct']:.1f} % | {bt.get('after_continuation', '–')} % | "
            f"{bt.get('after_complete_sentence', '–')} % | {rd['mean']:.0f} ms | {rd['p90']:.0f} ms |")


def main(path: str) -> int:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    p = r["paralingo_with_transcripts"]
    sweep = r["paralingo_threshold_sweep"]
    same = r["fixed_timeouts_same_vad"]
    ideal = r["fixed_timeouts_ideal_vad"]
    lines = [
        "# Turn-taking benchmark",
        "",
        "How well does Paralingo know when a person has **finished** speaking, compared with the fixed silence "
        "timeout most voice agents use? Produced by `validation/bench_turntaking.py`; Paralingo "
        f"{r['paralingo_version']}, random seed {r['seed']} (a held-out set: no setting was tuned on it).",
        "",
        "## Method",
        "",
        f"* **{p['turns']} simulated turns** built from real recorded speech (RAVDESS, 24 actors). Each turn has "
        f"1–3 spoken sentences joined by hesitation pauses of 0.2–2 s (median 0.5 s), {p['mid_turn_pauses']} mid-turn "
        "pauses in total, then the turn ends.",
        "* **Half the pauses follow a word that signals continuation** (\"…and\"). The other half follow a complete "
        "sentence with falling pitch, which by construction look exactly like a finished turn: no system can tell "
        "them apart, so they set a floor on cut-ins.",
        f"* Transcripts reach Paralingo **{r['asr_lag_s'] * 1000:.0f} ms late**, as with streaming ASR: a partial "
        "(unpunctuated) text during each sentence and the final text after it.",
        "* **Fixed timeouts run through the same speech detector** as Paralingo, so both face the same detection "
        "errors. Figures with perfect (\"ideal\") speech detection are shown separately for reference.",
        "",
        "Measured: **cut-ins** (the system decided the turn was over during a mid-turn pause or mid-sentence, i.e. "
        "it would have interrupted) and **response delay** (time from the real end of speech to the decision).",
        "",
        "## Results",
        "",
        "| System | Cut-ins | …after \"and\" | …after a complete sentence | Mean delay | 90th pct delay |",
        "|---|---|---|---|---|---|",
        row("**Paralingo**, threshold 0.5", sweep["0.5"]),
        row("**Paralingo**, threshold 0.6 (default)", p),
        row("**Paralingo**, threshold 0.7", sweep["0.7"]),
        row("**Paralingo**, threshold 0.8", sweep["0.8"]),
        row("Paralingo, audio only (no transcripts)", r["paralingo_audio_only"]),
    ] + [row(f"Fixed {k} timeout", v) for k, v in same.items()] + [
        "",
        "Fixed timeouts with perfect speech detection (reference only): "
        + ", ".join(f"{k}: {v['cut_in_rate_pct']:.1f} % cut-ins" for k, v in ideal.items()) + ".",
        "",
        "## What this shows",
        "",
        f"* **At the same speed, Paralingo interrupts less.** Compare Paralingo at threshold 0.6 "
        f"({p['cut_in_rate_pct']:.0f} % cut-ins, {p['response_delay_ms']['mean']:.0f} ms) with a 500 ms timeout "
        f"({same['500ms']['cut_in_rate_pct']:.0f} %, {same['500ms']['response_delay_ms']['mean']:.0f} ms), and "
        f"threshold 0.8 ({sweep['0.8']['cut_in_rate_pct']:.0f} %, {sweep['0.8']['response_delay_ms']['mean']:.0f} ms) "
        f"with a 700 ms timeout ({same['700ms']['cut_in_rate_pct']:.0f} %, "
        f"{same['700ms']['response_delay_ms']['mean']:.0f} ms).",
        "* **It almost never interrupts someone who stopped on \"and…\".** That is where knowing the words matters.",
        "* **After a complete sentence with falling pitch it behaves like a fast timeout**, because those pauses are "
        "indistinguishable from real endings in this test.",
        "* **Speech detection errors hurt every system** (compare the ideal-detection figures). A stronger optional "
        "speech detector is on the roadmap.",
        "",
        f"Speed: Paralingo used about {p['real_time_factor'] * 100:.1f} % of one CPU core in real time.",
        "",
        "## Limitations",
        "",
        "* The turns are **simulated** from acted read speech, not recorded conversations. The \"and\" exists only "
        "in the transcript, not in the audio, so the audio-only result understates prosody: real hesitations "
        "usually sound unfinished (level pitch, no final fall).",
        "* The ratio of continuation to complete-sentence pauses (50/50) is a choice; real conversations differ.",
        "* English only. The Mandarin, Japanese and Korean word cues are covered by unit tests, not yet by a "
        "benchmark.",
        "* Next: evaluate on real multi-speaker conversation recordings with annotated turns.",
        "",
        "Reproduce: download RAVDESS (Zenodo record 1188976), then run "
        "`python validation/bench_turntaking.py --data <folder> --per-actor 20 --seed 2026` and "
        "`python validation/make_turntaking_report.py validation/results/turntaking.json`.",
        "",
    ]
    out = HERE / "TURN_TAKING.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results" / "turntaking.json")))
