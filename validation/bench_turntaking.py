# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Turn-taking benchmark: Paralingo's end-of-turn detector vs fixed silence timeouts.

Builds simulated conversational turns from real recorded speech (RAVDESS): each turn is 1-3 spoken
sentences joined by hesitation pauses, followed by the end of the turn. Two questions:

* **Cut-ins**: how often does a system decide the turn is over during a mid-turn pause (interrupting)?
* **Response delay**: once the person has really finished, how long until the system responds?

Transcripts are given to Paralingo 300 ms after each sentence ends, as streaming ASR would. Half of the
mid-turn pauses come after a word that signals continuation ("and"); the other half come after a
complete sentence, where neither words nor pitch reveal that more is coming.

    python validation/bench_turntaking.py --data path/to/Audio_Speech_Actors_01-24 --out validation/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paralingo import __version__  # noqa: E402
from paralingo.audio import load_wav  # noqa: E402
from paralingo.realtime import TurnDetector  # noqa: E402

SR = 16000
STATEMENTS = {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"}
FIXED_TIMEOUTS_MS = (500, 700, 1000, 1500)
ASR_LAG_S = 0.30


def resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return x
    spec = np.fft.rfft(x)
    n = int(round(len(x) * target / sr))
    return np.fft.irfft(spec[: n // 2 + 1], n) * (n / len(x))


def trim(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(0.01 * sr)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
    on = np.flatnonzero(db > db.max() - 35)
    return x[on[0] * hop:(on[-1] + 1) * hop] if on.size else x


def build_turns(data: Path, rng: np.random.Generator, per_actor: int) -> list[dict]:
    turns = []
    for actor_dir in sorted(p for p in data.iterdir() if p.is_dir()):
        clips = sorted(actor_dir.glob("*.wav"))
        for _ in range(per_actor):
            n = int(rng.choice([1, 2, 2, 3]))
            parts, pauses, texts, cont = [], [], [], []
            for i in range(n):
                f = clips[int(rng.integers(len(clips)))]
                x, sr = load_wav(f)
                parts.append(trim(resample(x, sr), SR))
                texts.append(STATEMENTS[f.stem.split("-")[4]])
                if i < n - 1:
                    pauses.append(float(np.clip(rng.lognormal(np.log(0.5), 0.5), 0.2, 2.0)))
                    cont.append(bool(rng.random() < 0.5))
            turns.append({"parts": parts, "pauses": pauses, "texts": texts, "continuation": cont})
    return turns


def run_paralingo(turns: list[dict], rng: np.random.Generator, transcripts: bool, threshold: float | None = None,
                  fixed_timeout_s: float | None = None) -> dict:
    from paralingo.realtime import TurnConfig

    cfg = TurnConfig()
    if threshold is not None:
        cfg.threshold = threshold
    cfg.fixed_timeout_s = fixed_timeout_s
    det = TurnDetector(SR, cfg)
    by_type = {"after_continuation": [0, 0], "after_complete_sentence": [0, 0]}
    noise = lambda d: rng.normal(0, 0.001, int(d * SR))  # noqa: E731
    chunk = 320
    cutins = within = 0
    delays, missed = [], 0
    t_proc = 0.0
    audio_s = 0.0

    def push(audio: np.ndarray, pending_text: list) -> list:
        nonlocal t_proc, audio_s
        evs = []
        for i in range(0, len(audio), chunk):
            t0 = time.perf_counter()
            evs += det.push_audio(audio[i:i + chunk].astype(np.float32))
            t_proc += time.perf_counter() - t0
            if pending_text and det.now >= pending_text[0][0]:
                det.push_transcript(pending_text.pop(0)[1])
        audio_s += len(audio) / SR
        return evs

    push(noise(1.0), [])
    for turn in turns:
        spoken = ""
        pending: list = []
        for i, part in enumerate(turn["parts"]):
            if transcripts:
                words = turn["texts"][i].split()
                partial = (spoken + " " + " ".join(words[: len(words) // 2])).strip()
                pending.append((det.now + min(ASR_LAG_S + 0.4, len(part) / SR), partial))
            evs = push(part, pending)
            cutins += sum(e.type == "end_of_turn" for e in evs)  # ended mid-sentence: also a cut-in
            spoken = (spoken + " " + turn["texts"][i]).strip()
            last = i == len(turn["parts"]) - 1
            text = spoken + ("." if last or not turn["continuation"][i] else " and")
            if transcripts:
                pending.append((det.now + ASR_LAG_S, text))
            if not last:
                within += 1
                evs = push(noise(turn["pauses"][i]), pending)
                hit = any(e.type == "end_of_turn" for e in evs)
                cutins += hit
                kind = "after_continuation" if turn["continuation"][i] else "after_complete_sentence"
                by_type[kind][0] += hit
                by_type[kind][1] += 1
            else:
                speech_end = det.now
                evs = push(noise(3.0), pending)
                ends = [e for e in evs if e.type == "end_of_turn"]
                if ends:
                    delays.append((ends[0].t - speech_end) * 1000)
                else:
                    missed += 1
    d = np.asarray(delays)
    return {"cut_in_rate_pct": round(100 * cutins / max(within, 1), 1), "mid_turn_pauses": within,
            "cut_in_by_pause_type_pct": {k: round(100 * a / max(b, 1), 1) for k, (a, b) in by_type.items()},
            "turns": len(turns), "missed_turn_ends": missed,
            "response_delay_ms": {"mean": round(float(d.mean()), 0), "p50": round(float(np.percentile(d, 50)), 0),
                                  "p90": round(float(np.percentile(d, 90)), 0)},
            "real_time_factor": round(t_proc / audio_s, 4)}


def run_fixed(turns: list[dict], timeout_ms: int) -> dict:
    pauses = [p for t in turns for p in t["pauses"]]
    cut = sum(p * 1000 >= timeout_ms for p in pauses)
    return {"cut_in_rate_pct": round(100 * cut / max(len(pauses), 1), 1), "mid_turn_pauses": len(pauses),
            "turns": len(turns), "response_delay_ms": {"mean": timeout_ms, "p50": timeout_ms, "p90": timeout_ms}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "validation" / "results"))
    ap.add_argument("--per-actor", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sweep", type=float, nargs="*", default=[0.5, 0.7, 0.8],
                    help="extra end-of-turn thresholds to evaluate (with transcripts)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    turns = build_turns(Path(a.data), rng, a.per_actor)
    results = {"paralingo_version": __version__, "seed": a.seed, "asr_lag_s": ASR_LAG_S,
               "paralingo_with_transcripts": run_paralingo(turns, np.random.default_rng(1), True),
               "paralingo_audio_only": run_paralingo(turns, np.random.default_rng(1), False),
               "paralingo_threshold_sweep": {str(th): run_paralingo(turns, np.random.default_rng(1), True, th)
                                             for th in a.sweep},
               "fixed_timeouts_same_vad": {f"{ms}ms": run_paralingo(turns, np.random.default_rng(1), False,
                                                                     fixed_timeout_s=ms / 1000)
                                           for ms in FIXED_TIMEOUTS_MS},
               "fixed_timeouts_ideal_vad": {f"{ms}ms": run_fixed(turns, ms) for ms in FIXED_TIMEOUTS_MS}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "turntaking.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
