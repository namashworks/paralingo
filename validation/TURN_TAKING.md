# Turn-taking benchmark

How well does Paralingo know when a person has **finished** speaking, compared with the fixed silence timeout most voice agents use? Produced by `validation/bench_turntaking.py`; Paralingo 1.0.1, random seed 2026 (a held-out set: no setting was tuned on it).

## Method

* **480 simulated turns** built from real recorded speech (RAVDESS, 24 actors). Each turn has 1–3 spoken sentences joined by hesitation pauses of 0.2–2 s (median 0.5 s), 474 mid-turn pauses in total, then the turn ends.
* **Half the pauses follow a word that signals continuation** ("…and"). The other half follow a complete sentence with falling pitch, which by construction look exactly like a finished turn: no system can tell them apart, so they set a floor on cut-ins.
* Transcripts reach Paralingo **300 ms late**, as with streaming ASR: a partial (unpunctuated) text during each sentence and the final text after it.
* **Fixed timeouts run through the same speech detector** as Paralingo, so both face the same detection errors. Figures with perfect ("ideal") speech detection are shown separately for reference.

Measured: **cut-ins** (the system decided the turn was over during a mid-turn pause or mid-sentence, i.e. it would have interrupted) and **response delay** (time from the real end of speech to the decision).

## Results

| System | Cut-ins | …after "and" | …after a complete sentence | Mean delay | 90th pct delay |
|---|---|---|---|---|---|
| **Paralingo**, threshold 0.5 | 60.5 % | 10.0 % | 61.8 % | 382 ms | 570 ms |
| **Paralingo**, threshold 0.6 (default) | 55.3 % | 7.9 % | 55.4 % | 431 ms | 640 ms |
| **Paralingo**, threshold 0.7 | 47.7 % | 5.4 % | 47.6 % | 495 ms | 710 ms |
| **Paralingo**, threshold 0.8 | 39.2 % | 4.6 % | 37.3 % | 583 ms | 810 ms |
| Paralingo, audio only (no transcripts) | 63.7 % | 36.9 % | 34.8 % | 595 ms | 810 ms |
| Fixed 500ms timeout | 84.4 % | 53.1 % | 52.8 % | 452 ms | 510 ms |
| Fixed 700ms timeout | 55.7 % | 31.1 % | 28.3 % | 648 ms | 720 ms |
| Fixed 1000ms timeout | 28.3 % | 13.3 % | 15.9 % | 923 ms | 1010 ms |
| Fixed 1500ms timeout | 9.9 % | 5.0 % | 3.4 % | 1407 ms | 1510 ms |

Fixed timeouts with perfect speech detection (reference only): 500ms: 48.1 % cut-ins, 700ms: 24.5 % cut-ins, 1000ms: 8.0 % cut-ins, 1500ms: 1.7 % cut-ins.

## What this shows

* **At the same speed, Paralingo interrupts less.** Compare Paralingo at threshold 0.6 (55 % cut-ins, 431 ms) with a 500 ms timeout (84 %, 452 ms), and threshold 0.8 (39 %, 583 ms) with a 700 ms timeout (56 %, 648 ms).
* **It almost never interrupts someone who stopped on "and…".** That is where knowing the words matters.
* **After a complete sentence with falling pitch it behaves like a fast timeout**, because those pauses are indistinguishable from real endings in this test.
* **Speech detection errors hurt every system** (compare the ideal-detection figures). A stronger optional speech detector is on the roadmap.

Speed: Paralingo used about 1.3 % of one CPU core in real time.

## Limitations

* The turns are **simulated** from acted read speech, not recorded conversations. The "and" exists only in the transcript, not in the audio, so the audio-only result understates prosody: real hesitations usually sound unfinished (level pitch, no final fall).
* The ratio of continuation to complete-sentence pauses (50/50) is a choice; real conversations differ.
* English only. The Mandarin, Japanese and Korean word cues are covered by unit tests, not yet by a benchmark.
* Next: evaluate on real multi-speaker conversation recordings with annotated turns.

Reproduce: download RAVDESS (Zenodo record 1188976), then run `python validation/bench_turntaking.py --data <folder> --per-actor 20 --seed 2026` and `python validation/make_turntaking_report.py validation/results/turntaking.json`.
