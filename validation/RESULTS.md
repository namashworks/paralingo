# Validation results

Dataset: **RAVDESS** speech, 1440 clips from 24 professional actors (12 female, 12 male), 8 emotions, 48 kHz. Reference: **Praat 6.1** via parselmouth. Produced by `validation/validate.py`; paralingo 1.0.0, Windows AMD64, Python 3.11.9.

RAVDESS is acted speech of two short sentences, so it tests acoustic accuracy and arousal well, but says little about pauses, fluency, valence (the sentences are emotionally neutral) or spontaneous conversation.

## Agreement with Praat

| Measure | Before fixes | After fixes |
|---|---|---|
| Pitch: gross errors (frames off by > 20 %) | 4.8 % | 4.8 % |
| Pitch: fine error on matching frames | 0.83 % | 0.83 % |
| Pitch: voiced frames found (recall) | 97.7 % | 97.7 % |
| Median pitch: correlation | 0.994 | 0.994 |
| Median pitch: clips within 5 % | 91.5 % | 91.5 % |
| Pitch variability: correlation | 0.675 | 0.675 |
| Jitter: mean ours / Praat | 2.55× | 1.08× |
| Jitter: correlation | 0.685 | 0.808 |
| Shimmer: mean ours / Praat | 1.19× | 0.98× |
| Shimmer: correlation | 0.838 | 0.849 |
| HNR: mean difference | +0.03 dB | +0.03 dB |
| HNR: correlation | 0.848 | 0.848 |
| HNR: clips within 3 dB | 96.7 % | 96.7 % |
| Pause count: clips with the same count | 87.6 % | 87.6 % |

Held-out check: jitter and shimmer settings were tuned on 20 % of clips. On the other 1152 clips, jitter is 1.05× Praat (median, correlation 0.806) and shimmer 0.98× (correlation 0.845).

## Phone line

480 clips passed through a simulated phone channel (300–3400 Hz, 8 kHz, G.711 µ-law) and compared with the same clip at full quality.

| Measure | Before fixes | After fixes |
|---|---|---|
| Median pitch within 5 % of clean | 48.3 % | 67.9 % |
| Median pitch: mean shift | +19.6 Hz | +4.2 Hz |
| HNR: mean shift | -1.32 dB | -0.73 dB |
| Jitter: mean shift | +0.38 pp | +0.00 pp |
| Shimmer: mean shift | +1.96 pp | +1.57 pp |

Phone audio shifts voice-quality measures, which is why indicators are judged against the person's own baseline and poor-quality audio is excluded.

## Arousal (calm vs energised)

Per speaker, a baseline is built from their neutral and calm clips; the other clips are scored. High arousal = happy, angry, fearful, surprised; low = calm, sad. Disgust is excluded.

| | With personal baseline | Without baseline |
|---|---|---|
| AUC (0.5 = chance, 1.0 = perfect) | 0.896 | 0.734 |
| Balanced accuracy | 0.777 | 0.686 |
| High-arousal clips detected | 95.1 % | 73.7 % |
| Low-arousal clips correct | 60.4 % | 63.5 % |

Strong-intensity clips only: AUC 0.949. By sex: female 0.89, male 0.909. Most misses are 'sad' delivered at strong intensity, which actors perform loudly (crying), so it measures as high arousal.

## Health indicator false alarms (healthy speakers)

| Rule | Clips where it fires |
|---|---|
| Old rule: absolute cut-offs (jitter > 1.04 %, shimmer > 3.81 %, HNR < 20 dB), using Praat's own values | 100.0 % |
| Current rule: ≥ 2 of jitter, shimmer, HNR worse by ≥ 3 SD than the person's own baseline | 8.0 % of single clips |
| Flat prosody (pitch variability < 1 semitone) | 0.1 % |

The absolute cut-offs come from sustained vowels and do not apply to connected speech, so they were removed. Per emotion, the current rule fires on: angry 9.4 %, calm 2.1 %, disgust 10.4 %, fearful 5.2 %, happy 3.1 %, sad 6.8 %, surprised 16.1 %. In the product it is evaluated over whole sessions (3+ turns) and must recur in 3 of 5 sessions before it is called persistent, so these single-clip rates are a worst case. RAVDESS has no voice disorders, so how many real problems the rule catches is not measured yet.

## Speed

Median 314.2 ms per clip (p95 637.4 ms) for the full acoustic analysis of a ~3.7 s, 48 kHz clip.

## Not yet validated

* Accuracy on spontaneous conversation, background noise and non-English speech.
* Valence (positive vs negative), fluency and stuttering measures, speaking rate, and pause detection in longer turns.
* Whether the indicators detect real voice or speech disorders (needs clinical datasets).

Reproduce: download RAVDESS from Zenodo (record 1188976), then run `python validation/validate.py --data <folder>` and `python validation/make_report.py ...`.
