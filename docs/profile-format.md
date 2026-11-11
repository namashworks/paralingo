# Profile file and metrics

Each speaker has one JSON file, `<speaker_id>.speech.json`, defined by the JSON Schema in
[`paralingo/schema/paralingo.schema.json`](../paralingo/schema/paralingo.schema.json). Any language can
read it. To **write** it safely from outside Python, use the [HTTP API](http-api.md) or follow the write
protocol in [Privacy, consent and deployment](privacy-and-deployment.md#concurrency-and-storage).

A complete example is in [`examples/profiles/demo-caller.speech.json`](../examples/profiles/demo-caller.speech.json).

## Top-level sections

| Section | Contents |
|---|---|
| `format`, `version`, `profile_id`, `revision`, `created_at`, `updated_at` | File identity. `revision` increases on every write. |
| `speaker` | `speaker_id`, `display_name`, `languages`, `self_reported` (facts the person chose to share), `notes` |
| `consent` | `recording`, `voice_analysis`, `health_indicators`, `store_transcripts`, `retention_days`, `granted_at`, `granted_via` |
| `settings` | `max_detailed_sessions` (default 20), `max_utterances_per_session` (default 300) |
| `baseline` | The person's norm: `status`, `sessions_used`, `utterances_used`, and per-metric `n`, `mean`, `std`, `min`, `max` |
| `accent` | `current_estimate`, `estimates` history, `phonetic_markers`, `intelligibility` |
| `sessions` | One entry per call: `session_id`, `started_at`, `ended_at`, `status`, `channel`, `context`, `utterances`, `summary` |
| `observations` | Evidence-backed notes from agents: `category`, `observation`, `evidence`, `reasoning`, `confidence`, `agent_id` |
| `assessments` | `emotional_state`, `communication_skills`, `speaking_style`, `speech_indicators` |
| `trends` | Per metric, one value per session: `[{session_id, date, value}]` |
| `agent_guidance` | `recommended_speaking_rate_wpm`, `end_of_turn_silence_ms`, `adaptations` |
| `audit_log` | Who changed what and when: `timestamp`, `agent_id`, `action`, `revision` |
| `extensions` | Free space for your own namespaced data |

## One utterance (one user turn)

Each turn in `sessions[].utterances[]` holds the groups below. A value is `null` when its input was
missing, for example no audio, or a transcript too short to judge.

### `acoustics`: from audio

| Field | Meaning |
|---|---|
| `duration_s`, `speech_s` | Clip length, and time actually spent speaking (seconds) |
| `voiced_ratio` | Share of speech frames with a detectable pitch |
| `pitch.f0_mean_hz`, `pitch.f0_median_hz` | Average and median pitch (Hz) |
| `pitch.f0_std_hz`, `pitch.f0_min_hz`, `pitch.f0_max_hz` | Spread, 5th and 95th percentile of pitch (Hz) |
| `pitch.f0_std_semitones`, `pitch.f0_range_semitones` | Pitch variability and range in semitones: how expressive the intonation is |
| `pitch.terminal_contour`, `pitch.terminal_slope_st_per_s` | Pitch movement at the end of the turn: `rising` (> +3 semitones/s, often a question or uncertainty), `falling` (< −3) or `level` |
| `energy.rms_dbfs_mean`, `energy.rms_dbfs_std`, `energy.dynamic_range_db` | Loudness (dB relative to full scale) and how much it varies |
| `voice_quality.jitter_local_pct` | Cycle-to-cycle variation in pitch period (%); higher can mean a rough voice |
| `voice_quality.shimmer_local_pct` | Cycle-to-cycle variation in loudness (%); higher can mean a breathy voice |
| `voice_quality.hnr_db` | Harmonics-to-noise ratio (dB); lower means a noisier, breathier voice |
| `spectral.centroid_hz`, `spectral.hf_energy_ratio_db` | Brightness, and high- vs low-frequency energy (vocal effort) |
| `audio_quality.snr_db`, `audio_quality.clipping_ratio`, `audio_quality.quality` | Recording quality: `good`, `fair` or `poor` |

### `timing`

| Field | Meaning |
|---|---|
| `pauses.count`, `pauses.total_s`, `pauses.mean_s`, `pauses.max_s` | Silent pauses of 0.25 s or more inside the turn |
| `pauses.long_count` | Pauses of 1 s or more |
| `pauses.pause_ratio` | Share of the turn spent pausing |
| `words_per_minute` | Speaking rate, including pauses |
| `articulation_rate_sps` | Syllables per second of actual speech, excluding pauses |
| `response_latency_s` | How long the person took to start answering (if you passed it) |

### `fluency`: from the transcript

| Field | Meaning |
|---|---|
| `filler_count`, `fillers`, `filler_rate_per_100w` | "um", "uh", "you know" and similar, counted and per 100 words |
| `word_repetitions`, `part_word_repetitions`, `phrase_repetitions` | "I I", "b-b-but", "I want I want" |
| `prolongations`, `revisions`, `incomplete_words` | "sooo", self-corrections ("I mean", "sorry"), cut-off words |
| `stuttering_like_per_100_syllables` | Part-word and whole-word repetitions and prolongations per 100 syllables |
| `total_disfluency_per_100w` | All disfluencies per 100 words |

### `lexical` and `sentiment`

| Field | Meaning |
|---|---|
| `word_count`, `content_word_count`, `syllable_count`, `unique_words` | Size of the turn |
| `type_token_ratio` | Vocabulary variety (used in scores and the baseline only for turns of 15+ words) |
| `mean_word_length`, `sentence_count`, `mean_sentence_length_words` | Complexity |
| `hedges`, `hedge_rate_per_100w` | "maybe", "I think", "probably": uncertainty |
| `vague_word_count`, `question_count` | "thing", "stuff"; question marks |
| `sentiment.score` | Word sentiment from −1 (negative) to 1 (positive), English only |

### `emotion`

| Field | Meaning |
|---|---|
| `arousal` | −1 (calm, low energy) to 1 (energised), judged against the person's own baseline |
| `valence` | −1 (negative) to 1 (positive), mainly from words |
| `label` | One of `neutral`, `calm`, `pleased`, `happy`, `excited`, `energetic`, `subdued`, `displeased`, `frustrated`, `angry`, `anxious`, `sad`, `tired`, or the label from an `external` emotion model you passed |
| `confidence` | 0–0.9; lower before the baseline exists or when audio is poor |
| `cues`, `method` | What drove the estimate, and how it was computed |

Arousal reached an AUC of 0.90 on 1,440 real recordings with a personal baseline
([validation](../validation/RESULTS.md)). Valence and the finer labels are less reliable; treat them as hints.

### `skills`, `flags` and `baseline_deviations`

* `skills`: 0–100 scores for `fluency`, `pace`, `clarity`, `expressiveness`, `confidence`, `vocabulary`,
  `responsiveness` and `overall`. A score is `null` when there isn't enough evidence for it.
* `flags`: quick markers for this turn:

  | Flag | Meaning |
  |---|---|
  | `poor_audio` | Recording too noisy or clipped to trust the acoustics |
  | `long_pause` | At least one pause of 1 s or more |
  | `high_disfluency` | 10 or more disfluencies per 100 words |
  | `part_word_repetition` | "b-b-but"-style repetition seen |
  | `slow_rate`, `fast_rate` | Below 100 or above 200 words per minute |
  | `flat_intonation` | Pitch variability under 1 semitone |
  | `rising_terminal` | Pitch rises at the end of the turn |
  | `deviates_from_baseline` | Two or more metrics far from this person's norm |
  | `emotion_<label>` | A negative emotion (angry, frustrated, anxious, sad) with reasonable confidence |

* `baseline_deviations`: metric name to z-score, for metrics at least 2 standard deviations from the
  person's own baseline.

## Assessments

| Field | Contents |
|---|---|
| `emotional_state` | `current` (latest emotion), `recent_labels`, `trend` (`improving`, `stable`, `worsening`) |
| `communication_skills` | Rolling `scores` over the last 5 sessions, `strengths`, `areas_to_improve` |
| `speaking_style` | `pace`, `pitch_expressiveness`, `loudness`, `pausing`, `verbosity`, `directness`, `mean_words_per_turn`, `typical_response_latency_s` |
| `speech_indicators` | Non-diagnostic screening signals (only with `health_indicators` consent), each with `status` (`watch`, `persistent` after 3 of the last 5 sessions, `resolved`), `evidence`, `confounders` and a `disclaimer` |

Indicators: `stuttering_like_disfluencies`, `voice_quality_strain` (judged only against the person's own
baseline), `slowed_speech`, `rapid_cluttered_speech`, `reduced_prosody`, `word_finding_difficulty`,
`reduced_loudness`, `significant_change_from_baseline`. They are screening signals, never diagnoses; don't
tell a person they have a condition.

## Agent guidance

| Field | Use |
|---|---|
| `recommended_speaking_rate_wpm` | Your TTS rate: about 95 % of the person's own pace, between 110 and 170 |
| `end_of_turn_silence_ms` | Your VAD timeout: the person's 90th-percentile pause plus 250 ms, between 600 and 2,500 ms |
| `adaptations` | `[{instruction, reason, priority}]`, e.g. "Wait longer before taking the turn", sorted by priority |
