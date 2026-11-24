# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Multilingual transcript analysis: language packs for English, Mandarin, Japanese and Korean
  (`paralingo.lang`) with fillers, hedges, repetitions and stutters, sentiment with negation, questions, and
  word and syllable (mora) counting. Other languages get counts through a generic fallback.
- Code-switching: turns that mix languages are split and each part analysed with its own pack. Each
  utterance gets a `language` block (main language, shares, borrowed terms), the profile keeps
  `assessments.language_use`, and the agent guidance says which language to reply in and to reuse the
  speaker's own terms.
- Language-fair pace: `timing.pace_ratio` and `timing.articulation_ratio` compare the speaker with typical
  speakers of the language they used; style, skills, indicators and guidance use them.
- `agent_guidance.recommended_tts_rate_pct` (any language, e.g. SSML `<prosody rate="-10%">`) and
  `agent_guidance.language`.
- `language` hint for `analyze_utterance`, the agent tool and the HTTP API; `analyze_turn` passes on the
  turn's language and ASR confidence.
- `register_pack` and `LanguagePack` to add languages; the real-time turn-taking cues now come from the packs.
- Mandarin: pitch-variability norms scaled for lexical tones, and final pitch counts half as much in
  end-of-turn decisions.
- Documentation: [languages and code-switching](docs/languages.md), and how to add a language in CONTRIBUTING.

### Changed
- `CITATION.cff` lists the v1.1.0 DOI (10.5281/zenodo.24506911).
- `recommended_speaking_rate_wpm` is now only given for English speakers; use `recommended_tts_rate_pct` for
  other languages.
- Emotion estimation uses `articulation_ratio` instead of `articulation_rate_sps` (identical for English;
  existing personal baselines rebuild this metric over the next 5 turns).

### Fixed
- "what I meant" was counted as two revisions ("what i meant" and "i meant").

## [1.1.0] - 2026-11-24

### Added
- Real-time turn-taking (`paralingo.realtime`): streaming end-of-turn prediction from silence (relative to
  each speaker's own pauses), pitch, loudness and words, with cues for English, Mandarin, Japanese and
  Korean; barge-in vs backchannel detection; latency measurement (end-of-turn delay, response time,
  processing cost). Pure numpy, about 1-2 % of one CPU core.
- Azure AI Speech adapter (`paralingo.integrations.azure`): feeds one audio stream to Azure and Paralingo,
  and Azure's live text, word timings, confidence and detected language to the turn detector.
- WebSocket endpoint `/v1/realtime` in the HTTP API, for browser and non-Python clients.
- `SpeechProfileEngine.record_realtime_metrics` stores a call's turn-taking statistics with its session.
- Optional extras: `paralingo[azure]`; `websockets` added to `paralingo[api]`.
- `TurnConfig.fixed_timeout_s`: a plain silence-timeout mode, as a fallback or for comparison.
- Turn-taking benchmark (`validation/bench_turntaking.py`, results in `validation/TURN_TAKING.md`). On 480
  held-out simulated turns of real speech with 300 ms ASR lag, Paralingo cut in mid-turn on 55 % of pauses at
  431 ms mean response, versus 84 % for a 500 ms timeout at 452 ms, and on under 10 % of pauses after "and…".
- Documentation: [real-time turn-taking](docs/realtime.md) and [Azure AI Speech](docs/azure.md) guides.

### Changed
- PyPI description now leads with real-time turn-taking. `CITATION.cff` lists the v1.0.1 DOI.

## [1.0.1] - 2026-11-11

### Added
- Documentation in `docs/`: getting started, integration guide (call flow, phone audio, LLM tools, MCP),
  HTTP API, generated tool reference, profile file and metrics reference, privacy and deployment,
  troubleshooting and architecture. Every runnable example is executed by the test suite.

### Fixed
- Turns with little surrounding silence (common with VAD-trimmed audio) were rated "poor" quality, which
  discarded their pitch and voice-quality data. Noise is now measured only on non-speech frames; with
  under 0.1 s of silence, quality is reported as "unknown" and the acoustics are kept.

## [1.0.0] - 2026-11-10

First public release.

### Added
- Speech profile file format (`.speech.json`) with JSON Schema.
- Acoustic analysis: pitch, intonation, loudness, pauses, jitter, shimmer, HNR, audio quality.
- Transcript analysis: fillers, repetitions, stuttering-like disfluencies, lexicon, hedging, sentiment.
- Personal baseline, emotion estimation, communication-skill scores, speaking style, consent-gated speech indicators and agent guidance.
- Concurrent-safe file store with locking, atomic writes, revisions, retention and audit log.
- 11 agent tools in Anthropic, OpenAI and MCP formats; MCP server; CLI.
- HTTP API with hashed API keys, tenants, scopes and rate limits.
- Validation against Praat on 1,440 real recordings (RAVDESS), with an arousal benchmark, a simulated
  phone line and indicator false-alarm rates. Results in `validation/RESULTS.md`:
  - pitch correlation 0.994 with Praat; HNR within 0.03 dB on average;
  - jitter 1.08x and shimmer 0.98x Praat's values (cycle-to-cycle cross-correlation);
  - arousal AUC 0.896 with a personal baseline vs 0.734 without.
- Voice-strain indicator judged against the person's own baseline (3 SD), because absolute clinical
  cut-offs flag 100 % of healthy speakers in normal speech.
- PyPI release workflow with Trusted Publishing, and Zenodo DOI via GitHub releases.
