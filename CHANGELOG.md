# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
