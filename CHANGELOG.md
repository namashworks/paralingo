# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Validation harness (`validation/`) comparing against Praat on real recordings, with an arousal benchmark,
  a simulated phone line and indicator false-alarm rates. Results in `validation/RESULTS.md`.

### Fixed
- Jitter/shimmer pulse detection could loop forever or crash on real 48 kHz speech.
- Jitter was 2.5x Praat's value; it now uses cycle-to-cycle cross-correlation (1.08x Praat).
- Pitch on 8 kHz phone audio read too high; within-5 % agreement rose from 48 % to 68 %.

### Changed
- Voice-strain indicator now compares with the person's own baseline (3 SD) instead of absolute cut-offs,
  which flagged 100 % of healthy speakers in normal speech.

## [1.0.0] - 2026-11-10

### Added
- Speech profile file format (`.speech.json`) with JSON Schema.
- Acoustic analysis: pitch, intonation, loudness, pauses, jitter, shimmer, HNR, audio quality.
- Transcript analysis: fillers, repetitions, stuttering-like disfluencies, lexicon, hedging, sentiment.
- Personal baseline, emotion estimation, communication-skill scores, speaking style, consent-gated speech indicators and agent guidance.
- Concurrent-safe file store with locking, atomic writes, revisions, retention and audit log.
- 11 agent tools in Anthropic, OpenAI and MCP formats; MCP server; CLI.
- HTTP API with hashed API keys, tenants, scopes and rate limits.
