# Paralingo: a shared file voice agents use to understand how someone speaks

[![CI](https://github.com/namashworks/paralingo/actions/workflows/ci.yml/badge.svg)](https://github.com/namashworks/paralingo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/paralingo.svg)](https://pypi.org/project/paralingo/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24116090.svg)](https://doi.org/10.5281/zenodo.24116090)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Created and maintained by **[Namash Aggarwal](https://github.com/namashworks)**.

Speech-to-text tells a voice agent **what** a person said. The speech profile also records **how** they said it.
Each speaker gets one JSON file, `<speaker_id>.speech.json`. Any number of AI voice agents can **read, write,
track and reason over** that file, across calls and over time.

| Captured | How |
|---|---|
| **Pitch / intonation** | F0 mean, range and variability (semitones), and rising/falling contour at the end of a turn |
| **Loudness** | RMS level, dynamic range, and high- vs low-band energy (vocal effort) |
| **Pauses and pace** | Silent pauses (count, mean, max, long pauses), pause ratio, words per minute, articulation rate, response latency |
| **Voice quality** | Jitter, shimmer and HNR (the standard Praat / MDVP measures) |
| **Fluency** | Fillers (um, uh, you know), whole-word, part-word and phrase repetitions, prolongations, self-corrections, stuttering-like disfluencies per 100 syllables |
| **Language** | Vocabulary richness, sentence length, hedging, vague words, questions, sentiment |
| **Emotion** | Arousal and valence measured against the person's **own baseline**, mapped to a label (calm, happy, frustrated, angry, anxious, sad…), with the cues behind it |
| **Communication skills** | 0–100 scores for clarity, fluency, pace, expressiveness, confidence, vocabulary and responsiveness, plus strengths and areas to improve |
| **Speaking style** | Pace, expressiveness, loudness, pausing, verbosity and directness |
| **Accent** | Accent estimates (from your accent-ID model or the LLM), phonetic markers, intelligibility |
| **Speech indicators** | *Non-diagnostic* screening signals: stuttering-like disfluency, voice strain, slowed speech, rapid/cluttered speech, flat prosody, word-finding difficulty, reduced loudness, change from baseline. Opt-in only, and a signal must recur across sessions before it is marked "persistent". |
| **Agent guidance** | What to do with all this: suggested speaking rate, **how long to wait before replying** (VAD endpointing), tone and turn-taking adaptations |
| **Observations** | Evidence-backed notes agents leave for each other (a reasoning journal) |
| **Trends and audit** | Per-session metric history, plus a log of who changed what and when |

## Layout

```
paralingo/
  schema/paralingo.schema.json   JSON Schema: the file contract (any language can read/write it)
  store.py        safe file access: cross-process lock, atomic write, revisions, validation, retention, audit
  audio.py        acoustic features from WAV/PCM (numpy only)
  language.py     transcript features: disfluency, lexical, hedging, sentiment
  reasoning.py    baseline, emotion, skills, style, indicators, guidance
  engine.py       high-level API (sessions, utterances, observations, accent, summary, trends)
  agent_tools.py  11 tool definitions (Anthropic / OpenAI / MCP format) + JSON dispatcher
  mcp_server.py   MCP server, so any MCP-capable agent can use the tools
  http_api.py     HTTP API protected by API keys (FastAPI), with tenants, scopes and rate limits
  api_keys.py     create / list / revoke API keys (only hashes are stored)
  __main__.py     CLI
examples/demo.py               simulates 3 calls and writes examples/profiles/demo-caller.speech.json
examples/claude_voice_agent.py Claude agent that reads and writes the profile through tools
tests/                         20 tests (audio accuracy, fluency, concurrency, consent, tools, HTTP API, regressions)
validation/                    accuracy harness against Praat on real recordings, and RESULTS.md
```

## Quick start

```bash
pip install "paralingo[api]"
```

To work on the code instead, clone the repo and install it in editable mode:

```bash
pip install -e ".[dev,api]"
```

```bash
python examples/demo.py
```

```bash
python -m pytest tests -q
```

```python
from paralingo import SpeechProfileEngine

eng = SpeechProfileEngine("profiles/", agent_id="support-bot")
eng.set_consent("caller-42", recording=True, voice_analysis=True, health_indicators=False)

sid = eng.start_session("caller-42", channel="phone", context="billing")
r = eng.analyze_utterance("caller-42", session_id=sid,
                          audio_path="turn_001.wav",              # or samples=..., sample_rate=...
                          transcript="um, I was charged twice",   # verbatim ASR, fillers included
                          word_timestamps=asr_words,              # optional, used when there is no audio
                          response_latency_s=0.8,
                          external={"asr_confidence": 0.93})      # optional outputs from other models
r["emotion"], r["flags"], r["guidance"]["end_of_turn_silence_ms"]  # adapt live, mid-call

eng.end_session("caller-42", sid)    # updates baseline, trends, skills, indicators, guidance
print(eng.get_summary("caller-42")["prompt_context"])   # paste into the agent's system prompt
```

## How agents use it

```
 mic ─► ASR ─► transcript ─┐
   └──► audio (WAV) ───────┼─► paralingo_analyze_utterance ─► <speaker>.speech.json
                           │                                          ▲      │
                           ▼                                          │      ▼
                       LLM agent ── get_summary / get_guidance / read_section / get_trend
                                 └─ add_observation / update_accent ──┘
```

* **Pipeline code**, which owns the audio, calls `start_session`, `analyze_utterance` after each user turn,
  and `end_session` when the call ends.
* **The LLM** loads `get_summary` at the start of the call; its `prompt_context` is a ready-made block for
  the system prompt. During the call it can read sections, trends and past observations, and **write**
  observations and accent estimates.
* Give the LLM only the read/observe tools with `SpeechProfileTools(..., include_pipeline_tools=False)`.

Tool definitions in each format:

```bash
python -m paralingo tools --format anthropic
```

```bash
python -m paralingo tools --format openai
```

To run the MCP server (`pip install mcp` first):

```bash
python -m paralingo.mcp_server --root ./profiles --audio-root ./recordings
```

## HTTP API with API keys

Run the API so voice agents in any language can use the profile over HTTPS.

Create a key. It is shown once; only its SHA-256 hash is stored in `keys.json`:

```bash
paralingo-keys --keys keys.json create --name support-bot --tenant acme --scopes profiles:read profiles:write audio:analyze
```

Start the server (put it behind TLS in production):

```bash
paralingo-api --root ./profiles --keys keys.json --port 8000
```

Send a user turn with its audio:

```bash
curl -H "Authorization: Bearer $PARALINGO_API_KEY" -F transcript="um, I was charged twice" -F audio=@turn.wav http://127.0.0.1:8000/v1/speakers/caller-42/utterances
```

| Method and path | Scope | Purpose |
|---|---|---|
| `GET /v1/health` | none | Liveness and version |
| `GET /v1/tools?format=anthropic\|openai\|mcp` | any | Tool definitions this key may call |
| `POST /v1/tools/{name}` | depends on tool | Run any of the 11 tools with a JSON body |
| `GET /v1/speakers` | `profiles:read` | List speakers in the key's tenant |
| `GET /v1/speakers/{id}/summary` | `profiles:read` | Snapshot plus `prompt_context` |
| `GET /v1/speakers/{id}/guidance` | `profiles:read` | Live guidance and current emotion |
| `POST /v1/speakers/{id}/sessions` | `profiles:write` | Start a session |
| `POST /v1/speakers/{id}/sessions/{sid}/end` | `profiles:write` | Close and roll up a session |
| `POST /v1/speakers/{id}/utterances` | `audio:analyze` | Multipart: `audio` (WAV), `transcript`, `word_timestamps`, `external`, `session_id`, `response_latency_s` |
| `DELETE /v1/speakers/{id}` | `admin` | Erase a speaker's data |

* **Tenants:** each key belongs to one tenant, and each tenant's profiles live in their own folder.
  A key can never read another tenant's data, even with `admin`.
* **Scopes:** `profiles:read`, `profiles:write`, `audio:analyze`, `admin`.
* **Rate limits:** per key, 120 requests per minute by default (`--rate` when creating the key); `429` with `Retry-After` when exceeded.
* **Audio** is analysed from a temporary file and deleted straight away; it is never stored. Server file paths are refused.
* **Audit:** every write is logged with the key's name (`api:<name>`).
* Interactive docs at `/docs`.

## Validation on real speech

Tested on **1,440 real recordings** (RAVDESS: 24 professional actors, 8 emotions) against **Praat**, the
reference tool in speech science. Full results, method and limits: [validation/RESULTS.md](validation/RESULTS.md).

| What | Result |
|---|---|
| Pitch vs Praat | correlation 0.994, 91.5 % of clips within 5 %, 4.79 % gross frame errors |
| HNR (voice clarity) vs Praat | mean difference +0.03 dB, 96.7 % of clips within 3 dB |
| Jitter / shimmer vs Praat | 1.08× / 0.98× Praat's values (correlation 0.808 / 0.849) |
| Arousal (calm vs energised) | AUC **0.896** with a personal baseline vs 0.734 without |
| Voice-strain false alarms on healthy speakers | 8.0 % of single clips (old absolute cut-offs: 100 %) |
| Simulated phone line (8 kHz µ-law) | pitch within 5 % of full quality on 67.9 % of clips |

Not yet validated: spontaneous conversation, noisy rooms, non-English speech, valence, fluency measures, and
whether the indicators catch real speech or voice disorders. Reproduce with `validation/validate.py`.

## Design decisions

* **Personal baseline first.** Pitch and loudness vary hugely between people and microphones, so emotion and
  change detection use z-scores against the speaker's own norm (Welford running stats). The baseline goes from
  `insufficient_data` to `provisional` (5 utterances) to `established` (3 sessions and 15 utterances).
  Population priors are used only for pace, pausing and intonation variability.
* **Explainable and humble.** Every emotion carries its `cues`, `confidence` (capped at 0.9, lowered for poor audio)
  and `method`. If you supply a dedicated speech-emotion model's result in `external.emotion`, it is blended in.
* **Safe for many agents at once.** Lock file plus atomic `os.replace`, `revision` counter with
  `expected_revision` for optimistic concurrency, JSON Schema validation on every write, and an audit log.
  Tested with 4 concurrent writers.
* **Privacy by design.** Consent flags control what is analysed: `voice_analysis`, `health_indicators` (opt-in)
  and `store_transcripts` (metrics only if false). `retention_days` purges old data automatically, older sessions
  are compacted to summaries, and `store.delete()` covers right-to-erasure. Audio paths passed through tools are
  confined to `audio_root`.
* **Extensible.** The `extensions` object and the `external` field per utterance take vendor model outputs
  without schema changes.

## License and attribution

Copyright 2026 Namash Aggarwal. Licensed under the [Apache License 2.0](LICENSE).

You may use, modify and distribute this software, including commercially. If you redistribute it or a
modified version, you must keep the [`NOTICE`](NOTICE) file, which credits the original author. The
project name is not licensed for promoting derived products.

**If this project helps you, please credit it:** link to `https://github.com/namashworks/paralingo`
and name **Namash Aggarwal** as the original author. For papers, use GitHub's "Cite this repository"
button (from [`CITATION.cff`](CITATION.cff)), or copy this:

> Aggarwal, N. (2026). *Paralingo: a shared speech-analysis file for AI voice agents* (Version 1.0.0)
> [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24116090

```bibtex
@software{aggarwal_paralingo_2026,
  author    = {Aggarwal, Namash},
  title     = {Paralingo: a shared speech-analysis file for AI voice agents},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24116090},
  url       = {https://github.com/namashworks/paralingo}
}
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). A short [CLA](CLA.md) is signed once,
by comment, on your first pull request. Please read the [Code of Conduct](CODE_OF_CONDUCT.md), and report
security issues as described in [SECURITY.md](SECURITY.md).

## Limits

* The speech indicators are **screening signals, not diagnoses**. Each stores its confounders and a disclaimer.
  Voice strain is judged only against the person's own baseline, because the usual clinical cut-offs are for
  sustained vowels and flag every healthy speaker in normal speech (see [validation](validation/RESULTS.md)).
  How well the indicators detect real disorders has not been measured.
* Accent is **not** inferred from acoustics here. Accent is better handled by a dedicated accent-ID model or the
  LLM, which write to it through `update_accent`.
* Disfluency counts depend on verbatim ASR. Turn on filler and disfluency output in your ASR
  (e.g. Deepgram `filler_words`, AssemblyAI `disfluencies`).
* The sentiment lexicon is small and English-only. For production or other languages, pass a proper sentiment or
  emotion model's output in `external`.
* Inferring emotions or health traits from voice may be regulated where you operate (e.g. the EU AI Act's limits
  on emotion recognition in workplaces and schools, biometric-data rules under GDPR/BIPA). Get explicit consent
  and check local law.
