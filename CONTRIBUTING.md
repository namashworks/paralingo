# Contributing to Paralingo

Thanks for helping. Bug reports, new speech features, better models, docs and examples are all welcome.

## Before you start

* For anything larger than a small fix, open an issue first so we can agree on the approach.
* Read the [Code of Conduct](CODE_OF_CONDUCT.md).
* Security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Contributor License Agreement

All contributions need a signed [CLA](CLA.md). On your first pull request a bot posts a comment;
reply with the sentence it asks for and you are done for all future pull requests. You keep the
copyright in your work; the CLA lets the maintainer keep the project's licensing consistent.

## Development setup

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate        # Windows; on macOS/Linux: source .venv/bin/activate
```

```bash
pip install -e ".[dev,api]"
```

```bash
python -m pytest -q
```

## Pull requests

* Keep each pull request focused on one change, with tests.
* New metrics must be added to the JSON Schema, documented in the README, and covered by a test with
  synthetic audio or text where the expected value is known.
* New inferences must store their evidence and confidence, and health-related ones must stay
  non-diagnostic and consent-gated.
* Add `# SPDX-License-Identifier: Apache-2.0` and the copyright line at the top of new source files.
* Add a line to `CHANGELOG.md` under "Unreleased".

## Adding a language

Language packs live in `paralingo/lang/`, one file per language, in plain Python with no dependencies.
Copy `ko.py` (a language written with spaces) or `zh.py` (written without spaces) and fill in:

* how to count words and syllables;
* fillers, hedges, vague words, revision markers, and a small sentiment lexicon with negation;
* real-time turn-taking cues: words that mean "not finished", sentence endings, backchannels;
* typical speaking and articulation rates, with a source.

Register it in `paralingo/lang/__init__.py` and add tests with real sentences, including words that
look like fillers but aren't (see `tests/test_languages.py`). Native speakers' reviews of the existing
Mandarin, Japanese and Korean word lists are just as welcome. See [docs/languages.md](docs/languages.md).

## Good first contributions

* Language packs: Hindi, Spanish, Arabic, Vietnamese, Cantonese, and native-speaker reviews of the
  Mandarin, Japanese and Korean packs.
* Adapters for ASR providers' word timestamps (Deepgram, AssemblyAI, Whisper, Google).
* Integrations with voice frameworks (Pipecat, LiveKit Agents, Vapi).
* A pluggable speech-emotion-recognition model through the `external` field.

## Releases

Maintainers publish releases as described in [RELEASING.md](RELEASING.md).
