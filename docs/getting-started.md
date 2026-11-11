# Getting started

## Install

Python 3.10 or newer.

```bash
pip install paralingo
```

Optional extras:

| Install | Adds |
|---|---|
| `pip install "paralingo[api]"` | The HTTP API server with API keys |
| `pip install "paralingo[mcp]"` | The MCP server, for MCP-capable agents |
| `pip install "paralingo[validation]"` | JSON Schema validation of every write (recommended in production) |

## Your first speech profile

This example creates a short audio clip, analyses it as one turn of a phone call, and prints what an
agent would learn. Copy it into a file and run it.

```python
import numpy as np
from paralingo import SpeechProfileEngine
from paralingo.audio import save_wav

# 1. A stand-in for one recorded user turn: 4 seconds of a 180 Hz voice-like tone, saved as WAV.
#    In a real agent this is the audio of the caller's turn from your voice platform.
sr = 16000
t = np.arange(int(4.0 * sr)) / sr
voice = 0.3 * sum((0.6 / k) * np.sin(2 * np.pi * 180 * k * t) for k in range(1, 6))
save_wav("turn1.wav", np.concatenate([np.zeros(sr // 4), voice, np.zeros(sr // 4)]), sr)

# 2. One engine per agent. Profiles live in a folder: one file per speaker.
eng = SpeechProfileEngine("profiles", agent_id="my-voice-bot")

# 3. A call = a session. Each user turn = one analyze_utterance call.
sid = eng.start_session("caller-42", channel="phone", context="billing question")
result = eng.analyze_utterance(
    "caller-42",
    session_id=sid,
    audio_path="turn1.wav",
    transcript="Um, hi, I think I was charged twice this month.",
    response_latency_s=0.8,
)
print("emotion:", result["emotion"]["label"], "| pitch:", result["key_metrics"]["f0_mean_hz"], "Hz")
print("flags:", result["flags"])

# 4. End of call: roll the session into the person's baseline, trends and skill scores.
summary = eng.end_session("caller-42", sid)
print("turns analysed:", summary["utterance_count"])

# 5. What any agent should know before the next call with this person.
print(eng.get_summary("caller-42")["prompt_context"])
```

You now have `profiles/caller-42.speech.json`. Any agent with access to the `profiles` folder (or the
HTTP API, or the MCP server) can read and update it.

## What `analyze_utterance` returns

| Key | What it is |
|---|---|
| `session_id`, `utterance_id` | IDs of the session and of this turn in the file |
| `emotion` | `label` (e.g. calm, frustrated), `arousal` and `valence` from −1 to 1, `confidence`, and the `cues` that produced it |
| `skills` | 0–100 scores: fluency, pace, clarity, expressiveness, confidence, vocabulary, responsiveness, overall |
| `flags` | Notable things in this turn, e.g. `long_pause`, `high_disfluency`, `slow_rate`, `rising_terminal` |
| `baseline_deviations` | Metrics more than 2 standard deviations from this person's norm |
| `key_metrics` | Pitch, loudness, pace, pauses, fillers and voice quality for this turn |
| `audio_quality` | `good`, `fair` or `poor`, so you know how far to trust the acoustics |
| `guidance` | What the agent should do now: speaking rate, how long to wait before replying, adaptations |

## How the profile improves over time

Emotion and change detection compare the person with **their own baseline**, which builds up as you
close sessions:

| Baseline status | When |
|---|---|
| `insufficient_data` | Fewer than 5 analysed turns |
| `provisional` | 5 or more turns |
| `established` | 3 or more sessions and 15 or more turns |

Before the baseline exists, Paralingo still analyses everything, but emotion estimates rely on words and
broad population averages, so confidence is lower.

## No audio? Transcripts work too

If your platform gives you only transcripts, or transcripts with word timings, pass those. You get
fluency, vocabulary, sentiment, pace and pauses; pitch and voice quality need audio.

```python
from paralingo import SpeechProfileEngine

eng = SpeechProfileEngine("profiles", agent_id="text-only-bot")
words = [
    {"word": "I", "start": 0.00, "end": 0.10},
    {"word": "need", "start": 0.15, "end": 0.40},
    {"word": "help", "start": 1.60, "end": 1.90},   # a 1.2-second pause before "help"
]
r = eng.analyze_utterance("caller-7", transcript="I need help", word_timestamps=words)
print(r["flags"])   # includes "long_pause"
```

Omitting `session_id` uses the speaker's active session, or starts one.

## Try the full demo

The repository's `examples/demo.py` simulates three calls with one caller, including a frustrated one,
and prints the resulting profile:

```bash
python examples/demo.py
```

## Next steps

* [Integration guide](integration-guide.md): wire Paralingo into a real voice agent.
* [Profile file and metrics](profile-format.md): what every number means.
* [HTTP API](http-api.md): use it from JavaScript, Go, Java or anything else.
