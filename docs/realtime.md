# Real-time turn-taking

`paralingo.realtime` answers, live and in about 10 ms steps, the questions every voice agent struggles with:

* **Has the person finished speaking?** Fixed silence timeouts either interrupt people who pause to
  think or make the agent feel slow. Paralingo combines how long the person has been silent (relative to
  *their own* usual pauses), how their pitch and loudness moved at the end, and their last words.
  "and…", "um…", Japanese けど, Korean -는데 and Mandarin 然后 mean "not finished".
* **Did they interrupt the agent, or just say "mm-hmm"?** A real interruption means stop talking; a
  backchannel ("yeah", "嗯", "はい", "네") means keep going.
* **How fast is the agent?** End-of-turn delay and response time, measured on every turn.

It runs on any audio source and with any ASR, TTS or framework. It is pure numpy, takes about 1–2 % of
one CPU core in real time, and also runs on-device.

**Measured:** on 480 simulated turns of real speech with 300 ms ASR lag, Paralingo cut in mid-turn
**29 points less often than a 500 ms timeout at the same speed** (55 % vs 84 %, 431 vs 452 ms), and on
only 4.6–10 % of pauses after "and…" versus 31–53 % for fixed timeouts. Method and limits:
[turn-taking benchmark](../validation/TURN_TAKING.md).

## Events

| Event | Meaning | What your agent should do |
|---|---|---|
| `speech_start` | The person started speaking | Optional: show a "listening" state |
| `end_of_turn` | The person has finished | Reply. `data` includes the transcript, the turn's audio, the probability and the reasons. |
| `barge_in` | The person interrupted while the agent was speaking | Stop your TTS now |
| `backchannel` | A short "mm-hmm" / "yeah" while the agent was speaking | Keep talking |

## Quick example

```python
import numpy as np
from paralingo.realtime import TurnDetector

sr = 16000
t = np.arange(int(1.5 * sr)) / sr
voice = 0.3 * np.sin(2 * np.pi * 170 * 2 ** (-6 * t / 1.5 / 12) * t)   # falling pitch, like a statement
stream = np.concatenate([np.zeros(sr // 2), voice, np.zeros(sr)])     # silence, speech, silence

det = TurnDetector(sample_rate=sr)
for i in range(0, len(stream), 320):                                  # 20 ms chunks, as from a mic
    chunk = stream[i:i + 320].astype(np.float32)
    if i == int(2.2 * sr) // 320 * 320:
        det.push_transcript("Where is my order?", is_final=True)      # from your ASR, whenever it arrives
    for ev in det.push_audio(chunk):
        print(ev.type, ev.t, {k: v for k, v in ev.data.items() if k in ("probability", "silence_ms", "reasons")})
```

## Feeding audio

`push_audio(chunk, encoding="pcm16")` accepts:

| Input | Example source |
|---|---|
| NumPy float array in [−1, 1] | Microphone libraries, decoded audio |
| NumPy `int16` array | Raw PCM arrays |
| `bytes` of 16-bit little-endian PCM (`encoding="pcm16"`, the default) | WebRTC, Azure push streams, most SDKs |
| `bytes` of G.711 µ-law (`encoding="mulaw"`) | Telephony, e.g. Twilio Media Streams (8 kHz) |

Any chunk size and any sample rate from 8 to 48 kHz works. Decisions are identical whatever the chunk size.

## Transcripts and languages

Push your ASR's partial and final results as they arrive:

```python
# sketch: inside your ASR callbacks
det.push_transcript(partial_text)                          # interim results
det.push_transcript(final_text, is_final=True)             # final result of a phrase
det.push_transcript(text, is_final=True, language="ja-JP") # if your ASR tells you the language
```

Transcripts are optional. Without them, decisions use timing and prosody only. With them, Paralingo
also recognises unfinished and finished sentences in **English, Mandarin, Japanese and Korean**,
detecting the language from the script when you don't pass one. Other languages fall back to timing and
prosody.

| Language | "Not finished" examples | "Finished" examples |
|---|---|---|
| English | and, but, so, because, um, uh, "you know" | sentence ending with `.`, `?`, `!` |
| Mandarin | 然后, 就是, 那个, 因为, 所以, 嗯 | 吗, 呢, 吧, 了, `。`, `？` |
| Japanese | けど, で, て, から, あの, えーと | です, ます, ね, よ, か, `。`, `？` |
| Korean | -는데, -고, -서, -지만, 그리고, 근데 | -요, -다, -까, -니다, `.`, `?` |

ASR text usually lags the audio by a few hundred milliseconds, so the latest words may not be in yet when
a decision is due. Paralingo accounts for that by waiting slightly longer before calling a barge-in when
no text has arrived.

## Interruptions

Tell the detector when your agent is speaking:

```python
# sketch: around your TTS playback
det.set_agent_speaking(True)    # when TTS audio starts playing
det.set_agent_speaking(False)   # when it finishes or you stop it
```

While the agent speaks, user speech becomes a `barge_in` after 0.3 s if it contains real words, or after
0.6 s if no text has arrived yet. A short burst that is (or turns out to be) a backchannel word becomes
`backchannel`. Use your platform's echo cancellation: Paralingo raises its speech threshold by 6 dB while
the agent talks, but it can't remove loud echo of your own TTS.

## Personalised to each speaker

People pause differently. Calibrate the detector from the speaker's Paralingo profile, and analyse each
finished turn in the background:

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from paralingo import SpeechProfileEngine
from paralingo.realtime import TurnDetector, analyze_turn

eng = SpeechProfileEngine("profiles", agent_id="voice-bot")
sid = eng.start_session("caller-42", channel="phone")
det = TurnDetector.from_profile(eng, "caller-42", sample_rate=16000)   # uses their usual pauses and language
pool = ThreadPoolExecutor(max_workers=2)

sr = 16000
t = np.arange(int(1.5 * sr)) / sr
stream = np.concatenate([np.zeros(8000), 0.3 * np.sin(2 * np.pi * 160 * t), np.zeros(sr)])
futures = []
for i in range(0, len(stream), 320):
    for ev in det.push_audio(stream[i:i + 320].astype(np.float32)):
        if ev.type == "end_of_turn":
            futures.append(pool.submit(analyze_turn, eng, "caller-42", ev, session_id=sid))

results = [f.result() for f in futures]
eng.record_realtime_metrics("caller-42", sid, det.latency.summary())   # store the call's latency stats
eng.end_session("caller-42", sid)
print(len(results), det.latency.summary()["end_of_turn_delay_ms"])
```

Once a speaker has a profile, the detector waits about as long as *they* usually pause, so slow or
thoughtful speakers aren't cut off.

## Latency

`det.latency.summary()` returns:

| Field | Meaning |
|---|---|
| `end_of_turn_delay_ms` | Speech stopped → turn detected (p50, p95, mean) |
| `response_ms` | Speech stopped → agent audio started (set by `set_agent_speaking(True)`) |
| `processing.real_time_factor` | Paralingo's CPU time ÷ audio duration (about 0.01–0.02) |
| `processing.per_10ms_step_us` | Time to process each 10 ms of audio, in microseconds |

`record_realtime_metrics` saves it with the session, so you can track latency across calls.

## Tuning

`TurnDetector(sr, TurnConfig(...))`:

| Setting | Default | Meaning |
|---|---|---|
| `threshold` | 0.6 | End-of-turn probability needed. Higher = fewer interruptions, slower replies. |
| `min_silence_s` | 0.20 | Never end a turn before this much silence |
| `default_pause_s` | 0.70 | Assumed "long pause" before a speaker has a profile |
| `max_silence_s` / `hard_cap_s` | 1.6 / 2.5 | Always end after this much silence |
| `hesitation_bonus_s` | 0.6 | Extra wait after "um", "and", けど… |
| `barge_in_min_s` / `barge_in_no_text_s` | 0.30 / 0.60 | Speech needed to call an interruption, with / without text |
| `backchannel_max_s` | 0.80 | Longest speech counted as a backchannel |
| `vad_margin_db`, `vad_floor_dbfs`, `agent_echo_margin_db` | 9, −55, 6 | Speech detection sensitivity |
| `resume_margin_db` | 4 | How much softer speech may be and still count as the person continuing mid-turn |
| `fixed_timeout_s` | `None` | Set it (e.g. `0.7`) to use a plain silence timeout instead of Paralingo's decisions: useful as a fallback or to compare |

## Use it from any language

The [HTTP API](http-api.md#real-time-websocket) has a WebSocket endpoint: stream audio frames in, get
these events out. For Azure Speech, see the [Azure guide](azure.md).

## How it decides

Each 10 ms, Paralingo updates loudness and pitch. During a pause it estimates the probability that the
turn is over:

* **Timing:** silence compared with the person's long-pause length (their usual 90th-percentile pause).
* **Pitch:** falling at the end (finished), rising on a question (finished), level (often more to come).
* **Loudness:** trailing off.
* **Words:** unfinished or finished, in four languages.

The turn ends when the probability passes the threshold, or after the maximum wait. Every
`end_of_turn` lists its `reasons`.

This is a transparent, rule-based model. It doesn't need training data, works for new speakers, and
improves as the speaker's profile grows. The [turn-taking benchmark](../validation/TURN_TAKING.md) shows
how it compares with fixed timeouts.
