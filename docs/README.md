# Paralingo documentation

Paralingo gives every speaker a shared file, `<speaker_id>.speech.json`, that AI voice agents read and
write to understand **how** a person speaks: pitch, pauses, pace, voice quality, fluency, emotion and
accent, judged against that person's own baseline. It also tells the agent what to do about it, for
example how long to wait before replying.

## Start here

| Guide | Read it when you want to |
|---|---|
| [Getting started](getting-started.md) | Install Paralingo and analyse your first speaker in 2 minutes |
| [Integration guide](integration-guide.md) | Plug Paralingo into your voice agent: call flow, audio formats, phone audio, LLM tools, MCP |
| [HTTP API](http-api.md) | Use Paralingo from any language over HTTPS, with API keys |

## Reference

| Reference | Contents |
|---|---|
| [Agent tools](tools-reference.md) | The 11 tools, their parameters and who should call them |
| [Profile file and metrics](profile-format.md) | Every section of the `.speech.json` file and what each metric means |
| [Privacy, consent and deployment](privacy-and-deployment.md) | Consent flags, retention, erasure, concurrency, where to run it |
| [Troubleshooting](troubleshooting.md) | Common problems and how to fix them |
| [Architecture](architecture.md) | How the pieces fit together, with diagrams |
| [Validation results](../validation/RESULTS.md) | Accuracy against Praat on 1,440 real recordings |

## The 30-second version

```python
from paralingo import SpeechProfileEngine

eng = SpeechProfileEngine("profiles", agent_id="my-voice-bot")
sid = eng.start_session("caller-42", channel="phone")
eng.analyze_utterance("caller-42", session_id=sid, transcript="Um, hi, I was charged twice.")
eng.end_session("caller-42", sid)
print(eng.get_summary("caller-42")["prompt_context"])   # paste into your agent's system prompt
```

Questions or ideas: [open an issue](https://github.com/namashworks/paralingo/issues).
