# Azure AI Speech

Use Paralingo with Azure AI Speech (Speech-to-Text). Azure turns speech into text; Paralingo decides
**when the person has finished**, detects **interruptions and backchannels**, measures **latency**, and
analyses **how** each turn was said.

Azure's own end-of-speech detection is a fixed silence timeout. Paralingo replaces it with decisions that
adapt to each person and understand unfinished sentences in English, Mandarin, Japanese and Korean.

Install:

```bash
pip install "paralingo[azure]"
```

## Python (server side)

The same 16 kHz, 16-bit PCM chunks go to Azure and to Paralingo. `AzureTurnBridge` does both and
passes Azure's live text, word timings and detected language to the turn detector.

```python
# sketch: requires an Azure Speech resource (key and region) and a source of PCM audio chunks
import azure.cognitiveservices.speech as speechsdk
from concurrent.futures import ThreadPoolExecutor
from paralingo import SpeechProfileEngine
from paralingo.realtime import TurnDetector, analyze_turn
from paralingo.integrations.azure import AzureTurnBridge, configure_speech_config

speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
configure_speech_config(speech_config)              # detailed results + word timings

# Detect English, Mandarin, Japanese and Korean, and allow switching mid-call
speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")
languages = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
    languages=["en-US", "zh-CN", "ja-JP", "ko-KR"])

stream = speechsdk.audio.PushAudioInputStream(
    stream_format=speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1))
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config, audio_config=speechsdk.audio.AudioConfig(stream=stream),
    auto_detect_source_language_config=languages)

eng = SpeechProfileEngine("profiles", agent_id="yuzu")
session_id = eng.start_session(caller_id, channel="web")
pool = ThreadPoolExecutor(max_workers=2)

def on_event(ev):
    if ev.type == "end_of_turn":
        send_to_llm(ev.data["transcript"], language=ev.data["language"])   # your agent
        pool.submit(analyze_turn, eng, caller_id, ev, session_id=session_id)
    elif ev.type == "barge_in":
        stop_tts()                                                         # the person interrupted
    # "backchannel": keep talking

bridge = AzureTurnBridge(TurnDetector.from_profile(eng, caller_id, 16000), stream=stream, on_event=on_event)
bridge.attach(recognizer)
recognizer.start_continuous_recognition()

for chunk in microphone_pcm16_chunks():      # bytes, e.g. 20 ms = 640 bytes at 16 kHz
    bridge.push_audio(chunk)                  # feeds Azure and Paralingo

# around your TTS playback:
bridge.set_agent_speaking(True)   # ... play audio ...   bridge.set_agent_speaking(False)

# at the end of the call
recognizer.stop_continuous_recognition()
eng.record_realtime_metrics(caller_id, session_id, bridge.detector.latency.summary())
eng.end_session(caller_id, session_id)
```

Each `end_of_turn` from the bridge also carries:

* `words`: Azure's word timings, relative to the turn's audio
* `asr_confidence`: Azure's average confidence for the turn
* `language`: the detected language (`en`, `zh`, `ja`, `ko`)

## Browser or JavaScript apps

If Azure Speech runs in the browser (the JavaScript Speech SDK), stream the same microphone audio to
Paralingo's [WebSocket endpoint](http-api.md#real-time-websocket) and forward Azure's text to it. Keep the
Paralingo API key on your server: open the WebSocket from a backend, or give the browser a short-lived
key with only the `audio:analyze` scope.

```javascript
// sketch: browser, with the Azure Speech JS SDK and an AudioWorklet that emits 16 kHz Int16 frames
const ws = new WebSocket(`wss://your-server/v1/realtime?speaker_id=${callerId}&sample_rate=16000&session_id=${sessionId}`);
ws.binaryType = "arraybuffer";
ws.onopen = () => ws.send(JSON.stringify({ type: "auth", api_key: paralingoKey }));
ws.onmessage = (msg) => {
  const ev = JSON.parse(msg.data);
  if (ev.type === "end_of_turn") sendToLLM(ev.data.transcript, ev.data.language);
  if (ev.type === "barge_in") stopTTS();
  if (ev.type === "analysis") showInsights(ev.data.emotion, ev.data.guidance);
};

workletNode.port.onmessage = (e) => ws.send(e.data);                 // Int16Array buffer from the mic

recognizer.recognizing = (s, e) =>
  ws.send(JSON.stringify({ type: "transcript", text: e.result.text, final: false }));
recognizer.recognized = (s, e) =>
  ws.send(JSON.stringify({ type: "transcript", text: e.result.text, final: true }));

// around TTS playback:
ws.send(JSON.stringify({ type: "agent_speaking", value: true }));   // ... and false when it stops
```

## Tips for Azure

* **Let Paralingo decide turn ends.** Azure still segments phrases by silence. A shorter segmentation
  silence makes final text arrive sooner (`configure_speech_config(speech_config,
  segmentation_silence_ms=400)`), which helps Paralingo's word cues. Azure's default also works.
* **Word timings** need detailed output and `request_word_level_timestamps()`; `configure_speech_config`
  sets both.
* **Continuous language identification** lets the language change mid-call. Each `end_of_turn` reports
  the language of that turn.
* **Measure first.** Run a few calls, then compare `latency.summary()` with your current setup:
  `end_of_turn_delay_ms` is the time Paralingo saves or adds compared with Azure's timeout.
