# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Azure AI Speech adapter.

Connects an Azure ``SpeechRecognizer`` to a Paralingo :class:`~paralingo.realtime.TurnDetector`: the same
audio goes to both, and Azure's live transcripts, word timings and detected language feed Paralingo's
end-of-turn, barge-in and backchannel decisions.

    import azure.cognitiveservices.speech as speechsdk
    from paralingo.realtime import TurnDetector
    from paralingo.integrations.azure import AzureTurnBridge, configure_speech_config

    speech_config = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
    configure_speech_config(speech_config)
    stream = speechsdk.audio.PushAudioInputStream()        # 16 kHz, 16-bit, mono PCM by default
    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config=speechsdk.audio.AudioConfig(stream=stream))

    bridge = AzureTurnBridge(TurnDetector(16000), stream=stream, on_event=handle_event)
    bridge.attach(recognizer)
    recognizer.start_continuous_recognition()
    for chunk in mic_pcm16_chunks:
        bridge.push_audio(chunk)          # feeds Azure and Paralingo; events go to handle_event

The SDK (``azure-cognitiveservices-speech``) is only needed by your app; this module never imports it at
load time, so it can be tested and used with any SDK version that has the standard event objects.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

from ..realtime import TurnDetector, TurnEvent

TICKS_PER_SECOND = 10_000_000  # Azure offsets and durations are in 100-nanosecond ticks


def configure_speech_config(speech_config: Any, *, word_timestamps: bool = True,
                            segmentation_silence_ms: int | None = None) -> Any:
    """Ask Azure for what Paralingo uses: detailed results with word-level timings.

    ``segmentation_silence_ms`` optionally shortens Azure's own phrase segmentation; Paralingo decides
    turn ends itself, so Azure only needs to deliver text promptly.
    """
    import azure.cognitiveservices.speech as speechsdk  # imported here so the module loads without the SDK

    speech_config.output_format = speechsdk.OutputFormat.Detailed
    if word_timestamps:
        speech_config.request_word_level_timestamps()
    if segmentation_silence_ms is not None:
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
                                   str(int(segmentation_silence_ms)))
    return speech_config


def _result_json(result: Any) -> dict[str, Any]:
    raw = getattr(result, "json", None)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def result_words(result: Any) -> list[dict[str, Any]]:
    """Word timings from a detailed Azure result, in seconds from the start of the audio stream."""
    data = _result_json(result)
    nbest = data.get("NBest") or []
    words = (nbest[0].get("Words") if nbest else None) or []
    out = []
    for w in words:
        if "Offset" not in w or "Duration" not in w:
            continue
        start = w["Offset"] / TICKS_PER_SECOND
        out.append({"word": w.get("Word", ""), "start": round(start, 3),
                    "end": round(start + w["Duration"] / TICKS_PER_SECOND, 3),
                    "confidence": w.get("Confidence")})
    return out


def result_language(result: Any) -> str | None:
    """Detected language (e.g. "ja-JP") when Azure language identification is on."""
    data = _result_json(result)
    lang = (data.get("PrimaryLanguage") or {}).get("Language")
    if lang:
        return lang
    props = getattr(result, "properties", None)
    if props is not None:
        try:
            import azure.cognitiveservices.speech as speechsdk

            value = props.get(speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult)
            return value or None
        except Exception:
            return None
    return None


def result_confidence(result: Any) -> float | None:
    nbest = _result_json(result).get("NBest") or []
    c = nbest[0].get("Confidence") if nbest else None
    return float(c) if c is not None else None


class AzureTurnBridge:
    """Feeds one audio stream to Azure and to a TurnDetector, and Azure's text to the detector.

    Azure calls its event handlers on its own threads; the bridge serialises access to the detector.
    """

    def __init__(self, detector: TurnDetector, *, stream: Any = None,
                 on_event: Callable[[TurnEvent], None] | None = None) -> None:
        self.detector = detector
        self.stream = stream
        self.on_event = on_event
        self._lock = threading.Lock()
        self.words: list[dict[str, Any]] = []       # words of recognised phrases, stream time
        self.confidences: list[float] = []
        self.language: str | None = None

    # ------------------------------------------------------------ wiring
    def attach(self, recognizer: Any) -> "AzureTurnBridge":
        recognizer.recognizing.connect(self.on_recognizing)
        recognizer.recognized.connect(self.on_recognized)
        return self

    def push_audio(self, chunk: bytes) -> list[TurnEvent]:
        """Send 16-bit PCM bytes to Azure (if a push stream was given) and to Paralingo."""
        if self.stream is not None:
            self.stream.write(chunk)
        with self._lock:
            events = self.detector.push_audio(chunk, encoding="pcm16")
        for ev in events:
            if ev.type == "end_of_turn":
                ev.data["words"] = self.words_for_turn(ev)
                ev.data["asr_confidence"] = (round(sum(self.confidences) / len(self.confidences), 3)
                                             if self.confidences else None)
                self.confidences = []
            if self.on_event:
                self.on_event(ev)
        return events

    def set_agent_speaking(self, speaking: bool) -> None:
        with self._lock:
            self.detector.set_agent_speaking(speaking)

    # ------------------------------------------------------------ Azure events
    def on_recognizing(self, evt: Any) -> None:
        text = getattr(evt.result, "text", "") or ""
        lang = result_language(evt.result) or self.language
        if text:
            with self._lock:
                self.detector.push_transcript(text, is_final=False, language=lang)

    def on_recognized(self, evt: Any) -> None:
        result = evt.result
        text = getattr(result, "text", "") or ""
        if not text:
            return
        lang = result_language(result)
        if lang:
            self.language = lang
        conf = result_confidence(result)
        with self._lock:
            if conf is not None:
                self.confidences.append(conf)
            self.words.extend(result_words(result))
            self.words = self.words[-2000:]
            self.detector.push_transcript(text, is_final=True, language=lang or self.language)

    # ------------------------------------------------------------ helpers
    def words_for_turn(self, event: TurnEvent) -> list[dict[str, Any]]:
        """Words that fall inside an end_of_turn event, with times relative to the turn's audio."""
        d = event.data
        start = d["turn_start_t"] - self.detector.cfg.preroll_s
        end = d["speech_end_t"] + 0.3
        return [{**w, "start": round(w["start"] - start, 3), "end": round(w["end"] - start, 3)}
                for w in self.words if start <= w["start"] <= end]
