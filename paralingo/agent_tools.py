# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Tool (function-calling) definitions + dispatcher so any LLM voice agent can read,
write and reason over speech profiles.

    from paralingo.agent_tools import SpeechProfileTools
    tools = SpeechProfileTools("profiles/", agent_id="support-bot", audio_root="recordings/")
    tools.definitions("anthropic")   # or "openai"
    result_json = tools.execute(name, tool_input)

Typical split of responsibilities in a voice pipeline:
* The *pipeline code* (which owns the audio) calls ``paralingo_analyze_utterance``
  after each user turn and ``paralingo_end_session`` when the call ends.
* The *LLM* reads ``paralingo_get_summary`` / ``get_guidance`` at call start,
  and writes ``add_observation`` / ``update_accent`` as it reasons during the call.
Pass ``include_pipeline_tools=False`` to hide the audio tools from the LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import OBS_CATEGORIES, SpeechProfileEngine
from .reasoning import TREND_METRICS
from .store import ProfileError, SpeechProfileStore

_SPEAKER = {"type": "string", "description": "Stable id of the person (e.g. hashed phone number or CRM id)."}
_SESSION = {"type": "string", "description": "Session (call) id. Omit to use the speaker's active session."}
SECTIONS = ["speaker", "consent", "baseline", "accent", "sessions", "observations", "assessments", "trends",
            "agent_guidance", "audit_log"]

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "paralingo_get_summary",
        "description": "Load a compact snapshot of how this person speaks: emotional state, communication-skill scores, "
                       "speaking style, accent, active (non-diagnostic) speech indicators, recent agent observations and "
                       "concrete guidance for how to talk to them. Call at the start of every conversation.",
        "input_schema": {"type": "object", "properties": {"speaker_id": _SPEAKER},
                         "required": ["speaker_id"], "additionalProperties": False},
    },
    {
        "name": "paralingo_get_guidance",
        "description": "Get the latest adaptations for talking to this person (speaking rate, how long to wait before "
                       "replying, tone). Cheap; call whenever you want to re-check mid-conversation.",
        "input_schema": {"type": "object", "properties": {"speaker_id": _SPEAKER},
                         "required": ["speaker_id"], "additionalProperties": False},
    },
    {
        "name": "paralingo_read_section",
        "description": "Read one raw section of the speech profile file for detailed reasoning (e.g. 'sessions' for "
                       "per-utterance pitch/pauses/fluency, 'baseline' for the person's norms, 'trends').",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "section": {"type": "string", "enum": SECTIONS},
            "last_n": {"type": "integer", "minimum": 1, "maximum": 50,
                       "description": "For list sections: return only the last N items (default 3 for sessions, 20 otherwise)."}},
            "required": ["speaker_id", "section"], "additionalProperties": False},
    },
    {
        "name": "paralingo_get_trend",
        "description": "Longitudinal trend of one speech metric across sessions, with direction and the personal baseline.",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "metric": {"type": "string", "enum": TREND_METRICS},
            "last_n": {"type": "integer", "minimum": 2, "maximum": 100}},
            "required": ["speaker_id", "metric"], "additionalProperties": False},
    },
    {
        "name": "paralingo_query_observations",
        "description": "Read earlier observations that agents recorded about this person, optionally by category.",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "category": {"type": "string", "enum": sorted(OBS_CATEGORIES)},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "required": ["speaker_id"], "additionalProperties": False},
    },
    {
        "name": "paralingo_add_observation",
        "description": "Record an evidence-backed observation about how the person speaks or feels, so future agents "
                       "can use it. Cite evidence (metrics, quotes) and be honest about confidence. Use "
                       "'health_indicator' only for non-diagnostic signals; never assert a medical condition.",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "category": {"type": "string", "enum": sorted(OBS_CATEGORIES)},
            "observation": {"type": "string", "description": "One clear sentence."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
            "session_id": _SESSION,
            "utterance_id": {"type": "string"}},
            "required": ["speaker_id", "category", "observation", "confidence"], "additionalProperties": False},
    },
    {
        "name": "paralingo_update_accent",
        "description": "Record an accent / regional-variety estimate and/or observed phonetic markers "
                       "(e.g. 'non-rhotic', 'retroflex t', 'th-stopping'), or an intelligibility score (0-1, e.g. mean ASR confidence).",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "label": {"type": "string", "description": "e.g. 'Indian English (Hindi L1)', 'Scottish English'"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "source": {"type": "string", "description": "Model or agent that produced the estimate."},
            "markers": {"type": "array", "items": {"type": "object", "properties": {
                "marker": {"type": "string"}, "example": {"type": "string"}}, "required": ["marker"]}},
            "intelligibility": {"type": "number", "minimum": 0, "maximum": 1},
            "session_id": _SESSION},
            "required": ["speaker_id"], "additionalProperties": False},
    },
    # ---------------- pipeline tools (normally called by code that owns the audio)
    {
        "name": "paralingo_start_session",
        "pipeline": True,
        "description": "Open a new conversation session for the speaker. Returns session_id.",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "channel": {"type": "string", "description": "phone, web, app..."},
            "context": {"type": "string", "description": "Purpose of the conversation."}},
            "required": ["speaker_id"], "additionalProperties": False},
    },
    {
        "name": "paralingo_analyze_utterance",
        "pipeline": True,
        "description": "Analyse one user turn (audio WAV and/or transcript and/or ASR word timestamps): pitch, loudness, "
                       "pauses, rate, voice quality, fluency, vocabulary, sentiment, emotion and skills; store it and "
                       "return findings plus updated guidance.",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "session_id": _SESSION,
            "transcript": {"type": "string", "description": "Verbatim transcript including fillers if possible."},
            "audio_path": {"type": "string", "description": "Path to a WAV file of this turn (inside the configured audio root)."},
            "word_timestamps": {"type": "array", "items": {"type": "object", "properties": {
                "word": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"},
                "confidence": {"type": "number"}}, "required": ["start", "end"]}},
            "t_start": {"type": "number"}, "t_end": {"type": "number"},
            "response_latency_s": {"type": "number", "description": "Silence between the agent finishing and the user starting."},
            "external": {"type": "object", "description": "Other model outputs, e.g. {\"emotion\": {\"label\": \"angry\", "
                         "\"confidence\": 0.8, \"arousal\": 0.7, \"valence\": -0.6, \"model\": \"ser-x\"}, \"asr_confidence\": 0.92}"}},
            "required": ["speaker_id"], "additionalProperties": False},
    },
    {
        "name": "paralingo_end_session",
        "pipeline": True,
        "description": "Close the session: summarise it, update the personal baseline, trends, skill scores, speaking "
                       "style, speech indicators and guidance.",
        "input_schema": {"type": "object", "properties": {"speaker_id": _SPEAKER, "session_id": _SESSION},
                         "required": ["speaker_id"], "additionalProperties": False},
    },
    {
        "name": "paralingo_set_consent",
        "pipeline": True,
        "description": "Record the person's consent choices. Only call after the person has explicitly agreed.",
        "input_schema": {"type": "object", "properties": {
            "speaker_id": _SPEAKER,
            "recording": {"type": "boolean"}, "voice_analysis": {"type": "boolean"},
            "health_indicators": {"type": "boolean"}, "store_transcripts": {"type": "boolean"},
            "retention_days": {"type": "integer", "minimum": 1}, "granted_via": {"type": "string"}},
            "required": ["speaker_id"], "additionalProperties": False},
    },
]


class SpeechProfileTools:
    def __init__(self, root: str | Path | SpeechProfileStore, *, agent_id: str = "voice-agent",
                 audio_root: str | Path | None = None, include_pipeline_tools: bool = True) -> None:
        self.engine = SpeechProfileEngine(root, agent_id=agent_id)
        self.audio_root = Path(audio_root).resolve() if audio_root else None
        self.include_pipeline_tools = include_pipeline_tools

    # ------------------------------------------------------------------ definitions
    def definitions(self, fmt: str = "anthropic") -> list[dict[str, Any]]:
        specs = [s for s in TOOL_SPECS if self.include_pipeline_tools or not s.get("pipeline")]
        if fmt == "anthropic":
            return [{"name": s["name"], "description": s["description"], "input_schema": s["input_schema"]} for s in specs]
        if fmt == "openai":
            return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                      "parameters": s["input_schema"]}} for s in specs]
        if fmt == "mcp":
            return [{"name": s["name"], "description": s["description"], "inputSchema": s["input_schema"]} for s in specs]
        raise ValueError("fmt must be 'anthropic', 'openai' or 'mcp'")

    # ------------------------------------------------------------------ dispatch
    def execute(self, name: str, tool_input: dict[str, Any] | str) -> str:
        """Run a tool and return a JSON string (errors are returned as {"error": ...}, never raised)."""
        try:
            args = json.loads(tool_input) if isinstance(tool_input, str) else dict(tool_input or {})
            allowed = {s["name"] for s in TOOL_SPECS if self.include_pipeline_tools or not s.get("pipeline")}
            if name not in allowed:
                raise ProfileError(f"unknown tool '{name}'")
            result = getattr(self, "_t_" + name.removeprefix("paralingo_"))(**args)
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str)
        except TypeError as e:
            return json.dumps({"ok": False, "error": f"bad arguments: {e}"})
        except (ProfileError, ValueError, FileNotFoundError) as e:
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    # ------------------------------------------------------------------ implementations
    def _t_get_summary(self, speaker_id: str) -> Any:
        return self.engine.get_summary(speaker_id)

    def _t_get_guidance(self, speaker_id: str) -> Any:
        p = self.engine.get_profile(speaker_id)
        return {"guidance": p["agent_guidance"], "current_emotion": p["assessments"]["emotional_state"].get("current")}

    def _t_read_section(self, speaker_id: str, section: str, last_n: int | None = None) -> Any:
        if section not in SECTIONS:
            raise ProfileError(f"section must be one of {SECTIONS}")
        data = self.engine.get_profile(speaker_id)[section]
        if isinstance(data, list):
            return data[-(last_n or (3 if section == "sessions" else 20)):]
        if section == "trends" and last_n:
            return {k: v[-last_n:] for k, v in data.items()}
        return data

    def _t_get_trend(self, speaker_id: str, metric: str, last_n: int = 10) -> Any:
        return self.engine.get_trend(speaker_id, metric, last_n)

    def _t_query_observations(self, speaker_id: str, category: str | None = None, limit: int = 20) -> Any:
        return self.engine.query_observations(speaker_id, category, limit)

    def _t_add_observation(self, speaker_id: str, **kw: Any) -> Any:
        return {"observation_id": self.engine.add_observation(speaker_id, **kw)}

    def _t_update_accent(self, speaker_id: str, **kw: Any) -> Any:
        return self.engine.update_accent(speaker_id, **kw)

    def _t_start_session(self, speaker_id: str, **kw: Any) -> Any:
        return {"session_id": self.engine.start_session(speaker_id, **kw)}

    def _t_analyze_utterance(self, speaker_id: str, audio_path: str | None = None, **kw: Any) -> Any:
        if audio_path:
            audio_path = str(self._check_audio_path(audio_path))
        return self.engine.analyze_utterance(speaker_id, audio_path=audio_path, **kw)

    def _t_end_session(self, speaker_id: str, session_id: str | None = None) -> Any:
        return self.engine.end_session(speaker_id, session_id)

    def _t_set_consent(self, speaker_id: str, **kw: Any) -> Any:
        return self.engine.set_consent(speaker_id, **kw)

    def _check_audio_path(self, path: str) -> Path:
        p = Path(path)
        if self.audio_root is None:
            raise ProfileError("audio_path not allowed: no audio_root configured")
        p = (self.audio_root / p).resolve() if not p.is_absolute() else p.resolve()
        if self.audio_root != p and self.audio_root not in p.parents:
            raise ProfileError("audio_path is outside the allowed audio root")
        if p.suffix.lower() != ".wav":
            raise ProfileError("only .wav files are supported")
        return p
