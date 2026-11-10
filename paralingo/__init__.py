# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""paralingo - a shared, per-speaker file that AI voice agents read and write to
understand not just WHAT a person says but HOW they say it.

Quick start::

    from paralingo import SpeechProfileEngine
    eng = SpeechProfileEngine("profiles/", agent_id="my-voice-bot")
    sid = eng.start_session("caller-42", channel="phone")
    eng.analyze_utterance("caller-42", session_id=sid, audio_path="turn1.wav", transcript="um, I need help")
    eng.end_session("caller-42", sid)
    print(eng.get_summary("caller-42")["prompt_context"])
"""
from .agent_tools import TOOL_SPECS, SpeechProfileTools
from .audio import analyze_audio, load_wav, pauses_from_word_timestamps
from .engine import SpeechProfileEngine, render_prompt_context
from .language import analyze_text
from .store import (ConcurrencyConflict, ConsentError, LockTimeout, ProfileError, ProfileExists, ProfileNotFound,
                    SpeechProfileStore, new_profile, validate)

__version__ = "1.0.0"
__author__ = "Namash Aggarwal"
__license__ = "Apache-2.0"
__url__ = "https://github.com/namashworks/paralingo"
__all__ = [
    "SpeechProfileEngine", "SpeechProfileStore", "SpeechProfileTools", "TOOL_SPECS",
    "analyze_audio", "analyze_text", "load_wav", "pauses_from_word_timestamps", "render_prompt_context",
    "new_profile", "validate",
    "ProfileError", "ProfileNotFound", "ProfileExists", "ConcurrencyConflict", "ConsentError", "LockTimeout",
]
