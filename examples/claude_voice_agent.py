# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Example: a Claude-powered voice agent's "brain" that reads and writes the speech profile.

The speech-to-text / text-to-speech parts of your stack (Twilio, LiveKit, Pipecat, Vapi,
Deepgram, ElevenLabs...) stay as they are. Per user turn your pipeline:

  1. calls tools.execute("paralingo_analyze_utterance", {...audio, transcript...})
     -> the profile file is updated and fresh guidance comes back;
  2. sends the transcript to Claude, which has the read/observe tools and the
     profile's prompt_context in its system prompt.

    pip install anthropic numpy
    python examples/claude_voice_agent.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paralingo import SpeechProfileTools  # noqa: E402

PROFILES = Path(__file__).resolve().parent / "profiles"
SPEAKER = "demo-caller"  # run examples/demo.py first so this profile has history

pipeline = SpeechProfileTools(PROFILES, agent_id="voice-pipeline", audio_root=Path.cwd())
llm_tools = SpeechProfileTools(PROFILES, agent_id="claude-agent", include_pipeline_tools=False)
client = anthropic.Anthropic()


def system_prompt() -> str:
    summary = json.loads(llm_tools.execute("paralingo_get_summary", {"speaker_id": SPEAKER}))
    ctx = summary["result"]["prompt_context"] if summary["ok"] else "No speech profile yet."
    return (
        "You are a friendly phone support agent. Your replies are spoken aloud, so keep them short and natural.\n"
        "You can read and update the caller's speech profile with the paralingo_* tools. When you notice "
        "something durable about HOW the caller communicates (preferences, emotional patterns, accent features), "
        "record it with paralingo_add_observation, citing evidence. Never tell the caller what the profile "
        "says about their emotions or speech, and never suggest they have a medical condition.\n\n" + ctx
    )


def respond(messages: list[dict]) -> str:
    """One agent turn with a manual tool loop."""
    while True:
        response = client.beta.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            system=system_prompt(),
            tools=llm_tools.definitions("anthropic"),
            messages=messages,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # retry on another model if a safety classifier declines
        )
        if response.stop_reason == "refusal":
            return "Sorry, I can't help with that one."
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")
        results = []
        for block in response.content:
            if block.type == "tool_use":
                out = llm_tools.execute(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out,
                                "is_error": not json.loads(out)["ok"]})
        messages.append({"role": "user", "content": results})


def main() -> None:
    sid = json.loads(pipeline.execute("paralingo_start_session",
                                      {"speaker_id": SPEAKER, "channel": "phone", "context": "support"}))["result"]["session_id"]
    messages: list[dict] = []
    print("Type what the caller says (empty line to hang up).")
    while (text := input("caller> ").strip()):
        # In production pass audio_path / word_timestamps from your ASR too.
        analysis = json.loads(pipeline.execute("paralingo_analyze_utterance",
                                               {"speaker_id": SPEAKER, "session_id": sid, "transcript": text}))
        emo = analysis["result"]["emotion"] if analysis["ok"] else {}
        print(f"   (emotion estimate: {emo.get('label')} {emo.get('confidence')})")
        messages.append({"role": "user", "content": text})
        print("agent>", respond(messages))
    print(json.loads(pipeline.execute("paralingo_end_session", {"speaker_id": SPEAKER, "session_id": sid})))


if __name__ == "__main__":
    main()
