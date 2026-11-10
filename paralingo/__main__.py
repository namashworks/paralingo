# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Command-line access to speech profiles.

    python -m paralingo --root profiles init caller-42 --name "Asha"
    python -m paralingo --root profiles consent caller-42 --health-indicators yes
    python -m paralingo --root profiles analyze caller-42 --wav turn.wav --text "um, I need help"
    python -m paralingo --root profiles end caller-42
    python -m paralingo --root profiles summary caller-42
    python -m paralingo --root profiles tool paralingo_get_trend '{"speaker_id":"caller-42","metric":"hnr_db"}'
    python -m paralingo validate profiles/caller-42.speech.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent_tools import SpeechProfileTools
from .engine import SpeechProfileEngine
from .store import validate


def _yn(v: str) -> bool:
    return v.lower() in {"y", "yes", "true", "1"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="paralingo")
    ap.add_argument("--root", default="profiles")
    ap.add_argument("--agent-id", default="cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init"); p.add_argument("speaker"); p.add_argument("--name"); p.add_argument("--lang", nargs="*")
    p = sub.add_parser("consent"); p.add_argument("speaker")
    for f in ("recording", "voice-analysis", "health-indicators", "store-transcripts"):
        p.add_argument(f"--{f}")
    p.add_argument("--retention-days", type=int)
    p = sub.add_parser("analyze"); p.add_argument("speaker"); p.add_argument("--wav"); p.add_argument("--text")
    p.add_argument("--session"); p.add_argument("--latency", type=float)
    p = sub.add_parser("end"); p.add_argument("speaker"); p.add_argument("--session")
    p = sub.add_parser("summary"); p.add_argument("speaker"); p.add_argument("--text-only", action="store_true")
    p = sub.add_parser("tool"); p.add_argument("name"); p.add_argument("json_input")
    p = sub.add_parser("tools"); p.add_argument("--format", default="anthropic", choices=["anthropic", "openai", "mcp"])
    p = sub.add_parser("validate"); p.add_argument("file")
    a = ap.parse_args(argv)

    eng = SpeechProfileEngine(a.root, agent_id=a.agent_id)
    out: object
    if a.cmd == "init":
        out = eng.ensure_profile(a.speaker, display_name=a.name, languages=a.lang)["profile_id"]
    elif a.cmd == "consent":
        kw = {k: _yn(getattr(a, k)) for k in ("recording", "voice_analysis", "health_indicators", "store_transcripts")
              if getattr(a, k) is not None}
        if a.retention_days:
            kw["retention_days"] = a.retention_days
        out = eng.set_consent(a.speaker, **kw)
    elif a.cmd == "analyze":
        out = eng.analyze_utterance(a.speaker, session_id=a.session, audio_path=a.wav, transcript=a.text,
                                    response_latency_s=a.latency)
    elif a.cmd == "end":
        out = eng.end_session(a.speaker, a.session)
    elif a.cmd == "summary":
        s = eng.get_summary(a.speaker)
        if a.text_only:
            print(s["prompt_context"])
            return 0
        out = s
    elif a.cmd == "tool":
        print(SpeechProfileTools(a.root, agent_id=a.agent_id, audio_root=".").execute(a.name, a.json_input))
        return 0
    elif a.cmd == "tools":
        out = SpeechProfileTools(a.root).definitions(a.format)
    else:  # validate
        errs = validate(json.loads(Path(a.file).read_text("utf-8")))
        print("valid" if not errs else "\n".join(errs))
        return 1 if errs else 0
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
