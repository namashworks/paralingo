# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""End-to-end demo: simulates three phone calls with one caller and writes
examples/profiles/demo-caller.speech.json, then prints what an agent would see.

    python examples/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from paralingo import SpeechProfileEngine  # noqa: E402

SR = 16000
rng = np.random.default_rng(7)


def voiced(f0: float, dur: float, glide_st: float = 0.0, amp: float = 0.3, jitter: float = 0.002) -> np.ndarray:
    """Very rough speech stand-in: harmonic source with a pitch glide and syllable-rate amplitude modulation."""
    out, t = [], 0.0
    while t < dur:
        f = f0 * 2 ** (glide_st * t / dur / 12)
        T = (1 / f) * (1 + rng.normal(0, jitter))
        tt = np.arange(int(T * SR)) / SR
        out.append(sum((0.6 / k) * np.sin(2 * np.pi * k * tt / T) for k in range(1, 7)))
        t += T
    x = np.concatenate(out)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4.5 * np.arange(len(x)) / SR) ** 2
    return amp * x * env


def gap(d: float) -> np.ndarray:
    return rng.normal(0, 0.0007, int(d * SR))


def turn(parts: list[tuple[float, float, float]], amp: float = 0.3) -> np.ndarray:
    """parts: (f0, relative_duration, glide) chunks separated by short pauses.
    Durations are rescaled later in main() so the speaking rate matches the transcript."""
    return parts, amp  # type: ignore[return-value]


def render(parts, amp: float, n_words: int, wps: float) -> np.ndarray:
    total = n_words / wps
    rel = sum(d for _, d, _ in parts)
    chunks = [gap(0.25)]
    for i, (f0, d, g) in enumerate(parts):
        chunks.append(voiced(f0, total * d / rel, g, amp))
        chunks.append(gap(0.35 if i < len(parts) - 1 else 0.25))
    return np.concatenate(chunks)


CALLS = [
    ("Order status", [
        ("Hi, um, I'm calling about my order, it hasn't arrived yet.", turn([(205, 1.6, 3), (200, 1.4, -3)]), 0.7),
        ("It's, uh, order number four five two... I think it was placed last week.", turn([(210, 1.2, 2), (198, 1.8, -4)]), 1.1),
        ("Okay great, thank you, that's really helpful.", turn([(220, 1.5, 4)]), 0.5),
        ("No, that's all. Thanks so much, bye!", turn([(215, 1.2, 2)]), 0.4),
    ]),
    ("Billing question", [
        ("Hello, I have a question about my bill this month.", turn([(200, 1.7, -2)]), 0.6),
        ("I was charged twice, I think, for the same thing.", turn([(205, 1.5, 3), (200, 1.0, -3)]), 0.9),
        ("Okay, so it will be refunded? That's good.", turn([(212, 1.4, 5)]), 0.5),
        ("Perfect, thank you for your help.", turn([(208, 1.2, -2)]), 0.4),
    ]),
    ("Delivery problem (frustrated)", [
        ("This is the third time I'm calling! The package is still not here.", turn([(265, 1.8, 6), (255, 1.5, -5)], amp=0.7), 0.2),
        ("No, no, I already did that, it doesn't work, it is ridiculous.", turn([(270, 1.6, 7)], amp=0.75), 0.1),
        ("I'm really frustrated, I need this today, I am so annoyed.", turn([(260, 1.7, -6), (250, 1.2, 5)], amp=0.7), 0.2),
        ("Fine. Just send it again, please.", turn([(240, 1.3, -4)], amp=0.6), 0.6),
    ]),
]


def main() -> None:
    profiles = ROOT / "profiles"
    profiles.mkdir(exist_ok=True)
    target = profiles / "demo-caller.speech.json"
    if target.exists():
        target.unlink()

    eng = SpeechProfileEngine(profiles, agent_id="demo-support-bot")
    eng.ensure_profile("demo-caller", display_name="Demo Caller", languages=["en-IN"])
    eng.set_consent("demo-caller", recording=True, voice_analysis=True, health_indicators=True,
                    store_transcripts=True, retention_days=180, granted_via="verbal-consent-recorded")

    for ctx, turns in CALLS:
        sid = eng.start_session("demo-caller", channel="phone", context=ctx)
        wps = 3.4 if "frustrated" in ctx else 2.5   # ~200 wpm when upset, ~150 wpm normally
        for text, (parts, amp), latency in turns:
            audio = render(parts, amp, len(text.split()), wps)
            r = eng.analyze_utterance("demo-caller", session_id=sid, samples=audio, sample_rate=SR,
                                      transcript=text, response_latency_s=latency,
                                      external={"asr_confidence": 0.9})
            print(f"[{ctx[:22]:22}] {r['emotion']['label']:<10} a={r['emotion']['arousal']:+.2f} "
                  f"v={r['emotion']['valence']:+.2f}  flags={r['flags']}")
        # what an LLM agent might add after reasoning over the call
        if "frustrated" in ctx:
            eng.add_observation("demo-caller", category="emotion", confidence=0.7, session_id=sid,
                                observation="Caller escalates quickly when asked to repeat steps already tried.",
                                evidence=["'I already did that'", "pitch & loudness well above baseline"],
                                reasoning="Repeated-contact frustration; skip basic troubleshooting next time.")
        eng.end_session("demo-caller", sid)

    eng.update_accent("demo-caller", label="Indian English", confidence=0.65, source="accent-id-model-x",
                      markers=[{"marker": "retroflex /t/, /d/", "example": "order"},
                               {"marker": "syllable-timed rhythm"}], intelligibility=0.9)

    s = eng.get_summary("demo-caller")
    print("\n=== prompt_context (inject into the voice agent's system prompt) ===\n")
    print(s["prompt_context"])
    print("\n=== communication skills ===")
    print(json.dumps(s["communication_skills"], indent=2))
    print(f"\nProfile file: {target}  ({target.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
