# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""The language-pack interface.

A language pack teaches Paralingo one language: how to count words and syllables, which words are fillers,
hedges, positive or negative, and which endings mean "not finished" in real-time turn-taking. Packs are
plain Python with no dependencies; to add a language, copy ``zh.py`` or ``ko.py`` and register it with
:func:`paralingo.lang.register_pack`.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Counts:
    """Raw counts for one stretch of text in one language. Packs fill these in; they are summed across
    code-switched segments and turned into rates by :func:`paralingo.language.analyze_text`."""

    words: float = 0.0                    # words, or an estimate for languages written without spaces
    content_words: float = 0.0            # words minus hesitation fillers
    syllables: int = 0                    # syllables (morae for Japanese)
    fillers: Counter = field(default_factory=Counter)
    part_word_repetitions: int = 0
    word_repetitions: int = 0
    phrase_repetitions: int = 0
    prolongations: int = 0
    revisions: int = 0
    incomplete_words: int = 0
    hedges: Counter = field(default_factory=Counter)
    vague_words: int = 0
    questions: int = 0
    positive: float = 0.0
    negative: float = 0.0
    tokens: list[str] | None = None       # real word tokens, if the pack can segment words (used for vocabulary)
    emotion_words: set[str] = field(default_factory=set)   # canonical cues: "worried", "tired"

    def add(self, o: "Counts") -> None:
        for k in ("words", "content_words", "syllables", "part_word_repetitions", "word_repetitions",
                  "phrase_repetitions", "prolongations", "revisions", "incomplete_words", "vague_words",
                  "questions", "positive", "negative"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.fillers.update(o.fillers)
        self.hedges.update(o.hedges)
        self.emotion_words |= o.emotion_words
        self.tokens = None if self.tokens is None or o.tokens is None else self.tokens + o.tokens


@dataclass
class LanguagePack:
    """Everything Paralingo needs to know about one language.

    Rate norms are typical conversational values and serve only as priors: once a speaker has a personal
    baseline (5+ turns), their own rates are used instead.
    """

    code: str                              # ISO 639-1, e.g. "zh"
    name: str
    word_unit: str = "word"                # "word", "eojeol" (Korean spacing unit) or "estimated"
    syllable_unit: str = "syllable"        # "syllable" or "mora"
    pace_unit: str = "words"               # unit of speaking_rate_norm: "words" or "syllables"/"morae"
    speaking_rate_norm: float = 2.5        # pace units per second, pauses included (2.5 words/s = 150 wpm)
    articulation_norm_sps: float = 4.8     # syllables (morae) per second of actual speech
    prosody_scale: float = 1.0             # typical pitch variability relative to English (tones raise it)
    tonal: bool = False
    # real-time turn-taking cues (see paralingo.realtime)
    continuation: set[str] = field(default_factory=set)
    complete: set[str] = field(default_factory=set)
    question: set[str] = field(default_factory=set)
    backchannel: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)

    def analyze(self, text: str) -> Counts:     # pragma: no cover - overridden
        raise NotImplementedError

    def pace_units(self, c: Counts) -> float:
        return c.content_words if self.pace_unit == "words" else float(c.syllables)


# --------------------------------------------------------------------------- helpers for packs
CLAUSE_SPLIT = re.compile(r"[,.;:!?，。；：！？、…]+")
SENTENCE_SPLIT = re.compile(r"[.!?。！？]+")


def count_phrases(text: str, phrases: Iterable[str], *, word_boundaries: bool) -> Counter:
    """Count non-overlapping occurrences, longest phrases first (so 我觉得 is not also counted as 觉得)."""
    out: Counter = Counter()
    taken = [False] * len(text)
    for p in sorted(set(phrases), key=len, reverse=True):
        pat = (r"\b" + re.escape(p) + r"\b") if word_boundaries else re.escape(p)
        for m in re.finditer(pat, text):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            out[p] += 1
    return out


def scan_lexicon(text: str, lexicon: dict[str, int]) -> list[tuple[int, int, int]]:
    """Longest-first, non-overlapping lexicon matches in unspaced text: (start, end, polarity)."""
    hits: list[tuple[int, int, int]] = []
    taken = [False] * len(text)
    for w in sorted(lexicon, key=len, reverse=True):
        start = 0
        while (i := text.find(w, start)) >= 0:
            if not any(taken[i:i + len(w)]):
                hits.append((i, i + len(w), lexicon[w]))
                for j in range(i, i + len(w)):
                    taken[j] = True
            start = i + 1
    return sorted(hits)


def score_sentiment(text: str, lexicon: dict[str, int], *, negated, intensified) -> tuple[float, float]:
    """Lexicon sentiment for unspaced or agglutinative text. ``negated(clause, start, end)`` and
    ``intensified(clause, start)`` are language-specific; negation never crosses a clause boundary."""
    pos = neg = 0.0
    for clause in CLAUSE_SPLIT.split(text):
        for s, e, pol in scan_lexicon(clause, lexicon):
            v = float(pol)
            if negated(clause, s, e):
                v = -v * 0.7
            if intensified(clause, s):
                v *= 1.5
            if v > 0:
                pos += v
            else:
                neg -= v
    return pos, neg
