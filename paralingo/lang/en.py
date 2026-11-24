# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""English."""
from __future__ import annotations

import re
from collections import Counter

from .base import Counts, LanguagePack, count_phrases

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "eh", "hmm", "mm", "mhm"}
PHRASE_FILLERS = ["you know", "i mean", "sort of", "kind of", "you see", "basically", "actually", "literally"]
HEDGES = ["maybe", "perhaps", "i think", "i guess", "i suppose", "probably", "possibly", "sort of", "kind of",
          "not sure", "i don't know", "might", "could be", "somewhat", "a bit"]
VAGUE_WORDS = {"thing", "things", "stuff", "thingy", "whatchamacallit", "whatsit", "something", "somewhere"}
REVISION_MARKERS = ["i mean", "sorry", "no wait", "or rather", "let me rephrase", "what i meant", "scratch that",
                    "i meant"]
POSITIVE = set("""good great happy glad love like enjoy excellent amazing awesome wonderful fantastic thanks thank
pleased nice perfect better best helpful appreciate relieved excited delighted calm comfortable easy sure
yes yeah okay ok cool brilliant lovely satisfied""".split())
NEGATIVE = set("""bad terrible awful hate angry upset sad unhappy annoyed annoying frustrated frustrating
disappointed disappointing worried anxious scared afraid nervous stressed tired exhausted problem problems
issue issues wrong broken fail failed failing worse worst pain hurt hurts sick confused difficult hard
unfortunately ridiculous useless never cannot can't won't nobody lonely depressed horrible upsetting""".split())
NEGATORS = {"not", "no", "never", "don't", "dont", "isn't", "isnt", "wasn't", "wasnt", "didn't", "didnt", "can't",
            "cant", "won't", "wont", "hardly", "nothing"}
INTENSIFIERS = {"very", "really", "so", "extremely", "totally", "absolutely", "super"}
WORRY = {"worried", "anxious", "scared", "afraid", "nervous", "stressed", "panic", "panicking"}
TIRED = {"tired", "exhausted", "sleepy", "drained", "worn"}
NO_REPEAT = {"very", "really", "no", "bye", "ha", "so", "yeah", "yes", "ok", "okay", "that"}

_WORD = re.compile(r"[a-zA-Z]+(?:['’][a-zA-Z]+)?|\d+")
_PART_WORD = re.compile(r"\b([a-zA-Z]{1,3})(?:-\1)+-?[a-zA-Z]*\b", re.IGNORECASE)   # b-b-but, st-st-start
_PROLONG = re.compile(r"\b\w*([a-zA-Z])\1{3,}\w*\b")                                  # sooooo, mmmmy
_CLAUSE = re.compile(r"[,.;:!?]+")


def tokenize(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in _WORD.findall(text or "")]


def count_syllables(word: str) -> int:
    w = word.lower()
    if not w.isalpha():
        return max(1, len(w) // 2)
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


class English(LanguagePack):
    def analyze(self, text: str) -> Counts:
        c = Counts()
        c.part_word_repetitions = len(_PART_WORD.findall(text))
        # collapse "b-b-buy" -> "buy" so stutter fragments are not double-counted as words / word repetitions
        text = _PART_WORD.sub(lambda m: m.group(0).split("-")[-1] or m.group(1), text)
        text_l = text.lower().replace("’", "'")
        toks = tokenize(text)
        content = [t for t in toks if t not in FILLERS]
        c.words, c.content_words, c.tokens = float(len(toks)), float(len(content)), content

        fillers = Counter(t for t in toks if t in FILLERS)
        fillers.update(count_phrases(text_l, PHRASE_FILLERS, word_boundaries=True))
        like_fill = len(re.findall(r"(?:^|[,.]\s*)like\s*,", text_l)) + len(re.findall(r",\s*like\s+", text_l))
        if like_fill:
            fillers["like"] += like_fill
        c.fillers = fillers

        c.word_repetitions = sum(1 for a, b in zip(content, content[1:]) if a == b and a not in NO_REPEAT)
        c.phrase_repetitions = sum(1 for i in range(len(content) - 3)
                                   if content[i:i + 2] == content[i + 2:i + 4] and content[i] != content[i + 1])
        c.prolongations = len(_PROLONG.findall(text))
        c.revisions = sum(count_phrases(text_l, REVISION_MARKERS, word_boundaries=True).values())
        c.incomplete_words = len(re.findall(r"\b[a-zA-Z]+-(?=\s|$|[,.])", text))   # "wh-", "some--"
        c.syllables = sum(count_syllables(t) for t in content)
        c.hedges = count_phrases(text_l, HEDGES, word_boundaries=True)
        c.vague_words = sum(1 for t in content if t in VAGUE_WORDS)
        c.questions = text.count("?")

        for clause in _CLAUSE.split(text_l):          # negation never crosses a clause boundary
            ctoks = tokenize(clause)
            for i, t in enumerate(ctoks):
                pol = 1 if t in POSITIVE else -1 if t in NEGATIVE else 0
                if not pol:
                    continue
                window = ctoks[max(0, i - 3):i]
                if any(w in NEGATORS for w in window) and t not in NEGATORS:
                    pol = -pol * 0.7
                if window and window[-1] in INTENSIFIERS:
                    pol *= 1.5
                if pol > 0:
                    c.positive += pol
                else:
                    c.negative -= pol
        tset = set(toks)
        c.emotion_words = ({"worried"} if tset & WORRY else set()) | ({"tired"} if tset & TIRED else set())
        return c


PACK = English(
    code="en", name="English", word_unit="word", syllable_unit="syllable",
    pace_unit="words", speaking_rate_norm=2.5, articulation_norm_sps=4.8,
    continuation={"and", "but", "so", "because", "cause", "or", "then", "that", "which", "who", "if", "when",
                  "while", "to", "the", "a", "an", "of", "with", "for", "my", "your", "our", "is", "was", "are",
                  "were", "um", "umm", "uh", "uhh", "er", "erm", "like", "well", "also", "about", "from", "in",
                  "on", "at", "it's", "i'm", "we", "i", "and then", "you know", "i mean", "kind of", "sort of"},
    complete=set(),
    question={"?"},
    backchannel={"yeah", "yes", "yep", "yup", "uh-huh", "uh huh", "mm-hmm", "mhm", "mm", "hmm", "ok", "okay",
                 "right", "sure", "i see", "got it", "alright", "all right", "true", "exactly", "cool"},
    sources=["150 words per minute and 4.8 syllables/s: typical conversational English (wide prior)"],
)
