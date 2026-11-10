# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Transcript-level analysis: fluency, disfluencies, lexical richness, hedging, sentiment.

Works on plain ASR text. Note that many ASR engines *remove* fillers ("um", "uh")
and repetitions by default - turn on verbatim / disfluency mode (e.g. Deepgram
``filler_words=true``, AssemblyAI ``disfluencies=true``, Whisper with a verbatim
prompt) or these counts will be under-estimated.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "eh", "hmm", "mm", "mhm"}
PHRASE_FILLERS = ["you know", "i mean", "sort of", "kind of", "you see", "basically", "actually", "literally"]
HEDGES = ["maybe", "perhaps", "i think", "i guess", "i suppose", "probably", "possibly", "sort of", "kind of",
          "not sure", "i don't know", "might", "could be", "somewhat", "a bit"]
VAGUE_WORDS = {"thing", "things", "stuff", "thingy", "whatchamacallit", "whatsit", "something", "somewhere"}
REVISION_MARKERS = ["i mean", "sorry", "no wait", "or rather", "let me rephrase", "what i meant", "scratch that",
                    "i meant"]
# word -> polarity. Small, transparent lexicon; agents can override with their own model.
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

_WORD = re.compile(r"[a-zA-Z]+(?:['’][a-zA-Z]+)?|\d+")
_PART_WORD = re.compile(r"\b([a-zA-Z]{1,3})(?:-\1)+-?[a-zA-Z]*\b", re.IGNORECASE)   # b-b-but, st-st-start
_PROLONG = re.compile(r"\b\w*([a-zA-Z])\1{3,}\w*\b")                                  # sooooo, mmmmy
_SENT = re.compile(r"[.!?]+")


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


def _count_phrases(text_l: str, phrases: list[str]) -> dict[str, int]:
    out = {}
    for p in phrases:
        c = len(re.findall(r"\b" + re.escape(p) + r"\b", text_l))
        if c:
            out[p] = c
    return out


def analyze_text(text: str) -> dict[str, Any]:
    """Return fluency, lexical and sentiment features for one utterance transcript."""
    text = text or ""
    part_word = len(_PART_WORD.findall(text))
    # collapse "b-b-buy" -> "buy" so stutter fragments are not double-counted as words / word repetitions
    text = _PART_WORD.sub(lambda m: m.group(0).split("-")[-1] or m.group(1), text)
    text_l = text.lower().replace("’", "'")
    toks = tokenize(text)
    n_words = len(toks)
    content = [t for t in toks if t not in FILLERS]

    # ---- fillers
    filler_counts = Counter(t for t in toks if t in FILLERS)
    filler_counts.update(_count_phrases(text_l, PHRASE_FILLERS))
    # "like" as a filler: when set off by commas or starting a clause
    like_fill = len(re.findall(r"(?:^|[,.]\s*)like\s*,", text_l)) + len(re.findall(r",\s*like\s+", text_l))
    if like_fill:
        filler_counts["like"] += like_fill
    filler_total = sum(filler_counts.values())

    # ---- repetitions
    word_reps = 0
    for a, b in zip(content, content[1:]):
        if a == b and a not in {"very", "really", "no", "bye", "ha", "so", "yeah", "yes", "ok", "okay", "that"}:
            word_reps += 1
    phrase_reps = 0
    for i in range(len(content) - 3):
        if content[i:i + 2] == content[i + 2:i + 4] and content[i] != content[i + 1]:
            phrase_reps += 1
    prolongations = len(_PROLONG.findall(text))
    revisions = sum(_count_phrases(text_l, REVISION_MARKERS).values())
    # ASR-annotated incomplete words, e.g. "wh-" or "some--"
    broken = len(re.findall(r"\b[a-zA-Z]+-(?=\s|$|[,.])", text))

    syllables = sum(count_syllables(t) for t in content)
    # Stuttering-like disfluencies (SLD): part-word reps, monosyllabic whole-word reps, prolongations, blocks.
    sld = part_word + word_reps + prolongations
    # Other (typical) disfluencies: fillers, phrase reps, revisions, broken words.
    other_df = filler_total + phrase_reps + revisions + broken

    # ---- lexical
    types = set(content)
    sentences = [s for s in _SENT.split(text) if s.strip()]
    hedges = _count_phrases(text_l, HEDGES)
    vague = sum(1 for t in content if t in VAGUE_WORDS)
    questions = text.count("?")

    # ---- sentiment (lexicon + negation window + intensifiers)
    pos = neg = 0.0
    for clause in re.split(r"[,.;:!?]+", text_l):  # negation never crosses a clause boundary
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
                pos += pol
            else:
                neg += -pol
    sentiment = (pos - neg) / (pos + neg + 2.0)  # smoothed, in (-1, 1)

    per100w = (lambda v: round(100.0 * v / n_words, 2)) if n_words else (lambda v: 0.0)
    return {
        "fluency": {
            "filler_count": int(filler_total),
            "fillers": dict(filler_counts),
            "filler_rate_per_100w": per100w(filler_total),
            "part_word_repetitions": int(part_word),
            "word_repetitions": int(word_reps),
            "phrase_repetitions": int(phrase_reps),
            "prolongations": int(prolongations),
            "revisions": int(revisions),
            "incomplete_words": int(broken),
            "stuttering_like_per_100_syllables": round(100.0 * sld / syllables, 2) if syllables else 0.0,
            "total_disfluency_per_100w": per100w(sld + other_df),
        },
        "lexical": {
            "word_count": n_words,
            "content_word_count": len(content),
            "syllable_count": syllables,
            "unique_words": len(types),
            "type_token_ratio": round(len(types) / len(content), 3) if content else None,
            "mean_word_length": round(sum(map(len, content)) / len(content), 2) if content else None,
            "sentence_count": len(sentences),
            "mean_sentence_length_words": round(len(content) / max(1, len(sentences)), 2) if content else None,
            "hedges": hedges,
            "hedge_rate_per_100w": per100w(sum(hedges.values())),
            "vague_word_count": vague,
            "question_count": questions,
        },
        "sentiment": {
            "score": round(sentiment, 3),
            "positive_weight": round(pos, 2),
            "negative_weight": round(neg, 2),
            "method": "lexicon-negation-v1",
        },
    }
