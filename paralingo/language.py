# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Transcript-level analysis: fluency, disfluencies, lexical richness, hedging, sentiment and language use.

Multilingual: English, Mandarin, Japanese and Korean have full language packs (``paralingo.lang``), and a
transcript that switches language mid-sentence is split and each part analysed with its own pack. Other
languages get counts only.

Works on plain ASR text. Note that many ASR engines *remove* fillers ("um", "嗯", "えーと", "음") and
repetitions by default - turn on verbatim / disfluency mode (e.g. Deepgram ``filler_words=true``,
AssemblyAI ``disfluencies=true``, Azure ``post-processing: none``, Whisper with a verbatim prompt) or these
counts will be under-estimated.
"""
from __future__ import annotations

import re
from typing import Any

from .lang import PACKS, Counts, get_pack, normalize_language, segment
from .lang.en import (FILLERS, HEDGES, INTENSIFIERS, NEGATIVE, NEGATORS, PHRASE_FILLERS, POSITIVE,  # noqa: F401
                      REVISION_MARKERS, VAGUE_WORDS, count_syllables, tokenize)

_SENT = re.compile(r"[.!?。！？]+")
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_KANA_HANGUL = re.compile(r"[ぁ-ヿ가-힣]")


def needs_language_hint(text: str) -> bool:
    """True when the script alone can't tell the language: Chinese characters with no kana (could be
    Mandarin or Japanese)."""
    return bool(_HAN.search(text or "")) and not _KANA_HANGUL.search(text or "")


def analyze(text: str, language: str | None = None) -> tuple[dict[str, Any], Counts]:
    """Analyse a transcript; returns (result, raw counts). ``language`` is an optional hint such as the
    language your ASR detected ("ja-JP"); the script of the text decides where the two disagree."""
    text = text or ""
    segs = segment(text, language)
    total = Counts(tokens=[])
    per_lang: dict[str, float] = {}
    speak_s = artic_s = 0.0
    methods, units = set(), set()
    for lang, s in segs:
        pack = get_pack(lang)
        c = pack.analyze(s)
        total.add(c)
        per_lang[lang] = per_lang.get(lang, 0.0) + (c.words or len(s.strip()))
        speak_s += pack.pace_units(c) / pack.speaking_rate_norm
        artic_s += c.syllables / pack.articulation_norm_sps
        methods.add({"word": "whitespace"}.get(pack.word_unit, pack.word_unit))
        units.add(pack.syllable_unit)

    n_words = total.words
    content = total.content_words
    fl_total = sum(total.fillers.values())
    sld = total.part_word_repetitions + total.word_repetitions + total.prolongations
    other_df = fl_total + total.phrase_repetitions + total.revisions + total.incomplete_words
    toks = total.tokens if segs else []      # None when a language without word segmentation is present
    sentences = [s for s in _SENT.split(text) if s.strip()]

    def per100w(v: float) -> float:
        return round(100.0 * v / n_words, 2) if n_words else 0.0

    result = {
        "fluency": {
            "filler_count": int(fl_total),
            "fillers": dict(total.fillers),
            "filler_rate_per_100w": per100w(fl_total),
            "part_word_repetitions": int(total.part_word_repetitions),
            "word_repetitions": int(total.word_repetitions),
            "phrase_repetitions": int(total.phrase_repetitions),
            "prolongations": int(total.prolongations),
            "revisions": int(total.revisions),
            "incomplete_words": int(total.incomplete_words),
            "stuttering_like_per_100_syllables": round(100.0 * sld / total.syllables, 2) if total.syllables else 0.0,
            "total_disfluency_per_100w": per100w(sld + other_df),
        },
        "lexical": {
            "word_count": int(round(n_words)),
            "content_word_count": int(round(content)),
            "syllable_count": int(total.syllables),
            "unique_words": len(set(toks)) if toks is not None else None,
            "type_token_ratio": round(len(set(toks)) / len(toks), 3) if toks else None,
            "mean_word_length": round(sum(map(len, toks)) / len(toks), 2) if toks else None,
            "sentence_count": len(sentences),
            "mean_sentence_length_words": round(content / max(1, len(sentences)), 2) if content else None,
            "hedges": dict(total.hedges),
            "hedge_rate_per_100w": per100w(sum(total.hedges.values())),
            "vague_word_count": int(total.vague_words),
            "question_count": int(total.questions),
            "word_count_method": methods.pop() if len(methods) == 1 else ("mixed" if methods else "whitespace"),
            "syllable_unit": units.pop() if len(units) == 1 else ("mixed" if units else "syllable"),
        },
        "sentiment": {
            "score": round((total.positive - total.negative) / (total.positive + total.negative + 2.0), 3),
            "positive_weight": round(total.positive, 2),
            "negative_weight": round(total.negative, 2),
            "method": "lexicon-negation-v1",
        },
        "language": language_block(segs, per_lang, language, speak_s, artic_s),
    }
    return result, total


def language_block(segs: list[tuple[str, str]], per_lang: dict[str, float], hint: str | None,
                   speak_s: float, artic_s: float) -> dict[str, Any]:
    tot = sum(per_lang.values())
    shares = {k: round(v / tot, 3) for k, v in sorted(per_lang.items(), key=lambda kv: -kv[1])} if tot else {}
    # The main ("matrix") language frames the sentence: it is the one the speaker keeps returning to, so it
    # has the most runs ("这个 project 的 budget 我们 next week 再 review 吧" is Chinese with English terms).
    # Ties go to the language with more words.
    runs: dict[str, int] = {}
    for lang, _ in segs:
        runs[lang] = runs.get(lang, 0) + 1
    primary = (max(per_lang, key=lambda k: (runs.get(k, 0), per_lang[k])) if per_lang else None) \
        or normalize_language(hint)
    embedded = [lang for lang in shares if lang != primary]
    terms: list[str] = []
    for lang, s in segs:
        t = re.sub(r"^[\W_]+|[\W_]+$", "", s.strip())
        if lang != primary and t and len(t) <= 40 and t not in terms:
            terms.append(t)
    return {
        "primary": primary,
        "supported": primary in PACKS if primary else False,
        "asr_hint": hint,
        "shares": shares,
        "code_switching": ({"switches": len(segs) - 1, "embedded_languages": embedded,
                            "embedded_terms": terms[:10]} if len(shares) > 1 else None),
        # how long a typical speaker of these languages would take: used to judge pace fairly across languages
        "typical_duration_s": {"speaking": round(speak_s, 3), "articulating": round(artic_s, 3)},
    }


def speech_rates(text_result: dict[str, Any] | None, span_s: float | None, speech_s: float | None) -> dict[str, Any]:
    """Speaking and articulation rates for one utterance.

    ``pace_ratio`` and ``articulation_ratio`` compare the speaker with typical speakers of the language(s)
    they used (1.0 = typical), so a fast Japanese speaker and a fast English speaker both read as "fast".
    For English, ``pace_ratio`` is words per minute / 150 and ``articulation_ratio`` is syllables/s / 4.8.
    """
    out: dict[str, Any] = {"words_per_minute": None, "articulation_rate_sps": None,
                           "pace_ratio": None, "articulation_ratio": None}
    if not text_result:
        return out
    typical = (text_result.get("language") or {}).get("typical_duration_s") or {}
    if span_s and span_s > 0.3:
        out["words_per_minute"] = round(text_result["lexical"]["content_word_count"] / span_s * 60.0, 1)
        if typical.get("speaking"):
            out["pace_ratio"] = round(typical["speaking"] / span_s, 3)
    if speech_s and speech_s > 0.3:
        out["articulation_rate_sps"] = round(text_result["lexical"]["syllable_count"] / speech_s, 2)
        if typical.get("articulating"):
            out["articulation_ratio"] = round(typical["articulating"] / speech_s, 3)
    return out


def analyze_text(text: str, language: str | None = None) -> dict[str, Any]:
    """Return fluency, lexical, sentiment and language features for one utterance transcript."""
    return analyze(text, language)[0]
