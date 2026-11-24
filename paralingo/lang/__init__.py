# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Language packs, language detection and code-switching segmentation.

Built in: English (``en``), Mandarin (``zh``), Japanese (``ja``) and Korean (``ko``). Other languages get
counts (words, syllables, repetitions) through a generic fallback, without fillers, hedges or sentiment.
Add a language with :func:`register_pack`.
"""
from __future__ import annotations

import re
from collections import Counter

from .base import Counts, LanguagePack
from .en import PACK as EN
from .ja import PACK as JA
from .ko import PACK as KO
from .zh import PACK as ZH

__all__ = ["Counts", "LanguagePack", "PACKS", "get_pack", "register_pack", "normalize_language",
           "detect_language", "segment"]


class _Generic(LanguagePack):
    """Fallback for languages without a pack: whitespace words and vowel-group syllables only."""

    def analyze(self, text: str) -> Counts:
        toks = re.findall(r"\w+", text.lower())
        c = Counts(words=float(len(toks)), content_words=float(len(toks)), tokens=toks)
        c.syllables = sum(max(1, len(re.findall(r"[aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ]+", t))) for t in toks)
        c.word_repetitions = sum(1 for a, b in zip(toks, toks[1:]) if a == b)
        c.questions = text.count("?") + text.count("？")
        return c


GENERIC = _Generic(code="und", name="Other", word_unit="word", pace_unit="words",
                   sources=["No language pack: counts only, with English-like rate norms"])

PACKS: dict[str, LanguagePack] = {p.code: p for p in (EN, ZH, JA, KO)}
_ALIASES = {"cmn": "zh", "yue": "zh", "jpn": "ja", "kor": "ko", "eng": "en", "zho": "zh", "chi": "zh"}


def register_pack(pack: LanguagePack) -> None:
    """Add or replace a language pack (e.g. a community pack for Hindi)."""
    PACKS[pack.code] = pack


def normalize_language(tag: str | None) -> str | None:
    """'en-US' -> 'en', 'zh-Hant-TW' / 'cmn' -> 'zh'. Returns the base code (lower case) or None."""
    if not tag:
        return None
    base = re.split(r"[-_]", tag.strip().lower())[0]
    return _ALIASES.get(base, base) or None


def get_pack(code: str | None) -> LanguagePack:
    return PACKS.get(normalize_language(code) or "", GENERIC)


# --------------------------------------------------------------------------- scripts
def _script(ch: str) -> str | None:
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "kana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or o == 0x3005:
        return "han"
    if ch.isalpha() and (o < 0x0250 or 0x1E00 <= o <= 0x1EFF):
        return "latin"
    if ch.isalpha():
        return "other"
    return None


def detect_language(text: str, hint: str | None = None) -> str | None:
    """Main language of a text from its script: kana -> ja, Hangul -> ko, Chinese characters -> zh (or ja when
    the hint or the text says Japanese), Latin -> en (or the hint, if it is another Latin-script language)."""
    segs = segment(text, hint)
    if not segs:
        return normalize_language(hint)
    weight: Counter = Counter()
    for lang, s in segs:
        weight[lang] += sum(1 for ch in s if _script(ch))
    return weight.most_common(1)[0][0]


def segment(text: str, hint: str | None = None) -> list[tuple[str, str]]:
    """Split a transcript into single-language runs, e.g. '我觉得这个 deadline 太紧了' ->
    [('zh', '我觉得这个 '), ('en', 'deadline '), ('zh', '太紧了')]. Spaces, digits and punctuation stay with
    the run they follow. A lone Latin letter inside Asian text (A套餐) is not treated as a switch."""
    if not text or not text.strip():
        return []
    hint_code = normalize_language(hint)
    has_kana = any(_script(ch) == "kana" for ch in text)
    han_lang = "ja" if (has_kana or hint_code == "ja") else "zh"
    latin_lang = hint_code if hint_code and hint_code not in ("zh", "ja", "ko") else "en"
    to_lang = {"hangul": "ko", "kana": "ja", "han": han_lang, "latin": latin_lang,
               "other": hint_code or "und"}

    runs: list[list] = []          # [lang, text, letter_count]
    pending = ""
    for ch in text:
        sc = _script(ch)
        if sc is None:
            if runs:
                runs[-1][1] += ch
            else:
                pending += ch
            continue
        lang = to_lang[sc]
        if runs and runs[-1][0] == lang:
            runs[-1][1] += ch
            runs[-1][2] += 1
        else:
            runs.append([lang, pending + ch, 1])
            pending = ""
    if not runs:
        return []
    # absorb single Latin letters between runs of another language
    merged: list[list] = []
    for r in runs:
        if merged and r[2] < 2 and r[0] == latin_lang and r[0] != merged[-1][0]:
            merged[-1][1] += r[1]
            continue
        if merged and merged[-1][0] == r[0]:
            merged[-1][1] += r[1]
            merged[-1][2] += r[2]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][2] < 2 and merged[0][0] == latin_lang:
        merged[1][1] = merged[0][1] + merged[1][1]
        merged.pop(0)
    return [(lang, s) for lang, s, _ in merged]
