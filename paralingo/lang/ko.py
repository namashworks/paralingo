# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Korean.

Words are eojeol (the units between spaces, e.g. 저는), and each Hangul block is one syllable.
"""
from __future__ import annotations

import re
from collections import Counter

from .base import Counts, LanguagePack, count_phrases, score_sentiment

_TOKEN = re.compile(r"[가-힣]+|\d+")
_HANGUL = re.compile(r"[가-힣]")
FILLERS_ALWAYS = {"음", "음음", "으음", "어", "어어", "으", "에", "아", "엄", "흠", "뭐랄까", "있잖아", "있잖아요"}
# also ordinary words ("that", "what", "now", "a bit"), so fillers only before a pause
FILLERS_BEFORE_PAUSE = {"그", "저", "뭐", "이제", "막", "그게", "약간", "좀", "그러니까", "그니까"}
HEDGES = ["아마도", "아마", "혹시", "것 같아요", "것 같아", "것 같습니다", "것 같은데", "거 같아요", "거 같아",
          "거 같은데", "잘 모르겠", "모르겠어요", "모르겠는데", "모르겠습니다", "좀", "약간", "어느 정도", "글쎄",
          "일단", "듯해요", "듯합니다"]
VAGUE = ["그런 거", "그런거", "뭔가", "이런 거", "저런 거", "거시기", "어떤 거", "등등", "같은 거", "이것저것"]
REVISIONS = ["아니 그게", "아니 아니", "그게 아니라", "다시 말하면", "잘못 말했", "제 말은", "아 아니", "정정할게"]
NO_REPEAT = {"네", "예", "응", "아니", "정말", "진짜", "너무", "네네", "감사합니다", "그래", "빨리", "자", "하하",
             "아니요", "맞아", "맞아요"}
# short words that are often followed by a longer word starting with the same syllable (다 다른, 안 안다)
_COMMON_SHORT = set("다안못잘더또꼭좀이그저한두세네몇새각첫제내너왜뭐수것거때곳걸줄뿐등")

POSITIVE = """좋 감사 고맙 고마워 행복 기쁘 기뻐 만족 최고 훌륭 멋지 멋져 멋있 편하 편해 편리 괜찮 다행 사랑 즐거 즐겁
재미있 재밌 완벽 친절 맛있 예쁘 예뻐 신나 깔끔 잘됐 반가""".split() + ["도움이 됐", "도움이 되", "잘 됐"]
NEGATIVE = """싫 나쁘 나빠 화나 화났 짜증 슬프 슬퍼 실망 걱정 불안 무서 무섭 피곤 힘들 힘드 문제 아프 아파 최악 불편
답답 속상 우울 고장 어렵 어려워 불만 귀찮 비싸 망했 엉망 틀렸 별로 손해 지쳐 지치""".split() + ["화가 나"]
LEXICON = {**{w: 1 for w in POSITIVE}, **{w: -1 for w in NEGATIVE}}
_NEG_AFTER = re.compile(r"^[가-힣]{0,2}\s?(지\s?(않|못|마)|없)")
_INT_BEFORE = re.compile(r"(너무|정말|진짜|아주|완전|엄청|되게|매우|굉장히|많이|제일|넘)\s?$")
WORRY = ["걱정", "불안", "무서", "무섭", "긴장", "초조"]
TIRED = ["피곤", "지쳐", "지치", "졸려", "졸리", "기운이 없"]


def _negated(clause: str, s: int, e: int) -> bool:
    before = clause[:s].rstrip()
    if before.endswith(("안", "못")) and (len(before) < 2 or not _HANGUL.match(before[-2])):
        return True                       # 안 좋아요, 못 해요
    if clause[max(0, s - 1):s] in ("안", "못") and (s < 2 or not _HANGUL.match(clause[s - 2])):
        return True                       # 안좋아요
    return bool(_NEG_AFTER.match(clause[e:e + 8]))   # 좋지 않아요, 문제없어요


class Korean(LanguagePack):
    def analyze(self, text: str) -> Counts:
        c = Counts()
        toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        words = [t for t, _, _ in toks]
        fillers: Counter = Counter()
        filler_idx = set()
        for i, (t, s, e) in enumerate(toks):
            after = text[e:e + 3]
            pause_next = bool(re.match(r"\s*([,…~]|\.\.\.)", after))
            repeated = i + 1 < len(toks) and toks[i + 1][0] == t
            if t in FILLERS_ALWAYS or (t in FILLERS_BEFORE_PAUSE and (pause_next or repeated)):
                fillers[t] += 1
                filler_idx.add(i)
        c.fillers = fillers
        content = [t for i, t in enumerate(words) if i not in filler_idx]
        c.words, c.content_words = float(len(words)), float(len(content))
        c.syllables = sum(len(_HANGUL.findall(t)) or len(t) for t in content)

        pw = len(re.findall(r"([가-힣])-\1", text))
        for (t, s, e), (u, s2, _) in zip(toks, toks[1:]):
            sep = text[e:s2]
            if len(t) <= 2 and u != t and u.startswith(t) and len(u) > len(t) and \
                    (re.search(r"[,…-]", sep) or (len(t) == 1 and t not in _COMMON_SHORT)):
                pw += 1
        c.part_word_repetitions = pw
        c.word_repetitions = sum(1 for a, b in zip(content, content[1:]) if a == b and a not in NO_REPEAT)
        c.phrase_repetitions = sum(1 for i in range(len(content) - 3)
                                   if content[i:i + 2] == content[i + 2:i + 4] and content[i] != content[i + 1])
        c.prolongations = len(re.findall(r"([가-힣])\1{3,}", text)) + \
            sum(1 for m in re.finditer(r"([가-힣]+)~{1,}", text) if m.group(1) not in FILLERS_ALWAYS)
        c.revisions = sum(count_phrases(text, REVISIONS, word_boundaries=False).values())
        c.incomplete_words = len(re.findall(r"[가-힣]-(?=\s|$|[,.])", text))

        hedges: Counter = Counter()
        for h in sorted(HEDGES, key=len, reverse=True):
            pat = re.escape(h) if h.startswith(("것", "거")) else r"(?<![가-힣])" + re.escape(h)
            n = len(re.findall(pat, text))
            if n and not any(h in longer for longer in hedges):
                hedges[h] = n
        c.hedges = hedges
        c.vague_words = sum(count_phrases(text, VAGUE, word_boundaries=False).values())

        q = 0
        for sent in re.split(r"(?<=[.!?。！？])", text):
            s = sent.strip()
            if s.endswith(("?", "？")) or re.sub(r"[.!。！\s]+$", "", s).endswith(("까", "까요", "나요", "냐")):
                q += 1
        c.questions = q

        c.positive, c.negative = score_sentiment(
            text, LEXICON, negated=_negated,
            intensified=lambda cl, s: bool(_INT_BEFORE.search(cl[max(0, s - 4):s])))
        c.emotion_words = ({"worried"} if any(w in text for w in WORRY) else set()) | \
                          ({"tired"} if any(w in text for w in TIRED) else set())
        return c


PACK = Korean(
    code="ko", name="Korean", word_unit="eojeol", syllable_unit="syllable",
    pace_unit="syllables", speaking_rate_norm=4.1, articulation_norm_sps=4.65,
    continuation={"는데", "은데", "인데", "던데", "고", "서", "면", "지만", "그리고", "그래서", "근데", "그런데", "음",
                  "어", "그", "저", "뭐", "니까"},
    complete={"요", "다", "까", "죠", "니다", "세요", "네요", "어요", "아요", "습니까"},
    question={"까", "?", "？"},
    backchannel={"네", "응", "예", "아", "그렇죠", "맞아요", "음", "어", "그래"},
    sources=["Seoul Korean spontaneous speech: speaking rate ~4.0-4.2 and articulation rate ~4.5-4.8 syllables/s "
             "(Kim 2018, Phonetics and Speech Sciences; koreascience.kr JAKO201810760742376)"],
)
