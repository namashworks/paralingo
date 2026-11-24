# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Mandarin Chinese (simplified and traditional characters).

Written without spaces, so word counts are estimated from characters (about 1.6 characters per word).
Each character is one syllable.
"""
from __future__ import annotations

import re
from collections import Counter

from .base import Counts, LanguagePack, count_phrases, score_sentiment

HAN = "㐀-䶿一-鿿豈-﫿"
_HAN = re.compile(f"[{HAN}]")
_SEP = r"[，,、。.!?！？…\s]"
CHARS_PER_WORD = 1.6

# hesitation sounds: always fillers / only when standing alone (额 is also part of 金额 "amount")
_HES_ALWAYS = re.compile(r"嗯+|呃+")
_HES_ALONE = re.compile(rf"(?:^|(?<={_SEP}))(额+|唔+|啊+|哎+)(?={_SEP}|$)")
# discourse fillers ("那个，", "就是，"): only when followed by a pause
_DISCOURSE = re.compile(r"(那个|这个|那個|這個|就是说|就是說|就是)(?=[，,、…])")

HEDGES = ["可能", "也许", "也許", "大概", "好像", "我觉得", "我覺得", "我感觉", "我感覺", "不太确定", "不确定",
          "不確定", "不知道", "差不多", "有点", "有點", "有一点", "或许", "或許", "估计", "估計", "应该是", "應該是",
          "似乎", "说不定", "說不定", "我猜"]
VAGUE = ["东西", "東西", "什么的", "什麼的", "之类的", "之類的", "那种", "那種", "某个", "某個"]
_REVISION = re.compile(r"我是说|我是說|我的意思是|应该说|應該說|说错了|說錯了|不是不是|不对不对|不對不對|"
                       rf"(?:^|(?<={_SEP}))(?:不对|不對)(?=[，,])")
NO_REPEAT = {"对", "對", "好", "是", "嗯", "哈", "行", "谢", "謝", "不", "啊", "哦", "呃", "嘿", "呵", "嘻", "对对",
             "好的", "是的", "谢谢", "謝謝", "没有", "沒有"}

POSITIVE = """很好 挺好 真好 太好了 好的 好极了 好極了 不错 不錯 棒 很棒 喜欢 喜歡 开心 開心 高兴 高興 满意 滿意 谢谢
谢谢你 謝謝 感谢 感謝 多谢 多謝 完美 优秀 優秀 舒服 放心 安心 方便 清楚 没问题 沒問題 厉害 厲害 爱 愛 漂亮 顺利 順利
有帮助 有幫助 好用 好吃 好看 好听 好聽 可以 快乐 快樂 幸福 感动 感動 太棒了""".split()
NEGATIVE = """不好 不太好 不怎么好 不怎麼好 不太满意 不太滿意 不太行 差 很差 太差 糟糕 生气 生氣 气死 氣死 难过 難過 伤心 傷心 失望 烦 煩 烦人 讨厌 討厭 担心 擔心 焦虑
焦慮 害怕 紧张 緊張 累 疲惫 疲憊 错 錯 坏 壞 不满意 不滿意 不满 不滿 麻烦 麻煩 痛 疼 难受 難受 郁闷 鬱悶 不行 投诉
投訴 太慢 很慢 太贵 太貴 垃圾 无语 無語 崩溃 崩潰 着急 著急 头疼 頭疼 倒霉 可惜 遗憾 遺憾""".split()
# words that contain a sentiment character but are neutral (matched first, then ignored)
NEUTRAL = """差不多 差点 差點 出差 好像 好多 只好 正好 刚好 剛好 问题 問題 麻烦你 麻煩你 麻烦您 麻煩您 积累 積累 累计
累計 爱好 愛好 不好意思 不客气 不客氣 没关系 沒關係 痛快""".split()
LEXICON = {**{w: 1 for w in POSITIVE}, **{w: -1 for w in NEGATIVE}, **{w: 0 for w in NEUTRAL}}
_NEG_BEFORE = re.compile(r"(不|没有?|沒有?|别|別|未|不太|不怎么|不怎麼|并不|並不|不是很?|不够|不夠)$")
_INT_BEFORE = re.compile(r"(非常|特别|特別|太|真|超级|超級|超|十分|极其|極其|很|挺|相当|相當)$")
WORRY = ["担心", "擔心", "焦虑", "焦慮", "害怕", "紧张", "緊張", "着急", "著急", "不安", "慌"]
TIRED = ["好累", "很累", "太累", "累了", "累死", "疲惫", "疲憊", "疲劳", "疲勞", "好困", "很困", "困了", "没精神",
         "沒精神", "没劲", "沒勁"]


class Mandarin(LanguagePack):
    def analyze(self, text: str) -> Counts:
        c = Counts()
        fillers: Counter = Counter()
        filler_chars = 0
        for m in _HES_ALWAYS.finditer(text):
            fillers[m.group(0)[0]] += 1
            filler_chars += len(m.group(0))
        for m in _HES_ALONE.finditer(text):
            fillers[m.group(1)[0]] += 1
            filler_chars += len(m.group(1))
        for m in _DISCOURSE.finditer(text):
            fillers[m.group(1)] += 1
        c.fillers = fillers
        hesitations = sum(v for k, v in fillers.items() if len(k) == 1)

        han = len(_HAN.findall(text))
        digits = len(re.findall(r"\d", text))
        content_syll = max(0, han - filler_chars) + digits
        c.syllables = content_syll
        c.content_words = content_syll / CHARS_PER_WORD
        c.words = c.content_words + hesitations

        c.part_word_repetitions = len(re.findall(rf"([{HAN}])(?:[-—]\1)+", text))
        reps = 0
        for m in re.finditer(rf"([{HAN}]{{1,2}})[，,、…\s]+\1", text):
            if m.group(1) not in NO_REPEAT:
                reps += 1
        for m in re.finditer(rf"([{HAN}])\1\1+", text):
            if m.group(1) not in NO_REPEAT:
                reps += 1
        c.word_repetitions = reps
        c.phrase_repetitions = len(re.findall(rf"([{HAN}]{{3,4}})[，,、…\s]+\1", text))
        c.prolongations = len(re.findall(rf"[{HAN}][～~]+", text))
        c.revisions = len(_REVISION.findall(text))
        c.incomplete_words = len(re.findall(rf"[{HAN}][-—]+(?={_SEP}|$)", text))
        c.hedges = count_phrases(text, HEDGES, word_boundaries=False)
        c.vague_words = sum(count_phrases(text, VAGUE, word_boundaries=False).values())

        q = 0
        for sent in re.split(r"(?<=[。！？.!?])", text):
            s = sent.strip()
            if s.endswith(("？", "?")) or re.sub(r"[。！.!\s]+$", "", s).endswith(("吗", "嗎", "呢")):
                q += 1
        c.questions = q

        c.positive, c.negative = score_sentiment(
            text, LEXICON,
            negated=lambda cl, s, e: bool(_NEG_BEFORE.search(cl[max(0, s - 4):s])),
            intensified=lambda cl, s: bool(_INT_BEFORE.search(cl[max(0, s - 3):s])))
        clean = text.replace("积累", "").replace("積累", "").replace("累计", "").replace("累計", "")
        c.emotion_words = ({"worried"} if any(w in clean for w in WORRY) else set()) | \
                          ({"tired"} if any(w in clean for w in TIRED) else set())
        return c


PACK = Mandarin(
    code="zh", name="Mandarin Chinese", word_unit="estimated", syllable_unit="syllable",
    pace_unit="syllables", speaking_rate_norm=3.6, articulation_norm_sps=4.5,
    prosody_scale=1.25, tonal=True,
    continuation={"然后", "就是", "那个", "这个", "因为", "所以", "但是", "可是", "还有", "而且", "如果", "比如", "嗯",
                  "呃", "额", "就", "和", "跟", "的话"},
    complete={"吗", "呢", "吧", "了", "的", "啦", "嘛"},
    question={"吗", "呢", "？", "?"},
    backchannel={"嗯", "嗯嗯", "对", "对对", "对对对", "好", "好的", "是", "是的", "哦", "啊", "明白"},
    sources=["Speaking rate ~3.6 syllables/s with pauses, ~4.5 syllables/s articulation: approximate conversational "
             "values scaled from Pellegrino, Coupé & Marsico (2011), Language 87(3)",
             "Wider pitch range than English because of lexical tones (e.g. Keating & Kuo 2012, JASA): scale 1.25, "
             "approximate"],
)
