# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Japanese.

Rate is counted in morae (each kana is one mora; small ゃゅょ join the previous one; っ, ん and ー count).
Kanji readings are not known without a dictionary, so each kanji counts as 2 morae, a typical average.
Word counts are estimated from characters (about 1.7 characters per word).
"""
from __future__ import annotations

import re
from collections import Counter

from .base import Counts, LanguagePack, count_phrases, score_sentiment

KANA = "ぁ-ゖゝ-ゟァ-ヺー-ヿｦ-ﾟ"
KANJI = "㐀-䶿一-鿿豈-﫿々"
_JP = re.compile(f"[{KANA}{KANJI}]")
_KANA = re.compile(f"[{KANA}]")
_KANJI = re.compile(f"[{KANJI}]")
_SMALL = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_SEP = r"[、,。.!?！？…\s]"
CHARS_PER_WORD = 1.7
MORAE_PER_KANJI = 2.0

_FILLER = re.compile(r"えーっと|えっと|えーと|ええと|えー+|あのー+|あのう|そのー+|うーん|うーむ|んー+|あー+|"
                     r"まあ(?!まあ)(?<!まあまあ)|"
                     rf"(?:あの|その|なんか)(?=[、,…\s])")
HEDGES = ["たぶん", "多分", "かもしれない", "かもしれません", "かも", "と思います", "と思う", "と思って", "みたい",
          "ちょっと", "一応", "らしい", "気がします", "気がする", "わからない", "分からない", "わかりません",
          "分かりません", "おそらく", "恐らく", "なんとなく", "っぽい"]
VAGUE = ["何か", "なにか", "いろいろ", "色々", "とか", "そういうの", "ああいうの", "なんとか"]
REVISIONS = ["じゃなくて", "ではなくて", "間違えました", "間違えた", "言い直します", "というか", "っていうか",
             "いやいや"]
NO_REPEAT = {"はい", "うん", "ええ", "そう", "ね", "いや", "どうも", "ほら", "まあ"}

POSITIVE = """良い 良かっ 良く いい よかっ 嬉し うれし 楽し たのし 好き 大好き ありがと 有難 有り難 素晴らし すばらし
最高 助かっ 助かり 助かる 安心 満足 便利 簡単 大丈夫 素敵 すてき すごい 凄い 感謝 美味し おいし 綺麗 きれい 完璧 幸せ
快適 親切 丁寧 気に入っ 楽しみ""".split()
NEGATIVE = """悪い 悪かっ 悪く 嫌い 嫌だ いやだ 怒っ 怒り 腹が立 悲し かなし 残念 困っ 困る 困り 心配 不安 怖 こわい
疲れ つかれ 最悪 ひどい 酷い 痛い 辛い つらい 不満 失望 がっかり 面倒 めんどくさ イライラ いらいら 遅い 遅すぎ 壊れ 故障
間違 難し むずかし 問題 不便 迷惑 うるさ 苦し 寂し さびし 落ち込 ムカつ むかつ""".split()
NEUTRAL = ["いいえ", "良くも悪くも"]
LEXICON = {**{w: 1 for w in POSITIVE}, **{w: -1 for w in NEGATIVE}, **{w: 0 for w in NEUTRAL}}
# Japanese negation follows the word: 嬉しくない, 好きじゃない, 良くありません, 心配ない
_NEG_AFTER = re.compile(r"^[ぁ-ん]{0,2}?(ない|なかった|なく|ありません|ません|ず)")
_INT_BEFORE = re.compile(r"(とても|すごく|凄く|本当に|ほんとに|ほんとうに|めっちゃ|非常に|かなり|超|大変|全く|"
                         r"まじで|マジで|一番|すっごく)$")
WORRY = ["心配", "不安", "怖", "こわい", "緊張", "焦っ", "焦る"]
TIRED = ["疲れ", "つかれ", "眠い", "ねむい", "しんどい", "だるい"]


def count_morae(text: str) -> int:
    kana = sum(1 for ch in _KANA.findall(text) if ch not in _SMALL)
    return int(round(kana + MORAE_PER_KANJI * len(_KANJI.findall(text)) + 2 * len(re.findall(r"\d", text))))


class Japanese(LanguagePack):
    def analyze(self, text: str) -> Counts:
        c = Counts()
        fillers: Counter = Counter()
        filler_spans = []
        for m in _FILLER.finditer(text):
            key = re.sub(r"ー+$", "ー", m.group(0))
            fillers[key] += 1
            filler_spans.append(m.group(0))
        c.fillers = fillers
        n_fill = sum(fillers.values())
        content = text
        for f in filler_spans:
            content = content.replace(f, "", 1)

        chars = len(_JP.findall(content))
        c.syllables = count_morae(content)
        c.content_words = chars / CHARS_PER_WORD
        c.words = c.content_words + n_fill

        pw = len(re.findall(r"([ぁ-んァ-ン])(?:[-‐]\1)+", text))
        reps = 0
        for m in re.finditer(rf"([{KANA}{KANJI}]{{1,2}})[、,…\s]+\1", text):
            frag = m.group(1)
            if frag in NO_REPEAT:
                continue
            if len(frag) == 1 and _KANA.match(frag):
                pw += 1             # わ、わたし: a syllable repeated before the word
            else:
                reps += 1
        for m in re.finditer(r"([ぁ-んァ-ン])\1\1+", text):
            if m.group(1) not in "ーんっふはへほあうえ":
                reps += 1
        c.part_word_repetitions, c.word_repetitions = pw, reps
        c.phrase_repetitions = len(re.findall(rf"([{KANA}{KANJI}]{{3,5}})[、,…\s]+\1", text))
        c.prolongations = sum(1 for m in re.finditer(r"(.)ー{2,}|(.)[～~]+", text)
                              if (m.group(1) or m.group(2) or "") not in "えあうんー")
        c.revisions = sum(count_phrases(text, REVISIONS, word_boundaries=False).values())
        c.incomplete_words = len(re.findall(rf"[{KANA}{KANJI}][-‐—]+(?={_SEP}|$)", text))
        c.hedges = count_phrases(text, HEDGES, word_boundaries=False)
        c.vague_words = sum(count_phrases(text, VAGUE, word_boundaries=False).values())

        q = 0
        for sent in re.split(r"(?<=[。！？.!?])", text):
            s = sent.strip()
            if s.endswith(("？", "?")) or re.sub(r"[。！.!\s]+$", "", s).endswith(("か", "かな", "かね")):
                q += 1
        c.questions = q

        c.positive, c.negative = score_sentiment(
            text, LEXICON,
            negated=lambda cl, s, e: bool(_NEG_AFTER.match(cl[e:e + 7])),
            intensified=lambda cl, s: bool(_INT_BEFORE.search(cl[max(0, s - 5):s])))
        c.emotion_words = ({"worried"} if any(w in text for w in WORRY) else set()) | \
                          ({"tired"} if any(w in text for w in TIRED) else set())
        return c


PACK = Japanese(
    code="ja", name="Japanese", word_unit="estimated", syllable_unit="mora",
    pace_unit="morae", speaking_rate_norm=5.6, articulation_norm_sps=7.0,
    continuation={"けど", "けれど", "けれども", "が", "で", "て", "って", "から", "し", "とか", "あの", "あのー",
                  "えーと", "えっと", "その", "まあ", "なんか", "ので", "のに", "たり"},
    complete={"です", "ます", "でした", "ました", "ね", "よ", "か", "ください", "だ", "ない", "ません", "ですか"},
    question={"か", "？", "?"},
    backchannel={"はい", "うん", "ええ", "そうですね", "なるほど", "そう", "うんうん", "へえ"},
    sources=["Speaking rate ~5.6 morae/s with pauses, ~7 morae/s articulation: approximate values for spontaneous "
             "Japanese, consistent with Pellegrino, Coupé & Marsico (2011), Language 87(3)"],
)
