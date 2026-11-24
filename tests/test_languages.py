# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Language packs (en/zh/ja/ko), code-switching and language-fair pace."""
from paralingo import LanguagePack, SpeechProfileEngine, analyze_text, register_pack, validate
from paralingo.lang import PACKS, Counts, detect_language, normalize_language, segment
from paralingo.lang.ja import count_morae
from paralingo.language import speech_rates
from paralingo.realtime import _norm_lang, is_backchannel, lexical_cues


def a(text, lang=None):
    return analyze_text(text, lang)


# ----------------------------------------------------------------------- language detection
def test_normalize_and_detect():
    assert normalize_language("en-US") == "en"
    assert normalize_language("zh-Hant-TW") == "zh"
    assert normalize_language("cmn") == "zh"
    assert normalize_language("ko_KR") == "ko"
    assert _norm_lang("ja-JP") == "ja" and _norm_lang("fr-FR") is None
    assert detect_language("我想退款") == "zh"
    assert detect_language("今日は会議があります") == "ja"
    assert detect_language("会議") == "zh" and detect_language("会議", "ja-JP") == "ja"
    assert detect_language("안녕하세요") == "ko"
    assert detect_language("Hello there") == "en"


def test_segment_code_switching():
    segs = segment("我觉得这个 deadline 太紧了，meeting 能不能改到明天？")
    assert [lang for lang, _ in segs] == ["zh", "en", "zh", "en", "zh"]
    assert "".join(s for _, s in segs) == "我觉得这个 deadline 太紧了，meeting 能不能改到明天？"
    # a lone letter in Asian text is not a language switch
    assert [lang for lang, _ in segment("我要A套餐")] == ["zh"]


# ----------------------------------------------------------------------- English is unchanged
def test_english_analysis():
    r = a("Um, hi, I was charged twice. I mean, I think the the bill is wrong, you know?")
    f, x = r["fluency"], r["lexical"]
    assert f["fillers"] == {"um": 1, "i mean": 1, "you know": 1}
    assert f["word_repetitions"] == 1 and x["hedges"] == {"i think": 1}
    assert x["word_count"] == 17 and x["word_count_method"] == "whitespace"
    assert x["type_token_ratio"] is not None
    assert r["language"]["primary"] == "en" and r["language"]["code_switching"] is None


# ----------------------------------------------------------------------- Mandarin
def test_mandarin_fluency_and_hedges():
    r = a("嗯，那个，我觉得这个服务不太好，我我我想退款。可能是系统的问题吧？")
    f, x = r["fluency"], r["lexical"]
    assert f["fillers"] == {"嗯": 1, "那个": 1}
    assert f["word_repetitions"] == 1
    assert x["hedges"] == {"我觉得": 1, "可能": 1}
    assert x["question_count"] == 1
    assert x["word_count_method"] == "estimated" and x["type_token_ratio"] is None
    assert r["sentiment"]["score"] < 0                         # 不太好
    assert r["language"]["primary"] == "zh"


def test_mandarin_avoids_false_positives():
    r = a("这个金额差不多，麻烦您看看，谢谢谢谢。")
    assert r["fluency"]["filler_count"] == 0                   # 额 inside 金额 is not a filler
    assert r["fluency"]["word_repetitions"] == 0               # 看看 / 谢谢 are words, not stutters
    assert r["sentiment"]["negative_weight"] == 0              # 差不多 / 麻烦您 are neutral


def test_mandarin_sentiment_negation_and_intensifiers():
    assert a("非常满意，太好了！")["sentiment"]["score"] > 0.5
    assert a("我不满意")["sentiment"]["score"] < 0
    assert a("不是很好")["sentiment"]["score"] < 0
    assert a("没问题")["sentiment"]["score"] > 0


# ----------------------------------------------------------------------- Japanese
def test_japanese_morae():
    assert count_morae("きょう") == 2          # きょ + う
    assert count_morae("がっこう") == 4        # が っ こ う
    assert count_morae("コーヒー") == 4
    assert count_morae("会議") == 4            # 2 kanji, ~2 morae each


def test_japanese_fluency_and_sentiment():
    r = a("えーと、あの、ちょっと分からないんですけど、多分大丈夫だと思います。")
    assert r["fluency"]["fillers"] == {"えーと": 1, "あの": 1}
    assert {"ちょっと", "多分", "と思います", "分からない"} <= set(r["lexical"]["hedges"])
    assert r["lexical"]["syllable_unit"] == "mora"
    assert a("全然嬉しくないです。本当に困っています。")["sentiment"]["score"] < -0.3
    assert a("心配ないです")["sentiment"]["score"] > 0          # "no need to worry"
    assert a("わ、わたしは楽しかったです")["fluency"]["part_word_repetitions"] == 1
    assert a("明日は大丈夫ですか")["lexical"]["question_count"] == 1


# ----------------------------------------------------------------------- Korean
def test_korean_fluency_and_sentiment():
    r = a("음, 저, 그게 좀 문제가 있는 것 같아요. 저, 저는 너무 피곤해요.")
    f, x = r["fluency"], r["lexical"]
    assert f["fillers"] == {"음": 1, "저": 2}
    assert f["part_word_repetitions"] == 1                     # 저, 저는
    assert {"것 같아요", "좀"} <= set(x["hedges"])
    assert x["word_count_method"] == "eojeol"
    assert r["sentiment"]["score"] < 0
    assert a("그 사람이 왔어요")["fluency"]["filler_count"] == 0   # 그 = "that", not a filler
    assert a("안 좋아요")["sentiment"]["score"] < 0
    assert a("좋지 않아요")["sentiment"]["score"] < 0
    assert a("문제없어요")["sentiment"]["score"] > 0
    assert a("내일 가능할까요")["lexical"]["question_count"] == 1


# ----------------------------------------------------------------------- code-switching
def test_code_switching_block():
    r = a("这个 project 的 budget 我们 next week 再 review 吧")
    lang = r["language"]
    assert lang["primary"] == "zh"                             # Chinese frame with English terms
    cs = lang["code_switching"]
    assert cs["embedded_languages"] == ["en"]
    assert cs["embedded_terms"] == ["project", "budget", "next week", "review"]
    assert a("I love 寿司 so much")["language"]["primary"] == "en"


def test_unsupported_language_falls_back_to_counts():
    r = a("Bonjour, je voudrais un rendez-vous demain.", "fr-FR")
    assert r["language"]["primary"] == "fr" and r["language"]["supported"] is False
    assert r["lexical"]["word_count"] == 7 and r["fluency"]["filler_count"] == 0


def test_register_custom_pack():
    class Pirate(LanguagePack):
        def analyze(self, text):
            toks = text.lower().split()
            return Counts(words=len(toks), content_words=len(toks), syllables=len(toks), tokens=toks,
                          fillers={"arr": toks.count("arr")})
    register_pack(Pirate(code="xp", name="Pirate", continuation={"arr"}))
    try:
        r = a("arr matey arr", "xp")
        assert r["language"]["primary"] == "xp" and r["fluency"]["fillers"] == {"arr": 2}
        assert lexical_cues("ahoy matey arr", "xp")["continuation"] is True
    finally:
        PACKS.pop("xp")


# ----------------------------------------------------------------------- real-time cues
def test_realtime_cues_come_from_packs():
    assert lexical_cues("我觉得然后", "zh-CN")["continuation"] is True
    assert lexical_cues("そうですね。", "ja")["complete"] is True
    assert is_backchannel("네", "ko-KR")


# ----------------------------------------------------------------------- engine: fair pace, language use
def _turn(eng, sid, text, span, lang=None):
    return eng.analyze_utterance("s", session_id=sid, transcript=text, t_start=0.0, t_end=span, language=lang)


def test_pace_is_judged_against_each_language():
    ja = a("今日はとても良い天気ですね。散歩に行きましょう。")
    morae = ja["lexical"]["syllable_count"]
    r = speech_rates(ja, morae / 5.6, morae / 7.0)             # a typical Japanese speaker
    assert abs(r["pace_ratio"] - 1.0) < 0.01 and abs(r["articulation_ratio"] - 1.0) < 0.01
    en = a("I would like to change my delivery address to the new one please.")
    words = en["lexical"]["content_word_count"]
    r = speech_rates(en, words / 2.5, None)                    # 150 words per minute
    assert abs(r["pace_ratio"] - 1.0) < 0.01 and abs(r["words_per_minute"] - 150) < 1


def test_profile_language_use_and_guidance(tmp_path):
    eng = SpeechProfileEngine(tmp_path, agent_id="t")
    sid = eng.start_session("s")
    turns = ["我觉得这个 deadline 太紧了，能不能改到明天？",
             "嗯，那个 meeting 我们可以下午开吗？",
             "好的，谢谢你，我明白了。",
             "这个 budget 我们 next week 再 review 吧。"]
    for t in turns:
        res = _turn(eng, sid, t, 3.0)
    assert res["language"]["primary"] == "zh"
    p = eng.get_profile("s")
    assert "zh" in p["speaker"]["languages"]
    lu = p["assessments"]["language_use"]
    assert lu["primary"] == "zh" and lu["mixed_turn_ratio"] == 0.75
    assert {"deadline", "meeting"} <= set(lu["frequent_embedded_terms"])
    g = p["agent_guidance"]
    assert g["language"] == "zh" and g["recommended_speaking_rate_wpm"] is None
    assert isinstance(g["recommended_tts_rate_pct"], int)
    instr = " ".join(x["instruction"] for x in g["adaptations"])
    assert "Reply in Mandarin" in instr and "mixes Mandarin Chinese with English" in instr
    summary = eng.end_session("s", sid)
    assert summary["languages"]["primary"] == "zh"
    ctx = eng.get_summary("s")["prompt_context"]
    assert "Languages: zh" in ctx and "mixes languages" in ctx
    validate(eng.get_profile("s"))


def test_borrowed_terms_not_kept_without_transcript_consent(tmp_path):
    eng = SpeechProfileEngine(tmp_path, agent_id="t")
    eng.set_consent("s", store_transcripts=False)
    sid = eng.start_session("s")
    r = _turn(eng, sid, "我觉得这个 deadline 太紧了", 2.0)
    assert r["language"]["code_switching"]["embedded_terms"] == []
    p = eng.get_profile("s")
    assert p["assessments"]["language_use"]["frequent_embedded_terms"] == []
    assert "deadline" not in str(p)


def test_han_only_text_uses_profile_language(tmp_path):
    eng = SpeechProfileEngine(tmp_path, agent_id="t")
    eng.update_speaker("jp", languages=["ja-JP"])
    sid = eng.start_session("jp")
    r = eng.analyze_utterance("jp", session_id=sid, transcript="会議室予約", t_start=0, t_end=1.5)
    assert r["language"]["primary"] == "ja"
    r = eng.analyze_utterance("jp", session_id=sid, transcript="会議室予約", t_start=0, t_end=1.5, language="zh-CN")
    assert r["language"]["primary"] == "zh"                    # the ASR hint wins over the profile
