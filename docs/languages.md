# Languages and code-switching

Paralingo analyses transcripts in **English, Mandarin Chinese, Japanese and Korean**, including turns where
the speaker **switches language mid-sentence** ("我觉得这个 deadline 太紧了"). It judges pace and pitch
against each language's own norms, so a fast Japanese speaker and a fast English speaker both read as
"fast". Other languages get the language-neutral analysis (pitch, pauses, voice quality, counts).

Audio analysis (pitch, loudness, pauses, voice quality, real-time turn-taking timing) works for every
language. What follows is about the **words**.

## What each language supports

| | English | Mandarin | Japanese | Korean | Other |
|---|---|---|---|---|---|
| Words | spaces | estimated (~1.6 characters/word) | estimated (~1.7 characters/word) | eojeol (spaces) | spaces |
| Rate unit | syllables | syllables (1 per character) | **morae** | syllables (1 per Hangul block) | vowel groups |
| Fillers | um, uh, "you know" | 嗯, 呃, 那个 (before a pause) | えーと, えっと, あのー | 음, 어, 저 (before a pause) | – |
| Hedges | maybe, I think | 可能, 我觉得, 好像 | たぶん, かもしれない, と思います | 아마, 것 같아요, 좀 | – |
| Repetitions and stutters | ✓ | ✓ (我我我, 我，我想) | ✓ (わ、わたし) | ✓ (저, 저는) | word repeats |
| Sentiment with negation | not good | 不好, 不满意, 没问题 | 嬉しくない, 心配ない | 안 좋아요, 문제없어요 | – |
| Questions | ? | ？, 吗, 呢 | ？, か | ?, 까요, 나요 | ? |
| Real-time turn-taking cues | and…, um… | 然后, 就是… | けど, て… | -는데, -고… | timing and pitch |
| Vocabulary richness | ✓ | – (needs word segmentation) | – | – | ✓ |

Every language pack is one readable Python file in [`paralingo/lang/`](../paralingo/lang/).

## Example

```python
from paralingo import SpeechProfileEngine, analyze_text

r = analyze_text("嗯，我觉得这个 deadline 太紧了，能不能改到 next week？")
print(r["language"]["primary"], r["language"]["code_switching"]["embedded_terms"])
# zh ['deadline', 'next week']
print(r["fluency"]["fillers"], r["lexical"]["hedges"])
# {'嗯': 1} {'我觉得': 1}

eng = SpeechProfileEngine("profiles", agent_id="support-bot")
sid = eng.start_session("caller-7", channel="phone")
for text, secs in [("嗯，我觉得这个 deadline 太紧了，能不能改到 next week？", 4.0),
                   ("那个 meeting 我们可以下午开吗？", 2.6),
                   ("好的，谢谢你，我明白了。", 2.0)]:
    eng.analyze_utterance("caller-7", session_id=sid, transcript=text, t_start=0, t_end=secs,
                          language="zh-CN")            # optional: what your ASR detected
eng.end_session("caller-7", sid)
print(eng.get_summary("caller-7")["prompt_context"])
```

The summary includes lines like:

```text
- Languages: zh 87%, en 13%; mixes languages in 67% of turns (e.g. deadline, next week, meeting)
- Speaking rate: +13% vs your default voice
- [medium] Reply in Mandarin Chinese; it is the language this speaker mostly uses.
- [medium] The speaker mixes Mandarin Chinese with English (e.g. deadline, next week, meeting). That is
  natural for them: reuse their terms rather than translating them.
```

## How the language is decided

1. **The script of the text**: kana means Japanese, Hangul means Korean, Latin letters mean English.
2. **Chinese characters with no kana** could be Mandarin or Japanese (会議室). Paralingo uses the
   `language` you pass (your ASR's detected language), then the speaker's profile languages, and
   otherwise assumes Mandarin.
3. Where your ASR hint and the script disagree, the script wins: if Azure says `en-US` but the text is
   Hangul, it is analysed as Korean.

Pass the hint with `analyze_utterance(..., language="ja-JP")`, in `external={"language_id": "ja-JP"}`, or
as the `language` form field of the [HTTP API](http-api.md). With real-time turn-taking, `analyze_turn`
passes on the language of each turn automatically.

Each utterance stores a `language` block:

```json
{"primary": "zh", "supported": true, "asr_hint": "zh-CN",
 "shares": {"zh": 0.753, "en": 0.247},
 "code_switching": {"switches": 3, "embedded_languages": ["en"], "embedded_terms": ["deadline", "next week"]},
 "typical_duration_s": {"speaking": 4.811, "articulating": 3.722}}
```

## Code-switching

A turn is split into runs of one language. The **main language** is the one the speaker keeps coming back
to (the most runs), so "这个 project 的 budget 我们 next week 再 review 吧" is Mandarin with English
terms, and "I love 寿司 so much" is English with a Japanese word. Each run is analysed with its own
language pack, and the results are added up.

Across turns, the profile keeps `assessments.language_use`: the language shares, how often the speaker
switches (`mixed_turn_ratio`) and the terms they borrow most. The agent guidance then says which language
to reply in and to reuse the speaker's own terms. Mixing languages is normal for bilingual speakers and is
never treated as a problem.

## Fair pace across languages

Languages are spoken at very different syllable rates: typical Japanese has many more morae per second
than English has syllables. Comparing everyone to English would call every Japanese speaker "fast".

So each pack has typical rates, and Paralingo stores two language-neutral ratios (1.0 = typical):

| Metric | Meaning |
|---|---|
| `timing.pace_ratio` | Speaking rate (pauses included) ÷ typical for the language. For English, words per minute ÷ 150. |
| `timing.articulation_ratio` | Syllables (morae) per second of actual speech ÷ typical for the language |

Speaking style, the pace and clarity skill scores, the slowed-speech indicator and the agent's recommended
TTS rate all use these ratios. `agent_guidance.recommended_tts_rate_pct` (e.g. `-10`) works with any TTS,
for example SSML `<prosody rate="-10%">`. `recommended_speaking_rate_wpm` is only given for English.

| Language | Typical speaking rate | Typical articulation rate | Source |
|---|---|---|---|
| English | 150 words/min | 4.8 syllables/s | conversational English |
| Mandarin | 3.6 syllables/s | 4.5 syllables/s | scaled from Pellegrino, Coupé & Marsico (2011) |
| Japanese | 5.6 morae/s | 7.0 morae/s | consistent with Pellegrino et al. (2011) |
| Korean | 4.1 syllables/s | 4.65 syllables/s | Seoul spontaneous speech, Kim (2018) |

These are **priors**: once a speaker has a personal baseline (5+ turns), their own rates are used.

## Tone languages

In Mandarin, every syllable has a lexical tone, so pitch moves more than in English for reasons that have
nothing to do with emotion. Paralingo therefore:

* expects about 25 % more pitch variability before calling a Mandarin speaker "expressive" or
  "flat", and before using pitch as an arousal cue without a personal baseline;
* doesn't treat a final pitch fall as a sign of confidence in Mandarin, because the last syllable's
  tone decides it;
* gives final pitch half its usual weight when deciding whether a Mandarin speaker has finished their turn.

## ASR settings matter

Fillers and repetitions can only be counted if your ASR keeps them. Many engines remove them by default:

* **Azure AI Speech:** check with your languages, as some fillers may be dropped. The display text keeps
  punctuation, which Paralingo uses for pauses and questions in Chinese, Japanese and Korean.
* **Deepgram:** `filler_words=true`. **AssemblyAI:** `disfluencies=true`.
* **Whisper:** prompt with a verbatim example ("嗯，那个，我觉得…").

## Add your language

A language pack is a small Python class with word lists and counting rules. Copy
[`paralingo/lang/ko.py`](../paralingo/lang/ko.py) (spaced language) or
[`paralingo/lang/zh.py`](../paralingo/lang/zh.py) (unspaced), fill in your language, and register it:

```python
from paralingo import LanguagePack, register_pack, analyze_text
from paralingo.lang import Counts


class Spanish(LanguagePack):
    FILLERS = {"eh", "este", "pues", "bueno"}

    def analyze(self, text):
        toks = text.lower().replace(",", " ").replace(".", " ").split()
        content = [t for t in toks if t not in self.FILLERS]
        return Counts(words=len(toks), content_words=len(content), tokens=content,
                      syllables=sum(max(1, sum(ch in "aeiouáéíóú" for ch in t)) for t in content),
                      fillers={f: toks.count(f) for f in self.FILLERS if f in toks})


register_pack(Spanish(code="es", name="Spanish", continuation={"y", "pero", "que", "porque", "este"}))
print(analyze_text("Eh, este, quiero cambiar mi cita.", "es-MX")["fluency"]["fillers"])
```

Packs also hold the real-time turn-taking cues (`continuation`, `complete`, `question`, `backchannel`) and
typical rates. Contributions of new language packs are very welcome: see
[CONTRIBUTING](../CONTRIBUTING.md#adding-a-language).

## Limitations

* Word lists are hand-curated, transparent and modest in size. They catch common patterns, not every
  idiom, and won't match a trained model on subtle sentiment.
* Mandarin and Japanese word counts are estimates (no word segmentation), so their per-100-word rates and
  words per minute are approximate, and vocabulary richness isn't computed for them.
* Japanese kanji are counted as 2 morae each without a reading dictionary.
* Typical rates are approximate priors from published studies, replaced by each speaker's own baseline.
* The Mandarin, Japanese and Korean analysis is covered by unit tests on hand-written sentences, not
  yet by a benchmark on recorded conversational speech in those languages.
