# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Real-time turn-taking for voice agents.

Feed live audio (and, if you have them, live ASR transcripts) into a :class:`TurnDetector`. It answers
the questions every voice agent struggles with, in about 10 ms steps:

* **Has the person finished their turn?** ``end_of_turn`` combines how long they have been silent
  (relative to *this person's* usual pauses), how their pitch and loudness moved at the end of speech,
  and the last words: "and…", "um…", Japanese "けど", Korean "-는데", Mandarin "然后" mean "not finished".
* **Did they interrupt the agent?** ``barge_in`` means stop talking; ``backchannel`` ("mm-hmm",
  "yeah", "嗯", "はい", "네") means keep going.
* **How fast is the agent?** :class:`LatencyTracker` measures end-of-turn delay and response time.

    det = TurnDetector(sample_rate=16000)
    for chunk in mic_chunks:                      # any chunk size, float or 16-bit PCM or mu-law
        for ev in det.push_audio(chunk):
            if ev.type == "end_of_turn": reply_to(ev.data["transcript"])
            if ev.type == "barge_in": stop_tts()

Pure numpy; works with any ASR, TTS or voice framework, and on-device.
"""
from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HOP_S = 0.010
FRAME_S = 0.040
LEXICAL_TOLERANCE_S = 0.15   # a transcript this soon before a pause still describes the speech that just ended

# --------------------------------------------------------------------------- language cues
# Words or endings that mean "I'm not finished" when a turn stops on them.
CONTINUATION = {
    "en": {"and", "but", "so", "because", "cause", "or", "then", "that", "which", "who", "if", "when", "while",
           "to", "the", "a", "an", "of", "with", "for", "my", "your", "our", "is", "was", "are", "were", "um",
           "umm", "uh", "uhh", "er", "erm", "like", "well", "also", "about", "from", "in", "on", "at", "it's",
           "i'm", "we", "i", "and then", "you know", "i mean", "kind of", "sort of"},
    "zh": {"然后", "就是", "那个", "这个", "因为", "所以", "但是", "可是", "还有", "而且", "如果", "比如", "嗯", "呃",
           "额", "就", "和", "跟", "的话"},
    "ja": {"けど", "けれど", "けれども", "が", "で", "て", "って", "から", "し", "とか", "あの", "あのー", "えーと",
           "えっと", "その", "まあ", "なんか", "ので", "のに", "たり"},
    "ko": {"는데", "은데", "인데", "던데", "고", "서", "면", "지만", "그리고", "그래서", "근데", "그런데", "음", "어",
           "그", "저", "뭐", "니까"},
}
# Endings that mark a complete sentence or a question (punctuation is handled separately).
COMPLETE = {
    "en": set(),
    "zh": {"吗", "呢", "吧", "了", "的", "啦", "嘛"},
    "ja": {"です", "ます", "でした", "ました", "ね", "よ", "か", "ください", "だ", "ない", "ません", "ですか"},
    "ko": {"요", "다", "까", "죠", "니다", "세요", "네요", "어요", "아요", "습니까"},
}
QUESTION = {"en": {"?"}, "zh": {"吗", "呢", "？", "?"}, "ja": {"か", "？", "?"}, "ko": {"까", "?", "？"}}
BACKCHANNEL = {
    "en": {"yeah", "yes", "yep", "yup", "uh-huh", "uh huh", "mm-hmm", "mhm", "mm", "hmm", "ok", "okay", "right",
           "sure", "i see", "got it", "alright", "all right", "true", "exactly", "cool"},
    "zh": {"嗯", "嗯嗯", "对", "对对", "对对对", "好", "好的", "是", "是的", "哦", "啊", "明白"},
    "ja": {"はい", "うん", "ええ", "そうですね", "なるほど", "そう", "うんうん", "へえ"},
    "ko": {"네", "응", "예", "아", "그렇죠", "맞아요", "음", "어", "그래"},
}
_PUNCT_END = re.compile(r"[.!?。！？]\s*$")
_TRAILING = re.compile(r"[\s,，、…\-—~～]+$")


def detect_language(text: str) -> str | None:
    """Guess en/zh/ja/ko from the script of the text (good enough for cue selection)."""
    if not text:
        return None
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return None


def _norm_lang(lang: str | None) -> str | None:
    if not lang:
        return None
    lang = lang.lower()
    for code in ("en", "zh", "ja", "ko"):
        if lang.startswith(code) or (code == "zh" and lang.startswith(("cmn", "yue"))):
            return code
    return None


def lexical_cues(text: str, language: str | None = None) -> dict[str, bool]:
    """Classify how a (partial) transcript ends: continuation, complete, question, filler."""
    out = {"continuation": False, "complete": False, "question": False}
    if not text or not text.strip():
        return out
    lang = _norm_lang(language) or detect_language(text) or "en"
    raw = text.strip()
    stripped = _TRAILING.sub("", raw)
    ends_punct = bool(_PUNCT_END.search(stripped))
    core = re.sub(r"[.!?。！？\s]+$", "", stripped)
    if lang == "en":
        words = re.findall(r"[a-z']+", core.lower())
        last1 = words[-1] if words else ""
        last2 = " ".join(words[-2:]) if len(words) >= 2 else last1
        out["continuation"] = (last1 in CONTINUATION["en"] or last2 in CONTINUATION["en"]) and not ends_punct
        out["question"] = stripped.endswith("?")
    else:
        out["continuation"] = any(core.endswith(tok) for tok in CONTINUATION[lang]) and not ends_punct
        out["question"] = any(stripped.endswith(q) or core.endswith(q) for q in QUESTION[lang])
        if not out["continuation"] and any(core.endswith(tok) for tok in COMPLETE[lang]):
            out["complete"] = True
    if ends_punct and not out["continuation"]:
        out["complete"] = True
    if raw.endswith(("...", "…", ",", "，", "、")):
        out["continuation"], out["complete"] = True, False
    return out


def is_backchannel(text: str, language: str | None = None) -> bool:
    if not text or not text.strip():
        return False
    lang = _norm_lang(language) or detect_language(text) or "en"
    t = re.sub(r"[.!?。！？,，、…\s]+$", "", text.strip().lower())
    return t in BACKCHANNEL.get(lang, set())


# --------------------------------------------------------------------------- data types
@dataclass
class TurnConfig:
    """Tuning knobs. Defaults suit 8–48 kHz conversational audio."""

    threshold: float = 0.6            # end-of-turn probability needed to end the turn
    min_silence_s: float = 0.20       # never end a turn before this much silence
    default_pause_s: float = 0.70     # assumed long within-turn pause before a personal profile exists
    max_silence_s: float = 1.60       # end the turn after this much silence regardless of cues
    hard_cap_s: float = 2.50          # absolute maximum wait, even for slow speakers
    hesitation_bonus_s: float = 0.60  # extra wait allowed after "um", "and", "けど"...
    vad_margin_db: float = 9.0        # speech must be this far above the noise floor
    vad_floor_dbfs: float = -55.0     # and above this absolute level
    min_speech_s: float = 0.10        # ignore sounds shorter than this (clicks, bumps)
    resume_speech_s: float = 0.05     # speech needed to cancel a pause
    barge_in_min_s: float = 0.30      # user speech this long, with real words, while the agent talks = interruption
    barge_in_no_text_s: float = 0.60  # same, before any ASR text has arrived (text lags audio)
    backchannel_max_s: float = 0.80   # short speech while the agent talks = backchannel
    agent_echo_margin_db: float = 6.0 # extra VAD margin while the agent talks (residual echo)
    preroll_s: float = 0.20           # audio kept before speech start in the turn's audio
    max_turn_s: float = 60.0          # cap on buffered turn audio
    resume_margin_db: float = 4.0     # lower margin to hear speech resuming mid-turn (soft sentence starts)
    fixed_timeout_s: float | None = None  # baseline mode: end turns after exactly this much silence, no cues


@dataclass
class TurnEvent:
    type: str                 # speech_start | end_of_turn | barge_in | backchannel
    t: float                  # stream time in seconds
    data: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- latency
class LatencyTracker:
    """End-of-turn delay (speech stopped -> turn detected) and response time (speech stopped -> agent
    audio started), plus Paralingo's own processing cost."""

    def __init__(self) -> None:
        self.eot_delay_ms: list[float] = []
        self.response_ms: list[float] = []
        self._pending_speech_end: float | None = None
        self.proc_s = 0.0
        self.audio_s = 0.0
        self.hop_times_us: deque[float] = deque(maxlen=20000)

    def turn_detected(self, speech_end_t: float, detected_t: float) -> None:
        self.eot_delay_ms.append((detected_t - speech_end_t) * 1000)
        self._pending_speech_end = speech_end_t

    def agent_started(self, t: float) -> float | None:
        if self._pending_speech_end is None:
            return None
        ms = (t - self._pending_speech_end) * 1000
        self.response_ms.append(ms)
        self._pending_speech_end = None
        return ms

    @staticmethod
    def _stats(v: list[float]) -> dict[str, float | int | None]:
        if not v:
            return {"n": 0, "p50": None, "p95": None, "mean": None}
        a = np.asarray(v)
        return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 1),
                "p95": round(float(np.percentile(a, 95)), 1), "mean": round(float(a.mean()), 1)}

    def summary(self) -> dict[str, Any]:
        hops = list(self.hop_times_us)
        return {
            "end_of_turn_delay_ms": self._stats(self.eot_delay_ms),
            "response_ms": self._stats(self.response_ms),
            "processing": {
                "real_time_factor": round(self.proc_s / self.audio_s, 4) if self.audio_s else None,
                "per_10ms_step_us": self._stats(hops),
                "audio_seconds": round(self.audio_s, 2),
            },
        }


# --------------------------------------------------------------------------- detector
def _to_float(chunk: Any, encoding: str) -> np.ndarray:
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        b = bytes(chunk)
        if encoding == "mulaw":
            u = ~np.frombuffer(b, dtype=np.uint8)
            sign = (u & 0x80) != 0
            exponent = ((u >> 4) & 0x07).astype(np.int32)
            mantissa = (u & 0x0F).astype(np.int32)
            mag = (((mantissa << 3) + 0x84) << exponent) - 0x84
            return np.where(sign, -mag, mag).astype(np.float64) / 32768.0
        return np.frombuffer(b, dtype="<i2").astype(np.float64) / 32768.0  # pcm16
    a = np.asarray(chunk)
    if a.dtype.kind in "iu":
        return a.astype(np.float64) / (32768.0 if a.dtype.itemsize == 2 else float(np.iinfo(a.dtype).max))
    return a.astype(np.float64).ravel()


class TurnDetector:
    """Streaming end-of-turn, barge-in and backchannel detection for one conversation."""

    def __init__(self, sample_rate: int, config: TurnConfig | None = None, *, language: str | None = None,
                 personal_pause_s: float | None = None, latency: LatencyTracker | None = None) -> None:
        self.sr = int(sample_rate)
        self.cfg = config or TurnConfig()
        self.language = language
        # the person's long within-turn pause (their ~90th-percentile pause), learned from their profile
        self.pause_s = personal_pause_s or self.cfg.default_pause_s
        self.latency = latency or LatencyTracker()
        self.hop = max(1, int(round(HOP_S * self.sr)))
        self.n = max(self.hop * 2, int(round(FRAME_S * self.sr)))
        self._win = np.hanning(self.n)
        nfft = 1 << int(math.ceil(math.log2(2 * self.n)))
        self._nfft = nfft
        w_ac = np.fft.irfft(np.abs(np.fft.rfft(self._win, nfft)) ** 2)[: self.n]
        self._w_ac = np.maximum(w_ac / w_ac[0], 1e-3)
        self._lag_min = max(2, int(self.sr / 500))
        self._lag_max = min(self.n - 2, int(math.ceil(self.sr / 60)))
        self._pick = 0.98 if self.sr <= 11025 else 0.9
        self._tail = np.zeros(self.n)          # last FRAME_S of audio
        self._pending = np.zeros(0)
        self._hops = 0                          # hops processed
        self._noise_hist: deque[float] = deque(maxlen=int(3.0 / HOP_S))
        self._noise = -70.0
        self._hist: deque[tuple[float, float, float]] = deque(maxlen=int(1.5 / HOP_S))  # (t, db, f0)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(self.cfg.preroll_s / HOP_S)))
        self.agent_speaking = False
        self._text = ""
        self._text_lang: str | None = None
        self._text_t = -1.0                     # stream time of the latest transcript
        self._reset_turn()
        self._run_len = 0                       # consecutive speech hops
        self._sil_len = 0                       # consecutive non-speech hops

    # ---------------------------------------------------------------- public API
    @classmethod
    def from_profile(cls, engine: Any, speaker_id: str, sample_rate: int, **kw: Any) -> "TurnDetector":
        """Calibrate to a known speaker: their usual pauses and languages from the speech profile."""
        pause = None
        lang = kw.pop("language", None)
        try:
            p = engine.get_profile(speaker_id)
            eot_ms = (p.get("agent_guidance") or {}).get("end_of_turn_silence_ms")
            if eot_ms:
                pause = max(0.2, (eot_ms - 250) / 1000.0)
            langs = (p.get("speaker") or {}).get("languages") or []
            lang = lang or (langs[0] if langs else None)
        except Exception:
            pass
        return cls(sample_rate, language=lang, personal_pause_s=pause, **kw)

    @property
    def now(self) -> float:
        return self._hops * self.hop / self.sr

    def push_audio(self, chunk: Any, encoding: str = "pcm16") -> list[TurnEvent]:
        """Add audio (numpy float/int16 array, PCM16 bytes, or mu-law bytes with encoding="mulaw")."""
        t0 = time.perf_counter()
        x = _to_float(chunk, encoding)
        if x.size == 0:
            return []
        self.latency.audio_s += x.size / self.sr
        buf = np.concatenate([self._pending, x]) if self._pending.size else x
        events: list[TurnEvent] = []
        i = 0
        while i + self.hop <= buf.size:
            h0 = time.perf_counter()
            events += self._step(buf[i:i + self.hop])
            self.latency.hop_times_us.append((time.perf_counter() - h0) * 1e6)
            i += self.hop
        self._pending = buf[i:].copy()
        self.latency.proc_s += time.perf_counter() - t0
        return events

    def push_transcript(self, text: str, is_final: bool = False, language: str | None = None) -> list[TurnEvent]:
        """Latest partial or final ASR text for the current user turn (cumulative for the turn is best)."""
        text = text.strip()
        if is_final:
            # ASR services split long turns into several final segments: keep them all
            if self._final_text and not text.startswith(self._final_text):
                self._final_text = f"{self._final_text} {text}".strip()
            else:
                self._final_text = text
            self._text = self._final_text
        else:
            self._text = f"{self._final_text} {text}".strip() if self._final_text else text
        if language:
            self._text_lang = language
        self._text_t = self.now
        return []

    def set_agent_speaking(self, speaking: bool) -> list[TurnEvent]:
        """Tell the detector when your TTS starts and stops playing."""
        if speaking and not self.agent_speaking:
            self.latency.agent_started(self.now)
        self.agent_speaking = speaking
        return []

    # ---------------------------------------------------------------- internals
    def _reset_turn(self) -> None:
        self.in_turn = False
        self._turn_start = 0.0
        self._turn_audio: list[np.ndarray] = []
        self._turn_samples = 0
        self._turn_dbs: list[float] = []
        self._pauses: list[float] = []
        self._pause_start: float | None = None
        self._last_speech_hist: list[tuple[float, float, float]] = []
        self._final_text = ""
        self._text = ""
        self._text_lang = None
        self._interrupt_state: str | None = None   # None | "candidate" | "barge_in"
        self._run_start = 0.0

    def _frame_f0(self, frame: np.ndarray) -> float:
        fr = (frame - frame.mean()) * self._win
        spec = np.fft.rfft(fr, self._nfft)
        ac = np.fft.irfft(np.abs(spec) ** 2)[: self.n]
        if ac[0] <= 0:
            return float("nan")
        ac = (ac / ac[0]) / self._w_ac
        row = ac[self._lag_min:self._lag_max + 1]
        lm = np.flatnonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])) + 1
        if lm.size == 0:
            return float("nan")
        best = row[lm].max()
        if best < 0.45:
            return float("nan")
        k = int(lm[row[lm] >= self._pick * best][0])
        if 0 < k < row.size - 1:
            a, b, c = row[k - 1], row[k], row[k + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den < 0 else 0.0
        else:
            off = 0.0
        return self.sr / (k + off + self._lag_min)

    def _step(self, hop: np.ndarray) -> list[TurnEvent]:
        self._tail = np.concatenate([self._tail[self.hop:], hop])
        self._hops += 1
        t = self.now
        rms = math.sqrt(float(np.mean(hop * hop))) + 1e-10
        db = 20 * math.log10(rms)
        # adaptive noise floor: low percentile of the last 3 s, refreshed every 10 hops
        self._noise_hist.append(db)
        if self._hops % 10 == 0 or self._hops < 20:
            # capped at -40 dBFS: if a stream starts mid-speech, the early "floor" is really speech
            self._noise = min(-40.0, max(-90.0, float(np.percentile(np.fromiter(self._noise_hist, float), 5))))
        margin = self.cfg.vad_margin_db + (self.cfg.agent_echo_margin_db if self.agent_speaking else 0.0)
        floor = self.cfg.vad_floor_dbfs
        if self.in_turn and not self.agent_speaking:
            # hysteresis: once someone holds the floor, softer sounds count as them continuing
            margin, floor = max(3.0, margin - self.cfg.resume_margin_db), floor - self.cfg.resume_margin_db
        speech = db > max(self._noise + margin, floor)
        f0 = self._frame_f0(self._tail) if speech else float("nan")
        self._hist.append((t, db, f0))
        if not self.in_turn:
            self._preroll.append(hop)
        if speech:
            self._run_len += 1
            self._sil_len = 0
        else:
            self._sil_len += 1
            self._run_len = 0
        events: list[TurnEvent] = []

        if not self.in_turn:
            if self._run_len >= max(1, int(self.cfg.min_speech_s / HOP_S)):
                events += self._start_turn(t)
        else:
            self._buffer(hop)
            if speech:
                self._turn_dbs.append(db)
                if self._pause_start is not None and self._run_len >= max(1, int(self.cfg.resume_speech_s / HOP_S)):
                    self._pauses.append(t - self._pause_start - self._run_len * HOP_S)
                    self._pause_start = None
            elif self._pause_start is None and self._sil_len >= 2:
                self._pause_start = t - self._sil_len * HOP_S
                self._last_speech_hist = [h for h in self._hist if h[0] <= self._pause_start]
            events += self._interruptions(t, speech)
            if self.in_turn and self._pause_start is not None:
                events += self._maybe_end(t)
        return events

    def _start_turn(self, t: float) -> list[TurnEvent]:
        start = t - self._run_len * HOP_S
        self.in_turn = True
        self._turn_start = start
        self._run_start = start
        self._text = self._final_text = ""   # drop late ASR text that belonged to the previous turn
        self._text_lang = None
        for h in self._preroll:
            self._buffer(h)
        self._preroll.clear()
        self._interrupt_state = "candidate" if self.agent_speaking else None
        return [TurnEvent("speech_start", round(start, 3), {"during_agent_speech": self.agent_speaking})]

    def _buffer(self, hop: np.ndarray) -> None:
        if self._turn_samples < self.cfg.max_turn_s * self.sr:
            self._turn_audio.append(hop)
            self._turn_samples += hop.size

    def _interruptions(self, t: float, speech: bool) -> list[TurnEvent]:
        if self._interrupt_state != "candidate":
            return []
        dur = (self._pause_start or t) - self._run_start
        has_text = bool(self._text.strip())
        backchannelish = has_text and is_backchannel(self._text, self._text_lang or self.language)
        # real words -> react fast; no text yet -> wait a little; "yeah"/"mm-hmm" -> only if it keeps going
        need = (self.cfg.backchannel_max_s if backchannelish
                else self.cfg.barge_in_min_s if has_text else self.cfg.barge_in_no_text_s)
        if speech and dur >= need and (not backchannelish or dur > self.cfg.backchannel_max_s):
            self._interrupt_state = "barge_in"
            return [TurnEvent("barge_in", round(t, 3), {"speech_s": round(dur, 3), "transcript": self._text})]
        if not speech and self._sil_len * HOP_S >= 0.15:
            if dur <= self.cfg.backchannel_max_s:
                ev = TurnEvent("backchannel", round(t, 3), {"speech_s": round(dur, 3), "transcript": self._text})
                self._reset_turn()
                return [ev]
            self._interrupt_state = "barge_in"
            return [TurnEvent("barge_in", round(t, 3), {"speech_s": round(dur, 3), "transcript": self._text})]
        return []

    def _prosody(self) -> dict[str, Any]:
        """Pitch slope and loudness at the end of the speech before the current pause."""
        hist = self._last_speech_hist
        voiced = [(tt, f) for tt, _, f in hist if not math.isnan(f)]
        out: dict[str, Any] = {"contour": "unknown", "slope_st_per_s": None, "energy_drop_db": None}
        if len(voiced) >= 5:
            t_end = voiced[-1][0]
            tail = [(tt, f) for tt, f in voiced if tt >= t_end - 0.5]
            if len(tail) >= 5:
                tt = np.array([a for a, _ in tail])
                st = 12 * np.log2(np.array([b for _, b in tail]) / np.median([b for _, b in voiced]))
                slope = float(np.polyfit(tt, st, 1)[0])
                out["slope_st_per_s"] = round(slope, 2)
                out["contour"] = "falling" if slope < -2.5 else "rising" if slope > 3.0 else "level"
        speech_db = [d for d in self._turn_dbs]
        last = [d for tt, d, _ in hist if tt >= (hist[-1][0] - 0.2 if hist else 0)]
        if len(speech_db) >= 10 and last:
            out["energy_drop_db"] = round(float(np.median(speech_db) - np.mean(last)), 2)
        return out

    def end_probability(self, silence_s: float) -> tuple[float, list[str], dict[str, Any], dict[str, bool]]:
        """Probability that the turn is over after `silence_s` of silence, with the reasons."""
        pros = self._prosody()
        reasons: list[str] = []
        # Words count only once the transcript has caught up with the latest speech: ASR text lags the
        # audio, and text from before the person resumed speaking says nothing about how they stopped now.
        fresh = self._pause_start is not None and self._text_t >= self._pause_start - LEXICAL_TOLERANCE_S
        if fresh:
            lex = lexical_cues(self._text, self._text_lang or self.language)
        else:
            lex = {"continuation": False, "complete": False, "question": False}
            if self._text:
                reasons.append("waiting for the latest words")
        logit = 4.0 * (silence_s / self.pause_s - 1.0)
        if pros["contour"] == "falling":
            logit += 1.2
            reasons.append("pitch fell at the end")
        elif pros["contour"] == "level":
            logit -= 0.6
            reasons.append("pitch stayed level (often 'more to come')")
        elif pros["contour"] == "rising":
            logit += 1.0 if lex["question"] else 0.2
            reasons.append("pitch rose" + (" on a question" if lex["question"] else ""))
        if pros["energy_drop_db"] is not None and pros["energy_drop_db"] >= 6:
            logit += 0.4
            reasons.append("voice trailed off")
        if lex["continuation"]:
            logit -= 4.0
            reasons.append("stopped on a word that usually continues")
        elif lex["complete"] or lex["question"]:
            logit += 1.0
            reasons.append("words form a complete sentence" if not lex["question"] else "asked a question")
        reasons.append(f"silent {silence_s * 1000:.0f} ms (their long pause ~{self.pause_s * 1000:.0f} ms)")
        return 1 / (1 + math.exp(-logit)), reasons, pros, lex

    def _maybe_end(self, t: float) -> list[TurnEvent]:
        if self._interrupt_state == "candidate":   # still deciding barge-in vs backchannel
            return []
        silence = t - self._pause_start
        if self.cfg.fixed_timeout_s is not None:   # baseline mode: plain silence timeout, no cues
            if silence < self.cfg.fixed_timeout_s:
                return []
            return self._emit_end(t, silence, 1.0, [f"fixed {self.cfg.fixed_timeout_s * 1000:.0f} ms timeout"],
                                  {"contour": "unknown", "slope_st_per_s": None, "energy_drop_db": None},
                                  {"continuation": False, "complete": False, "question": False})
        if silence < self.cfg.min_silence_s:
            return []
        p, reasons, pros, lex = self.end_probability(silence)
        max_wait = min(self.cfg.hard_cap_s, max(self.cfg.max_silence_s, self.pause_s + 0.25))
        if lex["continuation"]:
            max_wait = min(self.cfg.hard_cap_s, max_wait + self.cfg.hesitation_bonus_s)
        if p < self.cfg.threshold and silence < max_wait:
            return []
        if p < self.cfg.threshold:
            reasons.append("reached the maximum wait")
        return self._emit_end(t, silence, p, reasons, pros, lex)

    def _emit_end(self, t: float, silence: float, p: float, reasons: list[str], pros: dict[str, Any],
                  lex: dict[str, bool]) -> list[TurnEvent]:
        speech_end = self._pause_start
        self.latency.turn_detected(speech_end, t)
        audio = np.concatenate(self._turn_audio) if self._turn_audio else np.zeros(0)
        data = {
            "probability": round(p, 3), "reasons": reasons, "silence_ms": round(silence * 1000),
            "speech_end_t": round(speech_end, 3), "turn_start_t": round(self._turn_start, 3),
            "turn_duration_s": round(speech_end - self._turn_start, 3), "pauses_s": [round(x, 3) for x in self._pauses],
            "transcript": self._text, "language": _norm_lang(self._text_lang or self.language) or detect_language(self._text),
            "prosody": pros, "lexical": lex, "interrupted_agent": self._interrupt_state == "barge_in",
            "audio": audio.astype(np.float32), "sample_rate": self.sr,
        }
        self._reset_turn()
        return [TurnEvent("end_of_turn", round(t, 3), data)]


def analyze_turn(engine: Any, speaker_id: str, event: TurnEvent, *, session_id: str | None = None,
                 response_latency_s: float | None = None) -> dict[str, Any]:
    """Run the full Paralingo analysis on an ``end_of_turn`` event (call it in a background thread)."""
    d = event.data
    return engine.analyze_utterance(speaker_id, session_id=session_id, samples=d["audio"], sample_rate=d["sample_rate"],
                                    transcript=d.get("transcript") or None, response_latency_s=response_latency_s)
