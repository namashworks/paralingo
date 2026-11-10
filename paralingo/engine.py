# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""High-level API that voice agents use: sessions, utterance analysis, observations,
accent updates, consent, summaries, trends and guidance - all persisted to the
speaker's ``.speech.json`` file via :class:`SpeechProfileStore`.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from . import audio as A
from . import language as L
from . import reasoning as R
from .store import ConsentError, ProfileError, SpeechProfileStore, new_id, utcnow

OBS_CATEGORIES = {"emotion", "communication", "accent", "speech_pattern", "health_indicator", "voice_quality",
                  "behavior", "preference", "other"}


class SpeechProfileEngine:
    def __init__(self, store: SpeechProfileStore | str | os.PathLike[str], agent_id: str = "voice-agent") -> None:
        self.store = store if isinstance(store, SpeechProfileStore) else SpeechProfileStore(store)
        self.agent_id = agent_id

    # ================================================================== profile
    def ensure_profile(self, speaker_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.store.create(speaker_id, agent_id=self.agent_id, exist_ok=True, **kwargs)

    def set_consent(self, speaker_id: str, **consent: Any) -> dict[str, Any]:
        allowed = {"recording", "voice_analysis", "health_indicators", "store_transcripts", "retention_days", "granted_via"}
        bad = set(consent) - allowed
        if bad:
            raise ProfileError(f"unknown consent fields: {sorted(bad)}")
        with self.store.edit(speaker_id, agent_id=self.agent_id, action="set_consent", detail=str(consent),
                             create_if_missing=True) as p:
            p["consent"].update(consent)
            p["consent"]["granted_at"] = utcnow()
            if not p["consent"]["health_indicators"]:
                p["assessments"]["speech_indicators"] = []
                p["observations"] = [o for o in p["observations"] if o["category"] != "health_indicator"]
            return dict(p["consent"])

    def update_speaker(self, speaker_id: str, **fields: Any) -> dict[str, Any]:
        with self.store.edit(speaker_id, agent_id=self.agent_id, action="update_speaker", create_if_missing=True) as p:
            for k in ("display_name", "languages", "notes"):
                if k in fields:
                    p["speaker"][k] = fields[k]
            if "self_reported" in fields:
                p["speaker"]["self_reported"].update(fields["self_reported"])
            return dict(p["speaker"])

    # ================================================================== sessions
    def start_session(self, speaker_id: str, *, channel: str | None = None, context: str | None = None,
                      session_id: str | None = None) -> str:
        sid = session_id or new_id("sess")
        with self.store.edit(speaker_id, agent_id=self.agent_id, action="start_session", detail=sid,
                             create_if_missing=True) as p:
            if any(s["session_id"] == sid for s in p["sessions"]):
                return sid
            p["sessions"].append({"session_id": sid, "started_at": utcnow(), "ended_at": None, "status": "active",
                                  "agent_id": self.agent_id, "channel": channel, "context": context,
                                  "utterances": [], "utterances_compacted": False, "summary": None})
        return sid

    def _active_session(self, p: dict[str, Any], session_id: str | None, create: bool = True) -> dict[str, Any]:
        if session_id:
            for s in p["sessions"]:
                if s["session_id"] == session_id:
                    if s["status"] != "active":
                        raise ProfileError(f"session {session_id} is closed")
                    return s
            raise ProfileError(f"unknown session {session_id}")
        active = [s for s in p["sessions"] if s["status"] == "active"]
        if active:
            return active[-1]
        if not create:
            raise ProfileError("no active session")
        s = {"session_id": new_id("sess"), "started_at": utcnow(), "ended_at": None, "status": "active",
             "agent_id": self.agent_id, "channel": None, "context": None, "utterances": [],
             "utterances_compacted": False, "summary": None}
        p["sessions"].append(s)
        return s

    # ================================================================== utterances
    def analyze_utterance(
        self,
        speaker_id: str,
        *,
        session_id: str | None = None,
        transcript: str | None = None,
        audio_path: str | None = None,
        samples: np.ndarray | None = None,
        sample_rate: int | None = None,
        word_timestamps: list[dict[str, Any]] | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
        response_latency_s: float | None = None,
        acoustics: dict[str, Any] | None = None,
        external: dict[str, Any] | None = None,
        audio_ref: str | None = None,
    ) -> dict[str, Any]:
        """Analyse one user turn and store it. Supply any combination of:

        * ``audio_path`` (WAV) or ``samples`` + ``sample_rate``  -> full acoustic analysis
        * ``transcript``                                         -> fluency / lexical / sentiment
        * ``word_timestamps`` from ASR                           -> pauses & rate without audio
        * ``acoustics``  - precomputed features (same shape as audio.analyze_audio output)
        * ``external``   - other models' outputs, e.g. {"emotion": {"label": "angry", "confidence": .8,
                           "arousal": .7, "valence": -.6, "model": "my-ser"}, "asr_confidence": .93,
                           "language_id": "en-IN"}
        """
        # ---- heavy lifting outside the file lock
        ac: dict[str, Any] = {}
        if audio_path is not None:
            x, sr = A.load_wav(audio_path)
            ac = A.analyze_audio(x, sr)
            audio_ref = audio_ref or str(Path(audio_path).name)
        elif samples is not None:
            if not sample_rate:
                raise ProfileError("sample_rate is required with samples")
            ac = A.analyze_audio(np.asarray(samples), int(sample_rate))
        if acoustics:
            ac = _deep_merge(ac, acoustics)

        text = L.analyze_text(transcript or "") if transcript else None
        tokens = L.tokenize(transcript or "")

        pauses = ac.get("pauses")
        speech_s = ac.get("speech_s")
        span = ac.get("speech_span_s")
        if (not pauses) and word_timestamps:
            pauses = A.pauses_from_word_timestamps(word_timestamps)
            speech_s, span = pauses.pop("speech_s"), pauses.pop("speech_span_s")
        if span is None and t_start is not None and t_end is not None:
            span = max(0.0, t_end - t_start)
            speech_s = speech_s or span

        timing: dict[str, Any] = {"pauses": pauses, "response_latency_s": response_latency_s,
                                  "speech_s": speech_s, "words_per_minute": None, "articulation_rate_sps": None}
        if text and span:
            wc = text["lexical"]["content_word_count"]
            timing["words_per_minute"] = round(wc / span * 60.0, 1) if span > 0.3 else None
            if speech_s and speech_s > 0.3:
                timing["articulation_rate_sps"] = round(text["lexical"]["syllable_count"] / speech_s, 2)

        utt: dict[str, Any] = {
            "utterance_id": new_id("utt"),
            "timestamp": utcnow(),
            "t_start": t_start,
            "t_end": t_end,
            "transcript": transcript,
            "audio_ref": audio_ref,
            "acoustics": {k: v for k, v in ac.items() if k not in ("pauses", "speech_span_s")} or None,
            "timing": timing,
            "fluency": text["fluency"] if text else None,
            "lexical": text["lexical"] if text else None,
            "sentiment": text["sentiment"] if text else None,
            "external": external or None,
        }
        utt = {k: v for k, v in utt.items() if v is not None or k in ("transcript",)}

        with self.store.edit(speaker_id, agent_id=self.agent_id, action="analyze_utterance",
                             create_if_missing=True) as p:
            if not p["consent"].get("voice_analysis", False):
                raise ConsentError("speaker has not consented to voice analysis")
            sess = self._active_session(p, session_id)
            base = p["baseline"]
            utt["emotion"] = R.estimate_emotion(utt, base, tokens)
            utt["skills"] = R.score_skills(utt)
            metrics = R.utterance_metrics(utt)
            utt["baseline_deviations"] = R.deviations(metrics, base)
            utt["flags"] = _utterance_flags(utt)
            if not p["consent"].get("store_transcripts", True):
                utt["transcript"] = None
            sess["utterances"].append(utt)
            cap = p.get("settings", {}).get("max_utterances_per_session", 300)
            del sess["utterances"][:-cap]

            # live, per-turn updates so the agent can adapt mid-conversation
            es = p["assessments"]["emotional_state"]
            es["current"] = utt["emotion"]
            es["recent_labels"] = (es.get("recent_labels", []) + [utt["emotion"]["label"]])[-10:]
            self._refresh_style_and_guidance(p, sess["utterances"][-20:])
            sid = sess["session_id"]

        return {
            "session_id": sid,
            "utterance_id": utt["utterance_id"],
            "emotion": utt["emotion"],
            "skills": utt["skills"],
            "flags": utt["flags"],
            "baseline_deviations": utt["baseline_deviations"],
            "key_metrics": {k: v for k, v in metrics.items() if k in (
                "f0_mean_hz", "f0_std_semitones", "rms_dbfs_mean", "words_per_minute", "articulation_rate_sps",
                "pause_ratio", "filler_rate_per_100w", "hnr_db", "jitter_local_pct", "shimmer_local_pct")},
            "audio_quality": get(utt, "acoustics", "audio_quality", "quality"),
            "guidance": p["agent_guidance"],
        }

    def end_session(self, speaker_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Close a session: summarise it, update baseline, trends, skills, style, indicators, guidance."""
        with self.store.edit(speaker_id, agent_id=self.agent_id, action="end_session", detail=session_id) as p:
            sess = self._active_session(p, session_id, create=False)
            sess["status"] = "closed"
            sess["ended_at"] = utcnow()
            utts = sess["utterances"]
            rows = [R.utterance_metrics(u) for u in utts]
            agg = R.mean_metrics(rows)
            base = p["baseline"]
            when = sess["ended_at"]

            # indicators BEFORE folding this session into the baseline (compare to the past self)
            hits: list[dict[str, Any]] = []
            evaluated = bool(p["consent"].get("health_indicators"))
            if evaluated:
                hits = R.evaluate_indicators(sess, agg, base)

            skill_rows = [u.get("skills") or {} for u in utts]
            skills = {k: round(float(np.mean(v)), 1) for k in
                      {k for s in skill_rows for k in s}
                      if (v := [s[k] for s in skill_rows if s.get(k) is not None])}
            labels = Counter(u["emotion"]["label"] for u in utts if u.get("emotion"))
            sess["summary"] = {
                "utterance_count": len(utts),
                "total_speech_s": round(sum(get(u, "timing", "speech_s") or 0 for u in utts), 2),
                "total_words": int(sum(get(u, "lexical", "word_count") or 0 for u in utts)),
                "metrics_mean": agg,
                "skills_mean": skills,
                "dominant_emotion": labels.most_common(1)[0][0] if labels else None,
                "emotion_timeline": [u["emotion"]["label"] for u in utts if u.get("emotion")][:100],
                "deviations_from_baseline": R.deviations(agg, base),
                "indicators_evaluated": evaluated,
                "indicators_triggered": [h["indicator"] for h in hits],
            }
            if evaluated:
                R.merge_indicators(p, sess["session_id"], hits, when)

            # baseline: fold in per-utterance metrics (skip poor audio for acoustic metrics - handled in utterance_metrics)
            for r in rows:
                R.baseline_update(base, {k: v for k, v in r.items() if k not in ("valence", "arousal", "sentiment")})
            if rows:
                base["sessions_used"] += 1
                base["utterances_used"] += len(rows)
                base["speech_seconds_used"] = round(base.get("speech_seconds_used", 0) + sess["summary"]["total_speech_s"], 2)
            base["status"] = R.baseline_status(base)

            for m in R.TREND_METRICS:
                if m in agg:
                    p["trends"].setdefault(m, []).append({"session_id": sess["session_id"], "date": when, "value": agg[m]})

            self._refresh_skills(p)
            self._refresh_emotion_trend(p)
            recent = [u for s in p["sessions"][-3:] for u in s["utterances"]][-40:]
            self._refresh_style_and_guidance(p, recent)
            return dict(sess["summary"])

    # ================================================================== observations / accent
    def add_observation(self, speaker_id: str, *, category: str, observation: str, confidence: float,
                        evidence: list[str] | None = None, reasoning: str | None = None,
                        session_id: str | None = None, utterance_id: str | None = None,
                        supersedes: str | None = None) -> str:
        if category not in OBS_CATEGORIES:
            raise ProfileError(f"category must be one of {sorted(OBS_CATEGORIES)}")
        oid = new_id("obs")
        with self.store.edit(speaker_id, agent_id=self.agent_id, action="add_observation", detail=category,
                             create_if_missing=True) as p:
            if category == "health_indicator" and not p["consent"].get("health_indicators"):
                raise ConsentError("speaker has not consented to health-indicator tracking")
            p["observations"].append({
                "observation_id": oid, "timestamp": utcnow(), "agent_id": self.agent_id, "category": category,
                "observation": observation, "evidence": evidence or [], "reasoning": reasoning,
                "confidence": max(0.0, min(1.0, float(confidence))), "session_id": session_id,
                "utterance_id": utterance_id, "supersedes": supersedes})
        return oid

    def update_accent(self, speaker_id: str, *, label: str | None = None, confidence: float = 0.5,
                      source: str | None = None, markers: list[dict[str, Any] | str] | None = None,
                      intelligibility: float | None = None, session_id: str | None = None) -> dict[str, Any]:
        with self.store.edit(speaker_id, agent_id=self.agent_id, action="update_accent", detail=label,
                             create_if_missing=True) as p:
            acc = p["accent"]
            if label:
                acc["estimates"].append({"label": label, "confidence": max(0.0, min(1.0, float(confidence))),
                                         "source": source or self.agent_id, "timestamp": utcnow(),
                                         "session_id": session_id})
                del acc["estimates"][:-100]
                votes: Counter[str] = Counter()
                counts: Counter[str] = Counter()
                for e in acc["estimates"][-30:]:
                    votes[e["label"]] += e["confidence"]
                    counts[e["label"]] += 1
                top, w = votes.most_common(1)[0]
                # agreement share x mean confidence of the winning label
                conf = (w / sum(votes.values())) * (w / counts[top])
                acc["current_estimate"] = {"label": top, "confidence": round(conf, 3), "votes": round(w, 3)}
            for mk in markers or []:
                mk = {"marker": mk} if isinstance(mk, str) else mk
                cur = next((m for m in acc["phonetic_markers"] if m["marker"] == mk["marker"]), None)
                if cur:
                    cur["count"] = cur.get("count", 1) + 1
                    cur["last_seen"] = utcnow()
                    if mk.get("example"):
                        cur["example"] = mk["example"]
                else:
                    acc["phonetic_markers"].append({"marker": mk["marker"], "example": mk.get("example"),
                                                    "count": 1, "last_seen": utcnow()})
            if intelligibility is not None:
                old = acc.get("intelligibility")
                acc["intelligibility"] = round(float(intelligibility) if old is None else 0.7 * old + 0.3 * float(intelligibility), 3)
            self._refresh_style_and_guidance(p, [u for s in p["sessions"][-2:] for u in s["utterances"]][-30:])
            return dict(acc)

    # ================================================================== read side
    def get_profile(self, speaker_id: str) -> dict[str, Any]:
        return self.store.load(speaker_id)

    def get_summary(self, speaker_id: str) -> dict[str, Any]:
        """Compact, LLM-friendly snapshot (what an agent should load at call start)."""
        p = self.store.load(speaker_id)
        a = p["assessments"]
        last = next((s for s in reversed(p["sessions"]) if s.get("summary")), None)
        active = next((s for s in reversed(p["sessions"]) if s["status"] == "active"), None)
        obs = sorted(p["observations"], key=lambda o: o["timestamp"])[-8:]
        summary = {
            "speaker": {k: p["speaker"].get(k) for k in ("speaker_id", "display_name", "languages")},
            "consent": p["consent"],
            "revision": p["revision"],
            "baseline_status": p["baseline"]["status"],
            "sessions_total": len(p["sessions"]),
            "active_session_id": active["session_id"] if active else None,
            "emotional_state": a["emotional_state"],
            "communication_skills": a["communication_skills"],
            "speaking_style": a["speaking_style"],
            "accent": {"estimate": p["accent"].get("current_estimate"),
                       "markers": [m["marker"] for m in p["accent"].get("phonetic_markers", [])][:10],
                       "intelligibility": p["accent"].get("intelligibility")},
            "speech_indicators": [{k: i[k] for k in ("indicator", "status", "occurrences", "evidence")}
                                  for i in a["speech_indicators"] if i["status"] != "resolved"],
            "last_session": ({"session_id": last["session_id"], "ended_at": last.get("ended_at"),
                              "dominant_emotion": last["summary"].get("dominant_emotion"),
                              "deviations_from_baseline": last["summary"].get("deviations_from_baseline")}
                             if last else None),
            "recent_observations": [{k: o[k] for k in ("category", "observation", "confidence", "agent_id", "timestamp")}
                                    for o in obs],
            "agent_guidance": p["agent_guidance"],
        }
        summary["prompt_context"] = render_prompt_context(summary)
        return summary

    def get_trend(self, speaker_id: str, metric: str, last_n: int = 10) -> dict[str, Any]:
        p = self.store.load(speaker_id)
        pts = p["trends"].get(metric, [])[-last_n:]
        vals = [x["value"] for x in pts if x["value"] is not None]
        direction = "insufficient_data"
        slope = None
        if len(vals) >= 3:
            slope = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
            sd = R.STD_FLOOR.get(metric, 0.0) or (float(np.std(vals)) or 1.0)
            direction = "increasing" if slope > 0.25 * sd else "decreasing" if slope < -0.25 * sd else "stable"
        b = p["baseline"]["metrics"].get(metric)
        return {"metric": metric, "points": pts, "direction": direction,
                "slope_per_session": round(slope, 4) if slope is not None else None,
                "baseline": {k: b[k] for k in ("n", "mean", "std")} if b else None,
                "available_metrics": sorted(p["trends"])}

    def query_observations(self, speaker_id: str, category: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        obs = self.store.load(speaker_id)["observations"]
        if category:
            obs = [o for o in obs if o["category"] == category]
        return obs[-limit:]

    # ================================================================== internals
    def _refresh_style_and_guidance(self, p: dict[str, Any], recent_utts: list[dict[str, Any]]) -> None:
        rows = [R.utterance_metrics(u) for u in recent_utts]
        agg = R.mean_metrics(rows) if rows else {}
        lat = [r["response_latency_s"] for r in rows if "response_latency_s" in r]
        if rows:
            p["assessments"]["speaking_style"] = R.speaking_style(agg, p["baseline"], median(lat) if lat else None)
        g = R.build_guidance(p, recent_utts)
        g["updated_at"] = utcnow()
        p["agent_guidance"] = g

    def _refresh_skills(self, p: dict[str, Any]) -> None:
        recent = [s["summary"]["skills_mean"] for s in p["sessions"] if s.get("summary")][-5:]
        keys = {k for r in recent for k in r}
        scores = {k: round(float(np.mean([r[k] for r in recent if k in r])), 1) for k in sorted(keys)}
        named = {k: v for k, v in scores.items() if k != "overall"}
        cs = p["assessments"]["communication_skills"]
        cs["scores"] = scores
        cs["strengths"] = [k for k, v in sorted(named.items(), key=lambda kv: -kv[1]) if v >= 75][:3]
        cs["areas_to_improve"] = [k for k, v in sorted(named.items(), key=lambda kv: kv[1]) if v < 60][:3]
        cs["updated_at"] = utcnow()

    def _refresh_emotion_trend(self, p: dict[str, Any]) -> None:
        vals = [x["value"] for x in p["trends"].get("valence", [])]
        es = p["assessments"]["emotional_state"]
        if len(vals) >= 2:
            prev = float(np.mean(vals[-4:-1]))
            d = vals[-1] - prev
            es["trend"] = "improving" if d > 0.15 else "worsening" if d < -0.15 else "stable"


# --------------------------------------------------------------------------- helpers
def get(d: dict[str, Any] | None, *path: str) -> Any:
    return R.get_path(d, tuple(path))


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _utterance_flags(u: dict[str, Any]) -> list[str]:
    f = []
    if get(u, "acoustics", "audio_quality", "quality") == "poor":
        f.append("poor_audio")
    if (get(u, "timing", "pauses", "long_count") or 0) > 0:
        f.append("long_pause")
    if (get(u, "fluency", "total_disfluency_per_100w") or 0) >= 10 and (get(u, "lexical", "word_count") or 0) >= 5:
        f.append("high_disfluency")
    if (get(u, "fluency", "part_word_repetitions") or 0) > 0:
        f.append("part_word_repetition")
    wpm = get(u, "timing", "words_per_minute")
    if wpm and (get(u, "lexical", "word_count") or 0) >= 5:
        if wpm < 100:
            f.append("slow_rate")
        elif wpm > 200:
            f.append("fast_rate")
    if (get(u, "acoustics", "pitch", "f0_std_semitones") or 99) < 1.0:
        f.append("flat_intonation")
    if get(u, "acoustics", "pitch", "terminal_contour") == "rising":
        f.append("rising_terminal")
    if len(u.get("baseline_deviations") or {}) >= 2:
        f.append("deviates_from_baseline")
    emo = u.get("emotion") or {}
    if emo.get("label") in {"angry", "frustrated", "anxious", "sad"} and emo.get("confidence", 0) >= 0.35:
        f.append(f"emotion_{emo['label']}")
    return f


def render_prompt_context(s: dict[str, Any]) -> str:
    """Plain-text block to drop into a voice agent's system prompt."""
    lines = [f"SPEECH PROFILE for {s['speaker'].get('display_name') or s['speaker']['speaker_id']} "
             f"(baseline: {s['baseline_status']}, sessions: {s['sessions_total']})"]
    st = s["speaking_style"]
    lines.append(f"- Style: pace {st['pace']}, intonation {st['pitch_expressiveness']}, pausing {st['pausing']}, "
                 f"verbosity {st['verbosity']}, directness {st['directness']}")
    emo = s["emotional_state"].get("current")
    if emo:
        lines.append(f"- Current emotion (estimate): {emo['label']} (confidence {emo['confidence']}); "
                     f"trend {s['emotional_state'].get('trend')}")
    sc = s["communication_skills"].get("scores") or {}
    if sc:
        lines.append("- Communication scores: " + ", ".join(f"{k} {v:.0f}" for k, v in sc.items()))
    acc = s["accent"]
    if acc.get("estimate"):
        lines.append(f"- Accent (estimate): {acc['estimate']['label']} ({acc['estimate']['confidence']})")
    for i in s["speech_indicators"]:
        lines.append(f"- Speech indicator [{i['status']}]: {i['indicator']} (non-diagnostic)")
    g = s["agent_guidance"]
    if g.get("recommended_speaking_rate_wpm"):
        lines.append(f"- Speak at ~{g['recommended_speaking_rate_wpm']:.0f} wpm")
    if g.get("end_of_turn_silence_ms"):
        lines.append(f"- Wait ~{g['end_of_turn_silence_ms']} ms of silence before replying")
    for ad in g.get("adaptations", [])[:6]:
        lines.append(f"- [{ad['priority']}] {ad['instruction']}")
    lines.append("Never mention these inferences to the user unless asked; they are probabilistic.")
    return "\n".join(lines)
