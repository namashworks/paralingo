# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Acoustic / prosodic feature extraction (numpy only, no heavy deps).

Implements, in a lightweight form, the classic measures used in speech
science (cf. Praat / MDVP / eGeMAPS):

* F0 (pitch) via window-corrected normalised autocorrelation (Boersma 1993)
* Harmonics-to-noise ratio from the autocorrelation peak
* Jitter (local) and shimmer (local) from cycle-to-cycle waveform cross-correlation
* Loudness (RMS dBFS), dynamic range, spectral centroid, high/low band ratio
* Energy-based voice activity -> silent pauses, speech time, pause ratio
* Intonation: pitch variability / range in semitones, terminal contour
* Recording quality: SNR estimate, clipping ratio

Numbers are comparable *within* a speaker (personal baseline) and broadly
comparable to Praat defaults, but they are not clinically calibrated.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

FRAME_S = 0.040
HOP_S = 0.010
VOICING_THRESHOLD = 0.45
OCTAVE_PICK_RATIO = 0.9      # take the shortest-lag peak within this fraction of the best (guards octave-down errors)
OCTAVE_PICK_RATIO_NARROWBAND = 0.98  # telephone audio (sr <= 11025) lacks the fundamental; 0.9 causes octave-up errors
                                     # (validation/: phone F0 within 5 % of Praat 49 % -> 67 %, octave errors 4.2 % -> 0.7 %)
MAX_PERIOD_FACTOR = 1.3      # Praat default
MAX_AMPLITUDE_FACTOR = 1.6   # Praat default
CYCLE_CORR_MIN = 0.5         # min cycle-to-cycle correlation; tuned so jitter matches Praat on RAVDESS (validation/)


# --------------------------------------------------------------------------- I/O
def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a PCM WAV (8/16/24/32-bit, any channel count) as mono float in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        x = ints.astype(np.float64) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width {width}")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


def save_wav(path: str | Path, samples: np.ndarray, sr: int) -> None:
    """Write mono float samples as 16-bit PCM (handy for tests / demos)."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- helpers
def _r(v: Any, nd: int = 3) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else round(v, nd)


def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, n)[::hop]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _parabolic(y: np.ndarray, i: int) -> tuple[float, float]:
    """Sub-sample peak location / value around index i."""
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    # Interpolate only at a true local maximum (den < 0); then the offset is bounded to +/-0.5 sample.
    # On flat or edge peaks the formula can return offsets of hundreds of samples, so keep i.
    if b < a or b < c or den >= 0:
        return float(i), float(b)
    off = 0.5 * (a - c) / den
    return i + off, b - 0.25 * (a - c) * off


# --------------------------------------------------------------------------- main
def analyze_audio(
    samples: np.ndarray,
    sr: int,
    *,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    min_pause_s: float = 0.25,
    long_pause_s: float = 1.0,
    return_tracks: bool = False,
) -> dict[str, Any]:
    """Extract prosodic, voice-quality and timing features from one utterance.

    ``return_tracks=True`` adds the frame-level pitch track (``tracks.t`` seconds, ``tracks.f0_hz``,
    None where unvoiced), used by the validation harness.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    duration = len(x) / sr
    clipping_ratio = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    x = x - np.mean(x) if len(x) else x

    n = int(round(FRAME_S * sr))
    hop = int(round(HOP_S * sr))
    frames = _frames(x, n, hop)
    win = np.hanning(n)

    # ---------------- energy + VAD
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20 * np.log10(rms + 1e-10)
    # 3rd percentile: VAD-trimmed turns often contain only ~100 ms of silence, so a higher percentile
    # would land inside speech and make clean audio look noisy.
    noise_floor = float(np.percentile(rms_db, 3))
    peak = float(np.percentile(rms_db, 95))
    if peak - noise_floor < 6.0:
        speech = rms_db > -55.0 if peak > -55.0 else np.zeros_like(rms_db, dtype=bool)
    else:
        thr = noise_floor + max(6.0, 0.3 * (peak - noise_floor))
        speech = rms_db > max(thr, -60.0)

    min_gap = int(round(min_pause_s / HOP_S))
    min_blip = max(1, int(round(0.05 / HOP_S)))
    # remove tiny speech blips (clicks)
    for s, e in _runs(speech):
        if e - s < min_blip:
            speech[s:e] = False
    # pauses = silent runs strictly inside speech, >= min_pause_s
    seg = _runs(speech)
    pauses: list[tuple[float, float]] = []
    if seg:
        for (s1, e1), (s2, _e2) in zip(seg[:-1], seg[1:]):
            gap = s2 - e1
            if gap >= min_gap:
                pauses.append((e1 * HOP_S, s2 * HOP_S))
            else:
                speech[e1:s2] = True  # bridge micro-gaps (they're articulation, not pauses)
        seg = _runs(speech)
    speech_s = float(speech.sum() * HOP_S)
    pause_durs = np.array([b - a for a, b in pauses]) if pauses else np.zeros(0)
    span = (seg[-1][1] - seg[0][0]) * HOP_S if seg else 0.0

    # ---------------- pitch + HNR (Boersma window-corrected autocorrelation)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    w_ac = np.fft.irfft(np.abs(np.fft.rfft(win, nfft)) ** 2)[:n]
    w_ac = w_ac / w_ac[0]
    lag_min = max(2, int(np.floor(sr / f0_max)))
    lag_max = min(n - 2, int(np.ceil(sr / f0_min)))

    f0 = np.full(len(frames), np.nan)
    rpk = np.full(len(frames), np.nan)
    idx = np.flatnonzero(speech)
    if idx.size and lag_max > lag_min:
        fr = frames[idx] - frames[idx].mean(axis=1, keepdims=True)
        fr = fr * win
        spec = np.fft.rfft(fr, nfft, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
        ac0 = ac[:, :1]
        ac0[ac0 == 0] = 1e-12
        ac = (ac / ac0) / np.maximum(w_ac, 1e-3)
        seg_ac = ac[:, lag_min:lag_max + 1]
        pick_ratio = OCTAVE_PICK_RATIO_NARROWBAND if sr <= 11025 else OCTAVE_PICK_RATIO
        for k, row in enumerate(seg_ac):
            # local maxima; choose the shortest lag whose height is close to the best (anti octave-down)
            lm = np.flatnonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])) + 1
            if lm.size == 0:
                continue
            best = row[lm].max()
            if best < VOICING_THRESHOLD:
                continue
            pick = lm[row[lm] >= pick_ratio * best][0]
            lag, val = _parabolic(row, int(pick))
            f0[idx[k]] = sr / (lag + lag_min)
            rpk[idx[k]] = min(val, 0.999999)

    voiced = ~np.isnan(f0)
    if voiced.sum() >= 3:  # octave-jump correction vs utterance median
        med = np.nanmedian(f0)
        f0 = np.where(f0 / med > 1.8, f0 / 2, f0)
        f0 = np.where(f0 / med < 0.55, f0 * 2, f0)
    fv = f0[voiced]

    pitch: dict[str, Any] = {k: None for k in (
        "f0_mean_hz", "f0_median_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz",
        "f0_std_semitones", "f0_range_semitones", "terminal_contour", "terminal_slope_st_per_s")}
    hnr_db = None
    if fv.size >= 3:
        st = 12 * np.log2(fv / np.median(fv))
        pitch.update(
            f0_mean_hz=_r(fv.mean(), 1), f0_median_hz=_r(np.median(fv), 1), f0_std_hz=_r(fv.std(), 2),
            f0_min_hz=_r(np.percentile(fv, 5), 1), f0_max_hz=_r(np.percentile(fv, 95), 1),
            f0_std_semitones=_r(st.std(), 2),
            f0_range_semitones=_r(np.percentile(st, 95) - np.percentile(st, 5), 2),
        )
        # terminal contour over the last 0.5 s of voicing
        vi = np.flatnonzero(voiced)
        tail = vi[vi >= vi[-1] - int(0.5 / HOP_S)]
        if tail.size >= 5:
            slope = np.polyfit(tail * HOP_S, 12 * np.log2(f0[tail] / np.median(fv)), 1)[0]
            pitch["terminal_slope_st_per_s"] = _r(slope, 2)
            pitch["terminal_contour"] = "rising" if slope > 3 else "falling" if slope < -3 else "level"
        r = np.clip(rpk[voiced], 1e-6, 0.999999)
        hnr_db = _r(np.mean(10 * np.log10(r / (1 - r))), 2)

    # ---------------- jitter / shimmer: cycle-to-cycle waveform cross-correlation (like Praat's "periodic, cc")
    # Each period is the shift that best aligns one cycle with the next. Cycles that do not resemble the
    # previous one (correlation < CYCLE_CORR_MIN: onsets, offsets, creak, noise) end the current stretch.
    periods_all: list[np.ndarray] = []
    amps_all: list[np.ndarray] = []

    def _close(pers: list[float], amps: list[float]) -> None:
        if len(pers) >= 3:
            per, amp = np.asarray(pers), np.asarray(amps)
            ok = (per > 1 / f0_max) & (per < 1 / f0_min) & (amp > 0)
            if ok.sum() >= 3:
                periods_all.append(per[ok])
                amps_all.append(amp[ok])

    for s, e in _runs(voiced):
        if e - s < 3:
            continue
        a0, a1 = s * hop, min(len(x), e * hop + n)
        y = x[a0:a1]
        f0_run = f0[s:e]
        t_frames = np.arange(s, e) * hop + n / 2 - a0
        p = float(np.argmax(np.abs(y[: int(sr / f0_run[0]) + 1])))
        pers: list[float] = []
        amps: list[float] = []
        for _ in range(int(len(y) / (sr / f0_max)) + 2):  # hard bound on the number of cycles
            T = sr / np.interp(p, t_frames, f0_run)
            h = int(T / 2)
            dmin, dmax = int(0.8 * T), int(1.2 * T) + 1
            ip = int(p)
            if h < 2 or ip + dmax + h + 1 >= len(y):
                break
            if ip - h < 0:
                p += T
                continue
            tpl = y[ip - h: ip + h]
            cw = np.lib.stride_tricks.sliding_window_view(y[ip + dmin - h: ip + dmax + h], 2 * h)
            c = (cw @ tpl) / (np.sqrt((cw * cw).sum(axis=1) * (tpl @ tpl)) + 1e-12)
            k = int(np.argmax(c))
            kk, _ = _parabolic(c, k)
            q = p + dmin + kk
            if c[k] >= CYCLE_CORR_MIN:
                w0, w1 = max(0, int(q - h)), min(len(y), int(q + h) + 1)
                pers.append((dmin + kk) / sr)
                amps.append(float(np.max(np.abs(y[w0:w1]))))
            else:
                _close(pers, amps)
                pers, amps = [], []
            p = q
        _close(pers, amps)

    jitter = shimmer = None
    n_pulses = 0
    if periods_all:
        dj, ds, pm, am = [], [], [], []
        for per, amp in zip(periods_all, amps_all):
            n_pulses += per.size
            rp = per[1:] / per[:-1]
            okp = (rp < MAX_PERIOD_FACTOR) & (rp > 1 / MAX_PERIOD_FACTOR)
            dj.append(np.abs(np.diff(per))[okp])
            ra = amp[1:] / amp[:-1]
            oka = (ra < MAX_AMPLITUDE_FACTOR) & (ra > 1 / MAX_AMPLITUDE_FACTOR)
            ds.append(np.abs(np.diff(amp))[oka])
            pm.append(per)
            am.append(amp)
        dj_, ds_ = np.concatenate(dj), np.concatenate(ds)
        if dj_.size:
            jitter = _r(100 * dj_.mean() / np.concatenate(pm).mean(), 3)
        if ds_.size:
            shimmer = _r(100 * ds_.mean() / np.concatenate(am).mean(), 3)

    # ---------------- spectral
    centroid = hf_ratio = None
    if idx.size:
        mag = np.abs(np.fft.rfft(frames[idx] * win, nfft, axis=1)) ** 2
        freqs = np.fft.rfftfreq(nfft, 1 / sr)
        tot = mag.sum(axis=1) + 1e-20
        centroid = _r(np.mean((mag * freqs).sum(axis=1) / tot), 1)
        lo_b = mag[:, (freqs >= 50) & (freqs < 1000)].sum()
        hi_b = mag[:, (freqs >= 1000) & (freqs < min(5000, sr / 2))].sum()
        if lo_b > 0 and hi_b > 0:
            hf_ratio = _r(10 * np.log10(hi_b / lo_b), 2)

    sp_db = rms_db[speech]
    # Noise is measured on non-speech frames only. With under 0.1 s of silence there is nothing reliable
    # to measure, so quality stays "unknown" (acoustics are kept) instead of being misjudged as "poor".
    ns_db = rms_db[~speech]
    noise_level = float(np.median(ns_db)) if ns_db.size >= int(round(0.1 / HOP_S)) else None
    snr = _r(np.mean(sp_db) - noise_level, 1) if sp_db.size and noise_level is not None else None
    if noise_level is not None:
        noise_floor = noise_level
    quality = "unknown"
    if snr is not None:
        quality = "good" if snr >= 25 and clipping_ratio < 0.001 else "fair" if snr >= 12 and clipping_ratio < 0.01 else "poor"

    result = {
        "duration_s": _r(duration, 3),
        "speech_s": _r(speech_s, 3),
        "voiced_ratio": _r(voiced.sum() / max(1, speech.sum()), 3),
        "pitch": pitch,
        "energy": {
            "rms_dbfs_mean": _r(sp_db.mean(), 2) if sp_db.size else None,
            "rms_dbfs_std": _r(sp_db.std(), 2) if sp_db.size else None,
            "dynamic_range_db": _r(np.percentile(sp_db, 95) - np.percentile(sp_db, 5), 2) if sp_db.size else None,
        },
        "voice_quality": {"jitter_local_pct": jitter, "shimmer_local_pct": shimmer, "hnr_db": hnr_db,
                          "pulses_analyzed": int(n_pulses)},
        "spectral": {"centroid_hz": centroid, "hf_energy_ratio_db": hf_ratio},
        "audio_quality": {"snr_db": snr, "clipping_ratio": _r(clipping_ratio, 5),
                          "noise_floor_dbfs": _r(noise_floor, 1), "quality": quality},
        "pauses": {
            "count": int(len(pauses)),
            "total_s": _r(pause_durs.sum(), 3),
            "mean_s": _r(pause_durs.mean(), 3) if pause_durs.size else 0.0,
            "max_s": _r(pause_durs.max(), 3) if pause_durs.size else 0.0,
            "long_count": int((pause_durs >= long_pause_s).sum()),
            "pause_ratio": _r(pause_durs.sum() / span, 3) if span > 0 else 0.0,
            "intervals_s": [[_r(a, 2), _r(b, 2)] for a, b in pauses[:50]],
        },
        "speech_span_s": _r(span, 3),
    }
    if return_tracks:
        result["tracks"] = {
            "t": [round(i * HOP_S + FRAME_S / 2, 4) for i in range(len(f0))],
            "f0_hz": [None if np.isnan(v) else round(float(v), 2) for v in f0],
        }
    return result


def pauses_from_word_timestamps(words: list[dict[str, Any]], *, min_pause_s: float = 0.25,
                                long_pause_s: float = 1.0) -> dict[str, Any]:
    """Pause metrics from STT word timings: [{"word": "hi", "start": 0.1, "end": 0.3}, ...].

    Use this when the agent's ASR (Deepgram, AssemblyAI, Whisper, Google...) gives word
    timestamps but raw audio is not available to the analyser.
    """
    ws = sorted((w for w in words if w.get("start") is not None and w.get("end") is not None),
                key=lambda w: w["start"])
    if not ws:
        return {"count": 0, "total_s": 0.0, "mean_s": 0.0, "max_s": 0.0, "long_count": 0, "pause_ratio": 0.0,
                "intervals_s": []}
    gaps = [(a["end"], b["start"]) for a, b in zip(ws[:-1], ws[1:]) if b["start"] - a["end"] >= min_pause_s]
    d = np.array([b - a for a, b in gaps]) if gaps else np.zeros(0)
    span = ws[-1]["end"] - ws[0]["start"]
    return {
        "count": len(gaps),
        "total_s": _r(d.sum(), 3),
        "mean_s": _r(d.mean(), 3) if d.size else 0.0,
        "max_s": _r(d.max(), 3) if d.size else 0.0,
        "long_count": int((d >= long_pause_s).sum()),
        "pause_ratio": _r(d.sum() / span, 3) if span > 0 else 0.0,
        "intervals_s": [[_r(a, 2), _r(b, 2)] for a, b in gaps[:50]],
        "speech_span_s": _r(span, 3),
        "speech_s": _r(span - d.sum(), 3),
    }
