"""Find the best voice-cloning reference clip inside a long recording.

Zero-shot cloning (IndicF5, F5-TTS, XTTS) copies whatever it hears in the
reference: room tone, background music, a cough, an unusually emphatic delivery.
Picking that clip by hand means scrubbing a whole talk; picking it badly means
every dubbed line inherits the flaw. So we score the recording and rank
candidates.

What a good reference clip looks like:

  * 8-15 seconds of continuous speech — long enough to capture timbre and
    prosody, short enough that the model does not start copying the sentence.
    "Continuous" is relative to the speaker: a slow, pause-heavy delivery may
    never reach 80 % speech in an 11-second window, so the density threshold is
    derived from the recording itself rather than fixed.
  * No long silences inside it (the model learns the pause as part of the voice).
  * Consistent loudness, no clipping.
  * Speech, not music or applause. We detect this with the syllabic modulation
    signature: the energy envelope of speech has a strong peak around 3-6 Hz
    (the syllable rate), while music and ambience are far flatter there.
  * Boundaries that sit in a pause, so the clip does not start mid-word.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .media import read_pcm16

FRAME_MS = 25.0
HOP_MS = 10.0


@dataclass
class Candidate:
    start: float
    end: float
    score: float
    speech_fraction: float
    modulation: float          # 3-6 Hz envelope energy share; speech ~ high
    loudness_db: float
    loudness_spread_db: float
    longest_gap: float
    clipping: float
    edge_quietness: float      # how much of a pause the boundaries sit in

    @property
    def sparse(self) -> bool:
        """True when the clip is more pause than voice — usable, but check it."""
        return self.speech_fraction < 0.55

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# ----------------------------------------------------------------- features
def _frames(samples: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    n = int(FRAME_MS * sr / 1000)
    hop = int(HOP_MS * sr / 1000)
    count = max(0, 1 + (len(samples) - n) // hop)
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return samples[idx], hop


def analyse(wav_path: str) -> dict:
    samples16, sr = read_pcm16(wav_path)
    x = np.asarray(samples16, dtype=np.float32) / 32768.0
    frames, hop = _frames(x, sr)
    if len(frames) == 0:
        raise ValueError(f"{wav_path}: too short to analyse")

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)

    # Adaptive speech floor. The quiet tail of the distribution is the noise
    # floor and speech sits above it — but the gap between them varies hugely
    # (a broadcast-compressed recording, or one with music under the whole
    # thing, has almost no dynamic range). So the threshold is placed
    # proportionally inside the observed span and clamped to stay strictly
    # between floor and peak; a fixed offset can otherwise land above the
    # loudest frame and mark the entire file as silence.
    floor = float(np.percentile(db, 20))
    peak = float(np.percentile(db, 95))
    span = peak - floor
    if span > 12.0:
        threshold = floor + float(np.clip(0.35 * span, 6.0, span - 6.0))
    else:
        threshold = floor + span * 0.5
    speech = db > threshold

    clipped = np.mean(np.abs(frames) > 0.985, axis=1)
    env_rate = 1000.0 / HOP_MS                      # envelope sample rate (Hz)
    return {"db": db, "speech": speech, "clipped": clipped, "rms": rms,
            "env_rate": env_rate, "hop_seconds": hop / sr, "sr": sr,
            "threshold": float(threshold), "duration": len(x) / sr}


def modulation_score(rms_window: np.ndarray, env_rate: float) -> float:
    """Share of envelope energy in the 3-6 Hz syllabic band.

    Speech modulates its own loudness at roughly the syllable rate. Music,
    applause and steady ambience do not, which makes this a cheap and
    surprisingly reliable speech detector with no model behind it.
    """
    env = rms_window - rms_window.mean()
    if len(env) < 16 or not np.any(env):
        return 0.0
    window = np.hanning(len(env))
    spectrum = np.abs(np.fft.rfft(env * window)) ** 2
    freqs = np.fft.rfftfreq(len(env), d=1.0 / env_rate)
    band = (freqs >= 3.0) & (freqs <= 6.0)
    useful = (freqs >= 0.5) & (freqs <= 20.0)
    total = spectrum[useful].sum()
    return float(spectrum[band].sum() / total) if total > 0 else 0.0


# ---------------------------------------------------------------- searching
def find_candidates(wav_path: str, clip_seconds: float = 11.0,
                    hop_seconds: float = 0.5, top_k: int = 5,
                    min_separation: float = 20.0,
                    max_internal_gap: float = 0.7) -> list[Candidate]:
    """Rank every window, then keep the best well-separated ones.

    Density is judged against this speaker, not an absolute: we take the
    windows in the top of the observed distribution. The gap limit stays
    absolute, because a long silence inside the reference is a real defect
    whoever is speaking.
    """
    a = analyse(wav_path)
    db, speech, clipped, rms = a["db"], a["speech"], a["clipped"], a["rms"]
    hop_s = a["hop_seconds"]
    frames_per_clip = int(clip_seconds / hop_s)
    step = max(1, int(hop_seconds / hop_s))
    edge = int(0.25 / hop_s)                        # 250 ms boundary inspection

    # --- pass 1: measure every window -----------------------------------
    windows = []
    for start_i in range(0, max(1, len(db) - frames_per_clip), step):
        sl = slice(start_i, start_i + frames_per_clip)
        w_speech, w_db, w_rms = speech[sl], db[sl], rms[sl]
        if len(w_db) < frames_per_clip:
            break

        gap = best_gap = 0
        for is_speech in w_speech:
            gap = 0 if is_speech else gap + 1
            best_gap = max(best_gap, gap)
        windows.append((start_i, sl, w_speech, w_db, w_rms,
                        float(w_speech.mean()), best_gap * hop_s))

    if not windows:
        return []

    # --- adaptive density gate -------------------------------------------
    # Purely relative: keep the densest windows *this recording* offers. An
    # absolute floor here is a trap — it reports "no clean window" on a sparse
    # or heavily-paused recording whose best window is perfectly usable. The
    # absolute quality constraints (internal gap, clipping) do the real
    # filtering; scoring does the ranking; `sparse` warns the caller.
    fracs = np.array([w[5] for w in windows])
    density_cut = float(min(0.80, np.percentile(fracs, 85)))

    # --- pass 2: score what survives -------------------------------------
    results: list[Candidate] = []
    for start_i, sl, w_speech, w_db, w_rms, speech_frac, longest_gap in windows:
        if speech_frac < density_cut or longest_gap > max_internal_gap:
            continue

        clip_frac = float(clipped[sl].mean())
        loud = float(np.mean(w_db[w_speech])) if w_speech.any() else -99.0
        spread = float(np.std(w_db[w_speech])) if w_speech.any() else 99.0
        mod = modulation_score(w_rms, a["env_rate"])

        # boundaries should sit in a pause, not mid-word
        pre = db[max(0, start_i - edge):start_i]
        post = db[start_i + frames_per_clip:start_i + frames_per_clip + edge]
        quiet = 0.0
        for seg in (pre, post):
            if len(seg):
                quiet += float(np.clip((loud - seg.mean()) / 20.0, 0, 1)) / 2

        score = (2.2 * speech_frac
                 + 2.0 * min(mod / 0.35, 1.0)          # syllabic signature
                 + 1.0 * quiet                          # clean cut points
                 - 1.4 * min(spread / 9.0, 1.5)         # loudness consistency
                 - 3.0 * min(clip_frac * 40, 1.0)       # clipping is fatal
                 - 0.8 * min(longest_gap / 0.7, 1.0))

        results.append(Candidate(
            start=round(start_i * hop_s, 2),
            end=round((start_i + frames_per_clip) * hop_s, 2),
            score=round(score, 4), speech_fraction=speech_frac,
            modulation=mod, loudness_db=loud, loudness_spread_db=spread,
            longest_gap=longest_gap, clipping=clip_frac,
            edge_quietness=quiet))

    results.sort(key=lambda c: c.score, reverse=True)
    chosen: list[Candidate] = []
    for cand in results:
        if all(abs(cand.start - c.start) >= min_separation for c in chosen):
            chosen.append(cand)
        if len(chosen) == top_k:
            break
    return chosen
