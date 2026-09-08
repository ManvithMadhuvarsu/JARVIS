"""Isochrony fitting — the stage that actually makes audio and lips agree.

Two languages never take the same time to say the same thing.  Once the Telugu
is synthesised we know each utterance's natural duration; this module decides,
for every utterance, *where it starts* and *how much we bend its tempo* so that
speech lands on the same mouth movements as the English original.

Priority order (each rule only fires when the one above it cannot cope):

  1. Say it at natural tempo inside the original speech slot.
  2. Stretch/compress within the comfortable band (default 0.90x - 1.15x);
     listeners do not perceive tempo change of this size.
  3. Borrow part of the pause that follows the utterance (people pause between
     sentences; a dub may legitimately eat into it).
  4. Compress hard, up to `hard_max_speedup`, and flag the segment.
  5. Overflow: let it run long and push the following segment later, bounded by
     `max_shift`, so an error cannot cascade down the whole video.

Rule 1 matters most for lip-sync: keeping the *start* anchored to the original
onset is worth far more perceptually than filling the slot exactly, because the
viewer locks onto the moment the mouth opens.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import AlignCfg
from .schema import Segment

MIN_GAP = 0.06          # seconds of breathing room between consecutive dubs


@dataclass
class AlignStats:
    segments: int = 0
    natural: int = 0            # no tempo change needed
    in_band: int = 0            # bent within the comfortable band
    borrowed_pause: int = 0
    hard_compressed: int = 0
    overflowed: int = 0
    max_abs_shift: float = 0.0
    max_speed: float = 1.0
    min_speed: float = 1.0
    mean_abs_tempo_change: float = 0.0
    total_overflow: float = 0.0

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def fit_segments(segments: list[Segment], cfg: AlignCfg,
                 media_duration: float | None = None) -> AlignStats:
    """Assign fit_start / fit_duration / speed to every segment, in place."""
    stats = AlignStats(segments=len(segments))
    cursor = 0.0
    tempo_changes: list[float] = []

    for i, seg in enumerate(segments):
        dur = seg.tts_duration
        if not dur or dur <= 0:
            seg.fit_start, seg.fit_duration, seg.speed = seg.start, 0.0, 1.0
            continue

        # --- where may this segment start? -------------------------------
        start = seg.start
        if cursor > start:                       # previous segment ran long
            shift = min(cursor - start, cfg.max_shift)
            start = seg.start + shift
            if cursor - seg.start > cfg.max_shift:
                seg.notes.append(
                    f"overlaps previous by {cursor - start:.2f}s (shift capped)")
        stats.max_abs_shift = max(stats.max_abs_shift, abs(start - seg.start))

        # --- how much room is there? -------------------------------------
        next_start = (segments[i + 1].start if i + 1 < len(segments)
                      else (media_duration if media_duration else seg.end + 2.0))
        pause = max(0.0, next_start - seg.end)
        base_slot = max(0.05, seg.end - start)
        ext_slot = base_slot + pause * cfg.borrow_gap_fraction

        # --- choose a tempo ----------------------------------------------
        speed = 1.0
        if dur <= base_slot:
            # Fits. Optionally slow down a touch so the mouth is busy for the
            # whole original utterance instead of ending early.
            wanted = dur / base_slot
            speed = max(cfg.max_slowdown, wanted)
            if abs(speed - 1.0) < 1e-3:
                stats.natural += 1
            else:
                stats.in_band += 1
        else:
            need_base = dur / base_slot
            if need_base <= cfg.max_speedup:
                speed = need_base                       # fit the original slot
                stats.in_band += 1
            else:
                need_ext = dur / ext_slot
                if need_ext <= cfg.max_speedup:
                    speed = need_ext                    # borrow the pause
                    stats.borrowed_pause += 1
                elif need_ext <= cfg.hard_max_speedup:
                    speed = need_ext
                    stats.hard_compressed += 1
                    seg.notes.append(f"hard compression {speed:.2f}x")
                else:
                    speed = cfg.hard_max_speedup
                    stats.hard_compressed += 1
                    stats.overflowed += 1
                    seg.overflow = round(dur / speed - ext_slot, 3)
                    stats.total_overflow += seg.overflow
                    seg.notes.append(
                        f"overflow {seg.overflow:.2f}s — shorten the translation")

        seg.speed = round(speed, 4)
        seg.fit_start = round(start, 3)
        seg.fit_duration = round(dur / speed, 3)
        tempo_changes.append(abs(speed - 1.0))
        stats.max_speed = max(stats.max_speed, speed)
        stats.min_speed = min(stats.min_speed, speed)
        cursor = seg.fit_start + seg.fit_duration + MIN_GAP

    stats.mean_abs_tempo_change = (
        sum(tempo_changes) / len(tempo_changes) if tempo_changes else 0.0)
    return stats


def sync_report(segments: list[Segment]) -> list[dict]:
    """Per-segment QC rows: how far the dub drifts from the original onset."""
    rows = []
    for seg in segments:
        if seg.fit_start is None:
            continue
        rows.append({
            "id": seg.id,
            "src": [round(seg.start, 2), round(seg.end, 2)],
            "dub": [round(seg.fit_start, 2),
                    round(seg.fit_start + (seg.fit_duration or 0), 2)],
            "onset_drift": round(seg.fit_start - seg.start, 3),
            "end_drift": round(seg.fit_start + (seg.fit_duration or 0) - seg.end, 3),
            "speed": seg.speed,
            "overflow": seg.overflow,
            "notes": seg.notes,
        })
    return rows
