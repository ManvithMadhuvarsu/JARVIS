"""The aligner is where sync is won or lost, so it gets the most tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_dub.align import fit_segments          # noqa: E402
from telugu_dub.config import AlignCfg             # noqa: E402
from telugu_dub.schema import Segment              # noqa: E402


def seg(i, start, end, tts):
    return Segment(id=i, start=start, end=end, tts_duration=tts)


def test_natural_fit_keeps_tempo_and_onset():
    s = seg(0, 1.0, 4.0, 3.0)
    fit_segments([s], AlignCfg(), 10)
    assert s.fit_start == 1.0
    assert s.speed == pytest.approx(1.0, abs=1e-3)
    assert s.overflow == 0


def test_short_dub_is_slowed_only_to_the_band_floor():
    # 2 s of speech in a 4 s slot: stretch to 0.90x, never further
    s = seg(0, 0.0, 4.0, 2.0)
    fit_segments([s], AlignCfg(max_slowdown=0.90), 10)
    assert s.speed == pytest.approx(0.90, abs=1e-3)
    assert s.fit_duration == pytest.approx(2 / 0.9, abs=0.01)
    assert s.fit_start == 0.0                      # onset stays anchored


def test_mild_overrun_is_compressed_inside_the_original_slot():
    s = seg(0, 0.0, 4.0, 4.4)                      # 10 % long
    fit_segments([s], AlignCfg(), 10)
    assert s.speed == pytest.approx(1.1, abs=1e-3)
    assert s.fit_duration == pytest.approx(4.0, abs=0.01)


def test_pause_is_borrowed_before_hard_compression():
    # 5.0 s of dub in a 4 s slot, but a 2 s pause follows
    a, b = seg(0, 0.0, 4.0, 5.0), seg(1, 6.0, 8.0, 1.5)
    cfg = AlignCfg(max_speedup=1.15, borrow_gap_fraction=0.8)
    stats = fit_segments([a, b], cfg, 12)
    assert stats.borrowed_pause == 1
    assert a.speed <= cfg.max_speedup + 1e-6
    assert a.fit_duration > 4.0                    # it ran into the pause
    assert a.overflow == 0


def test_overflow_is_capped_and_reported():
    s = seg(0, 0.0, 2.0, 6.0)                      # 3x too long, no pause after
    cfg = AlignCfg(hard_max_speedup=1.3)
    stats = fit_segments([s], cfg, 2.0)
    assert s.speed == pytest.approx(1.3, abs=1e-3)
    assert s.overflow > 0
    assert stats.overflowed == 1
    assert any("overflow" in n for n in s.notes)


def test_cascade_shift_is_bounded():
    """One overlong segment must not push the whole video out of sync."""
    segs = [seg(0, 0.0, 2.0, 8.0)] + [seg(i, 2.0 * i, 2.0 * i + 1.5, 1.5)
                                      for i in range(1, 6)]
    cfg = AlignCfg(max_shift=0.30)
    stats = fit_segments(segs, cfg, 20)
    assert stats.max_abs_shift <= cfg.max_shift + 1e-6
    for s in segs[2:]:
        assert abs(s.fit_start - s.start) <= cfg.max_shift + 1e-6


def test_never_starts_before_the_original_onset():
    """Audio leading picture is the perceptually fatal direction."""
    segs = [seg(i, 3.0 * i, 3.0 * i + 2.0, 2.5) for i in range(8)]
    fit_segments(segs, AlignCfg(), 30)
    for s in segs:
        assert s.fit_start >= s.start - 1e-9


def test_empty_and_zero_duration_segments_are_safe():
    segs = [seg(0, 0.0, 2.0, 0.0), seg(1, 3.0, 5.0, 2.0)]
    fit_segments(segs, AlignCfg(), 8)
    assert segs[0].fit_duration == 0.0
    assert segs[1].fit_start == 3.0
