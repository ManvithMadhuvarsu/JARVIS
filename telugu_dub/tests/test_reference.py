"""Reference-clip selection, on synthesised audio with known properties."""
import math
import sys
from array import array
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telugu_dub.media import FFMPEG, write_pcm16                 # noqa: E402
from telugu_dub.reference import (analyse, find_candidates,      # noqa: E402
                                  modulation_score)

pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
SR = 16000


def synth(path: Path, plan: list[tuple[str, float]]) -> str:
    """Build a wav from a plan of ('speech'|'music'|'silence', seconds).

    'speech' carries the 3-6 Hz syllabic modulation the detector keys on;
    'music' is steady tones with no such modulation.
    """
    out = array("h")
    t = 0.0
    for kind, seconds in plan:
        for i in range(int(seconds * SR)):
            tt = t + i / SR
            if kind == "silence":
                v = 0.0
            elif kind == "music":
                # steady tones: no syllabic modulation
                v = 0.4 * (math.sin(2 * math.pi * 220 * tt)
                           + 0.6 * math.sin(2 * math.pi * 330 * tt)) / 1.6
            else:
                # 4 Hz syllabic envelope over a voiced-ish carrier. Real speech
                # dips between syllables but keeps voicing energy, so the
                # envelope has a floor rather than reaching digital silence.
                env = 0.3 + 0.7 * 0.5 * (1 - math.cos(2 * math.pi * 4.0 * tt))
                v = 0.6 * env * (math.sin(2 * math.pi * 120 * tt)
                                 + 0.5 * math.sin(2 * math.pi * 240 * tt)) / 1.5
            out.append(int(max(-1.0, min(1.0, v)) * 20000))
        t += seconds
    return write_pcm16(path, out, SR)


def test_speech_scores_higher_than_music_on_modulation(tmp_path):
    import numpy as np

    from telugu_dub.reference import analyse as an
    speech = an(synth(tmp_path / "s.wav", [("speech", 6)]))
    music = an(synth(tmp_path / "m.wav", [("music", 6)]))
    s = modulation_score(np.asarray(speech["rms"]), speech["env_rate"])
    m = modulation_score(np.asarray(music["rms"]), music["env_rate"])
    assert s > m * 2, f"speech {s:.3f} should beat music {m:.3f}"


def test_picks_the_speech_region_not_the_music(tmp_path):
    wav = synth(tmp_path / "mix.wav",
                [("music", 14), ("speech", 14), ("music", 14)])
    best = find_candidates(wav, clip_seconds=8.0, top_k=1)
    assert best, "expected a candidate"
    c = best[0]
    assert 13.0 <= c.start <= 21.0, f"picked {c.start}s, expected the speech run"


def test_rejects_windows_with_a_long_internal_silence(tmp_path):
    wav = synth(tmp_path / "gap.wav",
                [("speech", 5), ("silence", 3), ("speech", 5)])
    assert find_candidates(wav, clip_seconds=8.0, max_internal_gap=0.7) == []


def test_density_gate_adapts_to_a_pause_heavy_speaker(tmp_path):
    """A slow speaker never reaches 80 % density; we must still find a clip."""
    plan = []
    for _ in range(8):
        plan += [("speech", 1.6), ("silence", 0.5)]
    wav = synth(tmp_path / "slow.wav", plan)
    found = find_candidates(wav, clip_seconds=8.0)
    assert found, "adaptive density gate should still yield a candidate"
    assert found[0].speech_fraction < 0.85
    assert found[0].longest_gap <= 0.7


def test_candidates_are_separated_and_ranked(tmp_path):
    plan = [("speech", 12), ("music", 6)] * 4
    found = find_candidates(synth(tmp_path / "many.wav", plan),
                            clip_seconds=8.0, top_k=3, min_separation=15.0)
    assert len(found) >= 2
    assert found == sorted(found, key=lambda c: c.score, reverse=True)
    for a, b in zip(found, found[1:]):
        assert abs(a.start - b.start) >= 15.0


def test_analyse_reports_a_sensible_threshold(tmp_path):
    a = analyse(synth(tmp_path / "a.wav", [("speech", 4), ("silence", 4)]))
    assert 0.2 < float(a["speech"].mean()) < 0.8
    assert a["duration"] == pytest.approx(8.0, abs=0.1)
