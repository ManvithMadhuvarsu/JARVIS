import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_dub.telugu_prosody import (count_syllables,  # noqa: E402
                                       estimate_speech_duration, is_telugu,
                                       measure_rate, syllable_budget)


def test_detects_script():
    assert is_telugu("ధ్యానం అంటే ఏమిటి?")
    assert not is_telugu("What is meditation?")


def test_telugu_syllable_counting():
    # నమస్కారం = na-mas-kaa-ram -> 4 syllables
    # (5 consonants, 1 virama: 5 - 1 = 4)
    assert count_syllables("నమస్కారం") == 4
    assert count_syllables("ధ్యానం") == 2          # dhyaa-nam, cluster counts once
    assert count_syllables("") == 0


def test_virama_collapses_clusters():
    assert count_syllables("స్వ") == 1             # s + v joined by virama
    assert count_syllables("సవ") == 2


def test_latin_fallback():
    assert count_syllables("meditation") == 4
    assert count_syllables("the") == 1


def test_duration_scales_with_rate_and_punctuation():
    text = "ఈ ఒక్క క్షణం మీద శ్రద్ధ పెడితే, మిగతావన్నీ కుదురుకుంటాయి."
    fast = estimate_speech_duration(text, 8.0)
    slow = estimate_speech_duration(text, 4.0)
    assert slow > fast
    assert estimate_speech_duration(text, 6.5) > \
        estimate_speech_duration(text.replace(",", " "), 6.5)


def test_budget_grows_with_slot():
    assert syllable_budget(4.0, 6.5) > syllable_budget(2.0, 6.5)
    assert syllable_budget(0.01, 6.5) >= 1


def test_measure_rate_is_syllable_weighted():
    out = measure_rate([("నమస్కారం", 1.0), ("ధ్యానం", 1.0)])
    assert out["n"] == 2
    assert out["rate"] == 3.0                       # (4 + 2) syllables / 2 s
