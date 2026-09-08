"""The six use cases: mode x voice."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telugu_dub import modes                            # noqa: E402
from telugu_dub.cli import _coerce, apply_overrides     # noqa: E402
from telugu_dub.config import Config                    # noqa: E402
from telugu_dub.doctor import blocking, run_checks      # noqa: E402
from telugu_dub.stages.translate import shorten_source  # noqa: E402


def base() -> Config:
    return Config.load(ROOT / "config/default.yaml")


@pytest.mark.parametrize("mode", modes.MODES)
@pytest.mark.parametrize("voice", modes.VOICES)
def test_every_combination_is_coherent(mode, voice):
    cfg = modes.apply(base(), mode, voice)
    spec = modes.MODE_SPECS[mode]
    assert cfg.mode == mode and cfg.voice == voice
    assert cfg.output.container == spec.container
    assert ("lipsync" in spec.stages) == (mode == "video-lipsync")
    if mode != "video-lipsync":
        assert cfg.lipsync.provider == "none"
    else:
        assert cfg.lipsync.provider != "none"
    assert (cfg.tts.provider in {"indicf5", "elevenlabs"}) == (voice == "clone")


def test_preset_voice_clears_cloning_reference():
    cfg = base()
    cfg.tts.ref_audio = "samples/ref.wav"
    cfg.tts.ref_text = "something"
    cfg = modes.apply(cfg, "video", "preset")
    assert cfg.tts.ref_audio is None and cfg.tts.ref_text is None


def test_explicit_hosted_tts_survives_preset_selection():
    """A user who asked for Sarvam should not be silently switched to edge-tts."""
    cfg = base()
    cfg.tts.provider = "sarvam"
    assert modes.apply(cfg, "audio", "preset").tts.provider == "sarvam"


def test_audio_mode_skips_lipsync_stage():
    cfg = modes.apply(base(), "audio", "preset")
    assert "lipsync" not in modes.stages_for(cfg)
    assert modes.stages_for(cfg)[-1] == "mux"


def test_bad_mode_or_voice_is_rejected():
    with pytest.raises(ValueError):
        modes.apply(base(), "video-4k", None)
    with pytest.raises(ValueError):
        modes.apply(base(), None, "sadhguru")


def test_none_is_a_provider_name_not_a_null():
    assert _coerce("none") == "none"
    assert _coerce("null") is None
    assert _coerce("true") is True and _coerce("1.5") == 1.5
    cfg = apply_overrides(base(), ["lipsync.provider=none"])
    assert cfg.lipsync.provider == "none"


def test_doctor_blocks_lipsync_without_a_gpu_but_not_video_mode():
    audio = run_checks(modes.apply(base(), "audio", "preset"))
    assert not any("lipsync" in c.name for c in audio)

    lip = modes.apply(base(), "video-lipsync", "preset")
    names = [c.name for c in run_checks(lip)]
    assert any("lipsync" in n for n in names)
    assert any("gpu" in n for n in names)


def test_doctor_passes_the_offline_preset():
    cfg = Config.load(ROOT / "config/poc_offline.yaml")
    cfg.asr.transcript = str(ROOT / "samples/demo_en.srt")
    cfg = modes.apply(cfg, "video", None)
    assert blocking(run_checks(cfg)) == []


def test_filler_stripping_shortens_without_losing_content():
    out = shorten_source("So, you know, the mind is basically always seeking, right?")
    assert "mind" in out and "seeking" in out
    assert "you know" not in out and "basically" not in out
    assert len(out) < 40
    assert not out.rstrip().endswith(",")


def test_filler_stripping_leaves_clean_text_alone():
    text = "The more aware you become, the less you suffer."
    assert shorten_source(text) == text


def test_filler_stripping_never_returns_empty():
    assert shorten_source("So, actually, you know?").strip()
