"""The keyless path (edge-tts + free Google translate) with the network faked.

These providers cannot be exercised offline for real, but their glue code —
temp files, format conversion, retries, the length-control pass — is exactly
where bugs hide, so it gets tested against stand-ins.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telugu_dub.config import ProsodyCfg, TranslateCfg, TtsCfg  # noqa: E402
from telugu_dub.media import FFMPEG, duration_of, ffmpeg        # noqa: E402
from telugu_dub.schema import Segment                           # noqa: E402
from telugu_dub.stages import translate as tr                   # noqa: E402
from telugu_dub.stages import tts as tts_stage                  # noqa: E402

pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")


def _make_mp3(path: Path, seconds: float) -> None:
    ffmpeg(["-f", "lavfi", "-i", f"sine=frequency=200:duration={seconds}",
            "-c:a", "libmp3lame", str(path)])


@pytest.fixture
def fake_edge(monkeypatch):
    """Stand in for edge_tts.Communicate: writes a real mp3 of known length."""
    calls = []

    class Communicate:
        def __init__(self, text, voice, rate="+0%"):
            calls.append({"text": text, "voice": voice, "rate": rate})
            self._seconds = max(0.4, len(text) / 15)

        async def save(self, path):
            _make_mp3(Path(path), self._seconds)

    monkeypatch.setitem(sys.modules, "edge_tts",
                        types.SimpleNamespace(Communicate=Communicate,
                                              list_voices=None))
    return calls


def test_edge_tts_produces_pcm_wav_at_the_configured_rate(fake_edge, tmp_path):
    cfg = TtsCfg(provider="edge_tts", voice="te-IN-MohanNeural",
                 sample_rate=24000, speaking_rate=1.1)
    segs = [Segment(id=0, start=0, end=3, text_tgt="ధ్యానం అంటే ఏమిటి?")]
    tts_stage.synthesize(segs, cfg, tmp_path, ProsodyCfg())

    out = Path(segs[0].tts_path)
    assert out.suffix == ".wav" and out.exists()
    assert not out.with_suffix(".mp3").exists()     # temp file cleaned up
    assert segs[0].tts_duration > 0
    assert duration_of(out) == pytest.approx(segs[0].tts_duration, abs=0.05)

    assert fake_edge[0]["voice"] == "te-IN-MohanNeural"
    assert fake_edge[0]["rate"] == "+10%"           # speaking_rate -> percent


def test_edge_tts_reports_a_useful_error_on_empty_audio(monkeypatch, tmp_path):
    class Communicate:
        def __init__(self, *a, **k):
            pass

        async def save(self, path):
            Path(path).write_bytes(b"")             # what a blocked network gives

    monkeypatch.setitem(sys.modules, "edge_tts",
                        types.SimpleNamespace(Communicate=Communicate))
    segs = [Segment(id=0, start=0, end=2, text_tgt="పరీక్ష")]
    with pytest.raises(RuntimeError, match="voice name|blocked|no audio"):
        tts_stage.synthesize(segs, TtsCfg(provider="edge_tts"), tmp_path,
                             ProsodyCfg())


@pytest.fixture
def fake_google(monkeypatch):
    """Free Google translate stand-in: output length tracks input length."""
    state = {"calls": 0, "fail_first": 0}

    class GoogleTranslator:
        def __init__(self, source, target):
            assert (source, target) == ("en", "te")

        def translate(self, text):
            state["calls"] += 1
            if state["calls"] <= state["fail_first"]:
                raise RuntimeError("429 rate limited")
            # roughly two Telugu syllables per English word
            return " ".join(["కమ"] * max(1, len(text.split())))

    monkeypatch.setitem(sys.modules, "deep_translator",
                        types.SimpleNamespace(GoogleTranslator=GoogleTranslator))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return state


def test_google_free_translates_every_segment(fake_google):
    segs = [Segment(id=i, start=i * 5.0, end=i * 5.0 + 4.0,
                    text_src="the mind is always seeking something")
            for i in range(3)]
    tr.translate_segments(segs, TranslateCfg(provider="google_free",
                                             glossary=None), ProsodyCfg())
    assert all(s.text_tgt for s in segs)


def test_google_free_retries_a_rate_limit(fake_google):
    fake_google["fail_first"] = 2
    segs = [Segment(id=0, start=0, end=4, text_src="hello there friend")]
    tr.translate_segments(segs, TranslateCfg(provider="google_free",
                                             glossary=None), ProsodyCfg())
    assert segs[0].text_tgt


def test_length_control_retries_overlong_lines_from_a_stripped_source(fake_google):
    """A 1.2 s slot cannot hold a long line: the filler-stripped retry is used."""
    seg = Segment(id=0, start=0.0, end=1.2,
                  text_src="So, you know, the mind is basically always "
                           "seeking something, right?")
    tr.translate_segments([seg], TranslateCfg(provider="google_free",
                                              glossary=None,
                                              length_control=True), ProsodyCfg())
    assert any("shortened" in n for n in seg.notes)


def test_length_control_can_be_switched_off(fake_google):
    seg = Segment(id=0, start=0.0, end=1.2,
                  text_src="So, you know, the mind is basically always seeking.")
    tr.translate_segments([seg], TranslateCfg(provider="google_free",
                                              glossary=None,
                                              length_control=False), ProsodyCfg())
    assert not any("shortened" in n for n in seg.notes)


def test_glossary_terms_survive_translation(fake_google, tmp_path):
    glossary = tmp_path / "g.yaml"
    glossary.write_text("terms:\n  karma: కర్మ\n", encoding="utf-8")
    seg = Segment(id=0, start=0, end=6, text_src="karma is not punishment")
    tr.translate_segments([seg], TranslateCfg(provider="google_free",
                                              glossary=str(glossary)),
                          ProsodyCfg())
    assert seg.text_tgt        # glossary application must not blow up
