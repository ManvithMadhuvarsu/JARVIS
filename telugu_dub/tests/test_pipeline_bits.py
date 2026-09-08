import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_dub.cli import apply_overrides           # noqa: E402
from telugu_dub.config import AsrCfg, Config         # noqa: E402
from telugu_dub.media import atempo_chain            # noqa: E402
from telugu_dub.qc import evaluate                   # noqa: E402
from telugu_dub.schema import Manifest, Segment      # noqa: E402
from telugu_dub.stages.asr import build_units        # noqa: E402
from telugu_dub.stages.srt import (format_timestamp,  # noqa: E402
                                   parse_timestamp, read_srt, write_srt)


def test_srt_roundtrip(tmp_path):
    cues = [{"start": 1.5, "end": 4.25, "text": "hello there"},
            {"start": 5.0, "end": 6.0, "text": "మీరు ఎవరు?"}]
    path = tmp_path / "a.srt"
    write_srt(cues, path)
    back = read_srt(path)
    assert [c["text"] for c in back] == [c["text"] for c in cues]
    assert back[0]["start"] == 1.5 and back[0]["end"] == 4.25


def test_timestamp_formatting():
    assert format_timestamp(3661.5).startswith("01:01:01,5")
    assert parse_timestamp("00:00:04,200") == 4.2


def test_atempo_chain_stays_within_ffmpeg_limits():
    for factor in (0.3, 0.5, 1.0, 1.15, 2.0, 3.7):
        chain = atempo_chain(factor)
        product = 1.0
        for part in chain.split(","):
            if part.startswith("atempo="):
                product *= float(part.split("=")[1])
            for value in [float(p.split("=")[1]) for p in chain.split(",")
                          if p.startswith("atempo=")]:
                assert 0.5 <= value <= 2.0
        if chain != "anull":
            assert abs(product - factor) < 1e-3


def test_build_units_merges_within_gap_but_not_across_pauses():
    cfg = AsrCfg(merge_gap_seconds=0.35, min_segment_seconds=0.5,
                 max_segment_seconds=12)
    chunks = [
        {"start": 0.0, "end": 2.0, "text": "the mind is", "speaker": "S0"},
        {"start": 2.2, "end": 4.0, "text": "always seeking", "speaker": "S0"},
        {"start": 9.0, "end": 11.0, "text": "a new sentence", "speaker": "S0"},
    ]
    units = build_units(chunks, cfg)
    assert len(units) == 2
    assert units[0].text_src == "the mind is always seeking"
    assert units[0].end == 4.0
    assert units[1].id == 1


def test_build_units_splits_on_speaker_change():
    cfg = AsrCfg(merge_gap_seconds=1.0, min_segment_seconds=0.1)
    chunks = [{"start": 0.0, "end": 2.0, "text": "question", "speaker": "S0"},
              {"start": 2.1, "end": 4.0, "text": "answer", "speaker": "S1"}]
    assert len(build_units(chunks, cfg)) == 2


def test_qc_flags_early_audio_harder_than_late():
    early = Segment(id=0, start=1.0, end=3.0, tts_duration=2.0,
                    fit_start=0.85, fit_duration=2.0)
    late = Segment(id=1, start=5.0, end=7.0, tts_duration=2.0,
                   fit_start=5.10, fit_duration=2.0)
    result = evaluate([early, late])
    grades = {r["id"]: r["grade"] for r in result.rows}
    assert grades[0] == "fail-early"       # 150 ms early = broken
    assert grades[1] == "in-sync"          # 100 ms late = fine
    assert not result.passed


def test_config_overrides_and_manifest_roundtrip(tmp_path):
    cfg = apply_overrides(Config(), ["tts.provider=mock", "align.max_speedup=1.25",
                                     "output.write_srt=false"])
    assert cfg.tts.provider == "mock"
    assert cfg.align.max_speedup == 1.25
    assert cfg.output.write_srt is False

    m = Manifest(segments=[Segment(id=0, start=0, end=1, text_tgt="ధ్యానం")])
    path = tmp_path / "m.json"
    m.save(path)
    assert Manifest.load(path).segments[0].text_tgt == "ధ్యానం"
