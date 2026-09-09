"""Voice/background separation — the stage that makes the background survive."""
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telugu_dub.config import Config                       # noqa: E402
from telugu_dub.pipeline import STAGES                     # noqa: E402
from telugu_dub.stages import separate as sep              # noqa: E402


def make_stems(outdir: Path) -> tuple[Path, Path]:
    d = outdir / "stems" / "htdemucs" / "track"
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocals.wav").write_bytes(b"RIFF----WAVEfmt ")
    (d / "no_vocals.wav").write_bytes(b"RIFF----WAVEfmt ")
    return d / "vocals.wav", d / "no_vocals.wav"


def test_separation_runs_before_transcription():
    """ASR must be able to read the isolated vocals, so order matters."""
    assert STAGES.index("separate") < STAGES.index("asr")
    assert STAGES.index("ingest") < STAGES.index("separate")
    assert STAGES.index("separate") < STAGES.index("mix")


def test_cached_stems_are_reused_without_rerunning_demucs(tmp_path, monkeypatch):
    make_stems(tmp_path)
    called = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: called.append(a) or None)
    out = sep.separate(tmp_path / "src.mp3", tmp_path)
    assert called == [], "demucs must not re-run when stems already exist"
    assert out["vocals"].endswith("vocals.wav")
    assert out["background"].endswith("no_vocals.wav")


def test_missing_demucs_raises_a_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sep, "is_available", lambda: False)
    with pytest.raises(sep.SeparationUnavailable, match="ducking"):
        sep.separate(tmp_path / "src.mp3", tmp_path)


def test_demucs_failure_surfaces_its_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(sep, "is_available", lambda: True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="",
                                              stderr="CUDA out of memory"))
    with pytest.raises(sep.SeparationUnavailable, match="CUDA out of memory"):
        sep.separate(tmp_path / "src.mp3", tmp_path)


def test_separation_uses_the_original_not_the_analysis_copy(tmp_path, monkeypatch):
    """The bed ends up in the final mix, so it must keep its bandwidth."""
    monkeypatch.setattr(sep, "is_available", lambda: True)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        make_stems(tmp_path)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sep.separate(tmp_path / "original.mp3", tmp_path)
    assert str(tmp_path / "original.mp3") in seen["cmd"]
    assert not any("16k" in str(c) for c in seen["cmd"])


def test_pipeline_falls_back_to_ducking_and_says_so(tmp_path, monkeypatch, capsys):
    """A long run should not die because demucs is missing — but the user asked
    for their background preserved, so the downgrade must be stated."""
    from telugu_dub.pipeline import Pipeline

    cfg = Config.load(ROOT / "config/native_telugu.yaml")
    cfg.workdir = str(tmp_path / "run")
    pipe = Pipeline(cfg, verbose=True)
    pipe.manifest.artifacts["source_video"] = str(tmp_path / "in.mp3")
    monkeypatch.setattr(sep, "is_available", lambda: False)

    pipe._stage_separate()
    out = capsys.readouterr().out
    assert "WARNING" in out and "ducking" in out
    assert pipe.cfg.mix.separator == "ducking"
    assert "background" not in pipe.manifest.artifacts


def test_no_separation_requested_is_a_clean_noop(tmp_path, capsys):
    from telugu_dub.pipeline import Pipeline

    cfg = Config.load(ROOT / "config/poc_offline.yaml")
    cfg.workdir = str(tmp_path / "run2")
    pipe = Pipeline(cfg, verbose=True)
    pipe._stage_separate()
    assert "background" not in pipe.manifest.artifacts
    assert "not requested" in capsys.readouterr().out
