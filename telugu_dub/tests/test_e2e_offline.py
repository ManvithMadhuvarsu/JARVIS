"""End-to-end smoke test: builds a clip, dubs it, checks the artefacts.

Uses the offline provider set (srt + mock translation + mock TTS), so it needs
ffmpeg but no GPU, no network and no API keys. Runs in a few seconds.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telugu_dub.config import Config          # noqa: E402
from telugu_dub.pipeline import Pipeline      # noqa: E402

try:
    from telugu_dub.media import FFMPEG       # noqa: E402
    HAVE_FFMPEG = bool(FFMPEG)
except Exception:                             # pragma: no cover
    HAVE_FFMPEG = False

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    out = tmp_path_factory.mktemp("sample") / "demo.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_sample_video.py"),
                    "--srt", str(ROOT / "samples/demo_en.srt"), "--out", str(out)],
                   check=True, cwd=ROOT,
                   env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    return out


def test_full_offline_run(sample, tmp_path):
    cfg = Config.load(ROOT / "config/poc_offline.yaml")
    cfg.workdir = str(tmp_path / "run")
    cfg.asr.transcript = str(ROOT / "samples/demo_en.srt")
    cfg.translate.glossary = str(ROOT / "config/glossary_te.yaml")

    manifest = Pipeline(cfg, verbose=False).run(source=str(sample))

    assert len(manifest.segments) == 6
    assert all(s.text_tgt for s in manifest.segments)
    assert all(s.fit_start is not None for s in manifest.segments)

    final = Path(manifest.artifacts["final"])
    assert final.exists() and final.stat().st_size > 10_000

    from telugu_dub.media import duration_of
    assert duration_of(final) == pytest.approx(manifest.video.duration, abs=1.0)

    # every dub starts at or after the original onset (never audio-leads)
    for seg in manifest.segments:
        assert seg.fit_start >= seg.start - 1e-6

    assert Path(manifest.artifacts["subtitles"]).read_text(encoding="utf-8").strip()
    assert (Path(cfg.workdir) / "qc.json").exists()
    assert (Path(cfg.workdir) / "report.md").exists()


def test_run_is_resumable(sample, tmp_path):
    cfg = Config.load(ROOT / "config/poc_offline.yaml")
    cfg.workdir = str(tmp_path / "resume")
    cfg.asr.transcript = str(ROOT / "samples/demo_en.srt")
    cfg.translate.glossary = str(ROOT / "config/glossary_te.yaml")

    Pipeline(cfg, verbose=False).run(source=str(sample), until="tts")
    second = Pipeline(cfg, verbose=False)
    assert "tts" in second.manifest.stages_done
    manifest = second.run(source=str(sample))
    assert Path(manifest.artifacts["final"]).exists()
