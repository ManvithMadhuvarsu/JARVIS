"""Stage 2 — split the recording into the voice and everything else.

This is what makes "keep my background exactly as it is" true rather than
approximate. There are two ways to keep a background under a dub:

  ducking     turn the original down while the dub speaks. The English voice is
              still in there, quietly, forever.
  separation  split the recording, delete the English voice, keep the rest
              untouched and put the Telugu on top.

Only the second one actually preserves the background, and it pays twice:

  * the background stem is passed through unmodified, so music, room tone and
    ambience survive at full level instead of being attenuated to hide a voice
    that is no longer wanted; and
  * the ASR reads isolated speech, which removes a whole class of transcription
    errors on recordings with music underneath.

Separation runs on the ORIGINAL file, not the 16 kHz analysis copy — the bed
ends up in the final mix, so it has to keep its bandwidth.

Caveat worth knowing before you rely on this: Demucs separates *human voice*
from everything else. Audience laughter, a second speaker, a crowd — those go
into the vocals stem along with the speaker and get discarded with it. For
audience-heavy recordings use `mix.separator: ducking` instead and accept a
faint English bed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class SeparationUnavailable(RuntimeError):
    """Demucs is not installed or could not run."""


def is_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separate(source: str | Path, workdir: str | Path, model: str = "htdemucs",
             device: str | None = None) -> dict[str, str]:
    """Split `source` into vocals and background. Returns both paths.

    Cached: if the stems are already on disk from a previous run we reuse them,
    because this is the slowest non-GPU step in the pipeline.
    """
    source = Path(source)
    workdir = Path(workdir)
    outdir = workdir / "stems"
    cached = _find_stems(outdir)
    if cached:
        return cached

    if not is_available():
        raise SeparationUnavailable(
            "demucs is not installed. Either `pip install demucs`, or set "
            "mix.separator: ducking to keep the original underneath instead.")

    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-n", model, "-o", str(outdir), str(source)]
    if device:
        cmd += ["-d", device]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SeparationUnavailable(
            f"demucs failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")

    stems = _find_stems(outdir)
    if not stems:
        raise SeparationUnavailable(
            f"demucs produced no stems under {outdir}")
    return stems


def _find_stems(outdir: Path) -> dict[str, str] | None:
    if not outdir.exists():
        return None
    vocals = next(outdir.rglob("vocals.wav"), None)
    background = next(outdir.rglob("no_vocals.wav"), None)
    if vocals and background:
        return {"vocals": str(vocals), "background": str(background)}
    return None


def ensure_ffmpeg_stems(stems: dict[str, str], workdir: str | Path,
                        sample_rate: int = 16000) -> str:
    """A 16 kHz mono copy of the vocals, for the ASR stage."""
    from ..media import extract_audio

    dst = Path(workdir) / "vocals_16k.wav"
    extract_audio(stems["vocals"], dst, sr=sample_rate)
    return str(dst)
