"""Stage 1 — get the source video onto disk and pull a 16 kHz mono track."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..media import extract_audio, probe
from ..schema import MediaInfo


def download(url: str, outdir: str | Path, max_height: int = 720,
             clip: tuple[float, float] | None = None) -> str:
    """Fetch a video with yt-dlp.

    `clip` is (start_seconds, end_seconds) — always work on a short excerpt
    while iterating; lip-sync on a full 20-minute talk is an overnight job.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp not installed: pip install yt-dlp")
    out_tmpl = str(outdir / "source.%(ext)s")
    cmd = [
        "yt-dlp", "-o", out_tmpl,
        "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]",
        "--merge-output-format", "mp4",
    ]
    if clip:
        start, end = clip
        cmd += ["--download-sections", f"*{start}-{end}", "--force-keyframes-at-cuts"]
    cmd.append(url)
    subprocess.run(cmd, check=True)
    files = sorted(outdir.glob("source.*"))
    if not files:
        raise RuntimeError("yt-dlp produced no output")
    return str(files[0])


def prepare(video_path: str | Path, workdir: str | Path,
            clip: tuple[float, float] | None = None) -> tuple[MediaInfo, str]:
    """Normalise the video (optional trim) and extract the analysis audio."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src = Path(video_path)

    if clip:
        from ..media import ffmpeg
        trimmed = workdir / "source_clip.mp4"
        start, end = clip
        ffmpeg(["-ss", str(start), "-to", str(end), "-i", str(src),
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-c:a", "aac", str(trimmed)])
        src = trimmed

    info = probe(src)
    audio = workdir / "source_16k.wav"
    extract_audio(src, audio, sr=16000)
    return MediaInfo(path=str(src), duration=info["duration"], width=info["width"],
                     height=info["height"], fps=info["fps"],
                     has_video=info["has_video"]), str(audio)
