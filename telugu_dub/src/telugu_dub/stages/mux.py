"""Stage 8 — final container: picture + dubbed audio + optional subtitles."""
from __future__ import annotations

from pathlib import Path

from ..config import OutputCfg
from ..media import ffmpeg, find_font, has_filter
from ..schema import Segment
from . import srt as srt_io


def write_subtitles(segments: list[Segment], path: str | Path,
                    use_fitted: bool = True) -> str:
    cues = []
    for seg in segments:
        if not seg.text_tgt.strip():
            continue
        start = seg.fit_start if (use_fitted and seg.fit_start is not None) else seg.start
        dur = seg.fit_duration if (use_fitted and seg.fit_duration) else seg.source_duration
        cues.append({"start": start, "end": start + dur, "text": seg.text_tgt})
    return srt_io.write_srt(cues, path)


def export_audio(audio: str, out_path: str, cfg: OutputCfg) -> str:
    """Audio-only deliverable (mode=audio): mp3 by default, wav if asked."""
    if out_path.endswith(".wav"):
        ffmpeg(["-i", audio, "-c:a", "pcm_s16le", out_path])
    else:
        ffmpeg(["-i", audio, "-c:a", "libmp3lame", "-b:a", cfg.audio_bitrate,
                out_path])
    return out_path


def finalize(video: str, audio: str, out_path: str, cfg: OutputCfg,
             subtitles: str | None = None) -> str:
    args = ["-i", video, "-i", audio]
    burn = bool(subtitles and cfg.burn_subtitles)
    if burn and not has_filter("subtitles"):
        print("[dub] warning: ffmpeg has no 'subtitles' filter (libass); "
              "writing a sidecar .srt instead of burning it in")
        burn = False
    if burn and not find_font("telugu"):
        print("[dub] warning: no Telugu font installed (try fonts-noto-telugu); "
              "burned-in text would render as boxes — writing a sidecar .srt")
        burn = False
    if burn:
        # Telugu needs a font with Telugu coverage (Noto Sans Telugu).
        style = ("FontName=Noto Sans Telugu,FontSize=20,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H80000000,BorderStyle=3,Outline=1,MarginV=40")
        args += ["-vf", f"subtitles={subtitles}:force_style='{style}'",
                 "-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset]
    else:
        args += ["-c:v", "copy"]
    args += ["-map", "0:v:0", "-map", "1:a:0",
             "-c:a", "aac", "-b:a", cfg.audio_bitrate, "-shortest", out_path]
    ffmpeg(args)
    return out_path
