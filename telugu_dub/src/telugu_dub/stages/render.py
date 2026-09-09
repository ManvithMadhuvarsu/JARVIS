"""Stage 6 — turn fitted segments into one continuous dubbed audio track.

  1. time-stretch each segment by the factor the aligner chose (ffmpeg atempo,
     pitch preserving),
  2. drop each one at its fitted start on a silent bed,
  3. optionally keep the original background (music, ambience, applause) under
     the dub,
  4. normalise loudness to a broadcast-ish target.
"""
from __future__ import annotations

from pathlib import Path

from ..config import MixCfg
from ..media import (assemble_timeline, ffmpeg, time_stretch, to_pcm16,
                     write_pcm16)
from ..schema import Segment


def render_dub_track(segments: list[Segment], workdir: str | Path,
                     total_seconds: float, sample_rate: int) -> str:
    workdir = Path(workdir)
    stretch_dir = workdir / "stretched"
    stretch_dir.mkdir(parents=True, exist_ok=True)

    clips: list[tuple[float, str]] = []
    for seg in segments:
        if not seg.tts_path or seg.fit_start is None:
            continue
        dst = stretch_dir / f"seg_{seg.id:04d}.wav"
        if abs(seg.speed - 1.0) < 1e-3:
            to_pcm16(seg.tts_path, dst, sample_rate)
        else:
            time_stretch(seg.tts_path, dst, seg.speed, sample_rate)
        clips.append((seg.fit_start, str(dst)))

    track = assemble_timeline(clips, total_seconds, sample_rate)
    return write_pcm16(workdir / "dub_track.wav", track, sample_rate)


def mix_with_background(dub_wav: str, bed_audio: str, out_wav: str,
                        cfg: MixCfg, sample_rate: int) -> str:
    """Put the dub over the bed.

    `bed_audio` is either the separated background (already voice-free, so it
    is mixed in at full level) or the untouched original (in which case it is
    side-chain ducked so the English voice sits under the dub rather than
    fighting it). The separate stage decides which; this function only mixes.
    """
    if not cfg.keep_background or cfg.separator == "none":
        ffmpeg(["-i", dub_wav, "-af",
                f"loudnorm=I={cfg.loudness_target_lufs}:TP=-1.5:LRA=11",
                "-ar", str(sample_rate), "-c:a", "pcm_s16le", out_wav])
        return out_wav

    if cfg.separator == "demucs":
        # Voice-free bed: no ducking needed, mix it straight in.
        ffmpeg([
            "-i", bed_audio, "-i", dub_wav, "-filter_complex",
            (f"[0:a]aresample={sample_rate},volume={cfg.background_gain_db}dB[bg];"
             f"[1:a]aresample={sample_rate},volume={cfg.dub_gain_db}dB[dub];"
             f"[bg][dub]amix=inputs=2:duration=longest:dropout_transition=0,"
             f"loudnorm=I={cfg.loudness_target_lufs}:TP=-1.5:LRA=11[out]"),
            "-map", "[out]", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            out_wav])
        return out_wav

    ffmpeg([
        "-i", bed_audio, "-i", dub_wav,
        "-filter_complex",
        (f"[0:a]aresample={sample_rate},volume={cfg.background_gain_db}dB[bg];"
         f"[1:a]aresample={sample_rate},volume={cfg.dub_gain_db}dB,asplit=2[dub][key];"
         f"[bg][key]sidechaincompress=threshold=0.03:ratio=12:attack=15:"
         f"release=350:makeup=1[ducked];"
         f"[ducked][dub]amix=inputs=2:duration=longest:dropout_transition=0,"
         f"loudnorm=I={cfg.loudness_target_lufs}:TP=-1.5:LRA=11[out]"),
        "-map", "[out]", "-ar", str(sample_rate), "-c:a", "pcm_s16le", out_wav])
    return out_wav


