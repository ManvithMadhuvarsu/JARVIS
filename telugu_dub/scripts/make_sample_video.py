#!/usr/bin/env python3
"""Build a stand-in "talking head" clip from an SRT.

The real POC input is a YouTube video (see scripts/fetch_youtube.sh). This
generator exists because a sandbox without YouTube access still needs a clip
whose mouth moves on known timings, so the timing, mixing and muxing stages can
be verified end to end and the sync drift can be measured against ground truth.

The face is deliberately crude: a head, eyes, and a mouth whose height is driven
by a syllable-rate oscillator that is only enabled inside speech segments.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_dub.config import TtsCfg                      # noqa: E402
from telugu_dub.media import (ffmpeg, write_pcm16, assemble_timeline,  # noqa: E402
                               has_filter)
from telugu_dub.schema import Segment                     # noqa: E402
from telugu_dub.stages import srt as srt_io               # noqa: E402
from telugu_dub.stages.tts import _mock_engine            # noqa: E402

W, H, FPS = 1280, 720, 25


def build_video(cues: list[dict], out: Path, duration: float) -> None:
    mouth_h = "6+16*abs(sin(2*PI*3.2*t))"
    filters = [
        f"color=c=0x101820:s={W}x{H}:d={duration}:r={FPS}[bg]",
        # head, eyes: static geometry
        f"[bg]drawbox=x={W//2-170}:y=140:w=340:h=420:color=0xE8C49A@1:t=fill[head]",
        "[head]drawbox=x=560:y=300:w=40:h=22:color=0x101820@1:t=fill[e1]",
        "[e1]drawbox=x=680:y=300:w=40:h=22:color=0x101820@1:t=fill[e2]",
    ]
    # a closed mouth line, then one animated box per speech cue
    last = "e2"
    filters.append(f"[{last}]drawbox=x=600:y=470:w=80:h=6:color=0x5A2D1A@1:t=fill[m0]")
    last = "m0"
    for i, cue in enumerate(cues):
        tag = f"m{i + 1}"
        filters.append(
            f"[{last}]drawbox=x=600:y='470-({mouth_h})/2':w=80:h='{mouth_h}'"
            f":color=0x5A2D1A@1:t=fill:"
            f"enable='between(t,{cue['start']:.3f},{cue['end']:.3f})'[{tag}]")
        last = tag
    if has_filter("drawtext"):
        filters.append(
            f"[{last}]drawtext=text='POC stand-in speaker':x=(w-text_w)/2:y=620:"
            f"fontsize=28:fontcolor=0x9AA5B1[out]")
    else:                       # static ffmpeg builds often lack freetype
        filters.append(f"[{last}]null[out]")
    ffmpeg(["-filter_complex", ";".join(filters), "-map", "[out]",
            "-t", f"{duration}", "-r", str(FPS), "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", str(out)])


def build_audio(cues: list[dict], out: Path, duration: float, sr: int) -> None:
    """English-side reference audio: mock speech placed on the SRT timings."""
    cfg = TtsCfg(provider="mock", sample_rate=sr, seed=7)
    engine = _mock_engine(cfg)
    tmp = out.parent / "_src_segments"
    tmp.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, cue in enumerate(cues):
        seg = Segment(id=i, start=cue["start"], end=cue["end"], text_tgt=cue["text"])
        # force the segment to exactly fill its slot so the mouth and the
        # reference audio agree — this clip is our ground truth.
        path = tmp / f"src_{i:03d}.wav"
        engine(seg, path)
        from telugu_dub.media import time_stretch, duration_of
        factor = duration_of(path) / (cue["end"] - cue["start"])
        fitted = tmp / f"src_fit_{i:03d}.wav"
        time_stretch(path, fitted, factor, sr)
        clips.append((cue["start"], str(fitted)))
    write_pcm16(out, assemble_timeline(clips, duration, sr), sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", default="samples/demo_en.srt")
    ap.add_argument("--out", default="samples/demo_source.mp4")
    ap.add_argument("--sample-rate", type=int, default=24000)
    args = ap.parse_args()

    cues = srt_io.read_srt(args.srt)
    duration = round(max(c["end"] for c in cues) + 2.0, 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    video = out.with_name("_video.mp4")
    audio = out.with_name("_audio.wav")
    build_video(cues, video, duration)
    build_audio(cues, audio, duration, args.sample_rate)
    ffmpeg(["-i", str(video), "-i", str(audio), "-c:v", "copy",
            "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)])
    video.unlink(missing_ok=True)
    audio.unlink(missing_ok=True)
    shutil.rmtree(out.parent / "_src_segments", ignore_errors=True)
    print(f"wrote {out} ({duration}s, {len(cues)} speech segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
