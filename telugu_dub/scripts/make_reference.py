#!/usr/bin/env python3
"""Pick voice-cloning reference clips out of a long recording.

    python scripts/make_reference.py --audio talk.mp3 --out samples/ref

Writes the top candidates as 24 kHz mono wavs plus a report, so you can listen
to three and pick one rather than scrubbing an hour of audio. The chosen clip
goes in the config as `tts.ref_audio`, with its exact transcript as
`tts.ref_text` — zero-shot cloning needs both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_dub.media import extract_audio, ffmpeg          # noqa: E402
from telugu_dub.reference import find_candidates            # noqa: E402


def timecode(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="source audio or video file")
    ap.add_argument("--out", default="samples/ref", help="output directory")
    ap.add_argument("--seconds", type=float, default=11.0,
                    help="clip length (8-15 s is the useful range)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--sample-rate", type=int, default=24000,
                    help="24000 for IndicF5 / F5-TTS")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    analysis_wav = out / "_analysis_16k.wav"
    extract_audio(args.audio, analysis_wav, sr=16000)

    cands = find_candidates(str(analysis_wav), clip_seconds=args.seconds,
                            top_k=args.top)
    if not cands:
        print("No clean speech window found. Try a shorter --seconds, or a "
              "recording with less music under the voice.")
        return 1

    rows = []
    print(f"\n{len(cands)} candidate reference clips "
          f"({args.seconds:.0f}s each), best first:\n")
    print(f"{'#':<3}{'start':<10}{'end':<10}{'score':<8}{'speech':<9}"
          f"{'syllabic':<10}{'level':<9}{'steady':<8}")
    for i, c in enumerate(cands, 1):
        dst = out / f"ref_{i:02d}.wav"
        ffmpeg(["-ss", str(c.start), "-t", str(args.seconds), "-i", args.audio,
                "-ac", "1", "-ar", str(args.sample_rate),
                "-af", "afade=t=in:d=0.02,afade=t=out:st="
                       f"{args.seconds - 0.02:.2f}:d=0.02,"
                       "loudnorm=I=-20:TP=-2:LRA=7",
                "-c:a", "pcm_s16le", str(dst)])
        print(f"{i:<3}{timecode(c.start):<10}{timecode(c.end):<10}"
              f"{c.score:<8.2f}{c.speech_fraction * 100:<8.0f}%"
              f" {c.modulation:<9.3f}{c.loudness_db:<9.1f}"
              f"{c.loudness_spread_db:<8.1f}"
              f"{'  sparse' if c.sparse else ''}")
        rows.append({"file": str(dst), **c.as_dict()})

    if all(c.sparse for c in cands):
        print("\nNote: every candidate is more pause than voice. That is normal "
              "for slow speakers, but listen before using one — a reference "
              "full of silence teaches the model to pause like that.")

    (out / "candidates.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    analysis_wav.unlink(missing_ok=True)

    print(f"""
Wrote {len(rows)} clips to {out}/  (report: {out}/candidates.json)

Next:
  1. Listen to ref_01..ref_0{len(rows)}. Pick the one that sounds most like the
     speaker's ordinary register — not the most dramatic sentence. Clean, calm,
     no music, no laughter.
  2. Transcribe it EXACTLY (every word, including false starts):
       python -m telugu_dub run --audio {out}/ref_01.wav \\
           --config config/free_cpu.yaml --until asr --workdir runs/ref
       cat runs/ref/manifest.json   # copy the text_src values
  3. Put both in your config:
       tts:
         provider: indicf5
         ref_audio: {out}/ref_01.wav
         ref_text: "the exact transcript"

Columns: speech = share of the clip that is voice, syllabic = 3-6 Hz envelope
energy (speech scores high, music and applause low), steady = loudness spread in
dB (lower is better).""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
