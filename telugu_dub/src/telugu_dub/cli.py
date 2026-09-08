"""Command line entry point:  python -m telugu_dub ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import modes
from .config import Config
from .doctor import blocking, format_checks, run_checks
from .pipeline import STAGES, Pipeline
from .schema import Manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="telugu-dub",
        description="Dub an English talk into Telugu with matching lip movement.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the pipeline")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube (or any yt-dlp) URL")
    src.add_argument("--video", help="local video file")
    src.add_argument("--audio", help="local audio file (implies --mode audio)")
    run.add_argument("--config", default="config/default.yaml")
    run.add_argument("--mode", choices=modes.MODES,
                     help="audio: Telugu audio track only | video: original "
                          "picture + Telugu dub | video-lipsync: also "
                          "regenerate the mouth (needs a GPU)")
    run.add_argument("--voice", choices=modes.VOICES,
                     help="preset: stock Telugu voice (no consent needed) | "
                          "clone: the speaker's own voice (needs consent + a "
                          "reference recording)")
    run.add_argument("--skip-checks", action="store_true",
                     help="run even if the preflight finds blockers")
    run.add_argument("--workdir", help="override config workdir")
    run.add_argument("--clip", help="excerpt as START:END seconds, e.g. 30:60")
    run.add_argument("--until", choices=STAGES, default="mux")
    run.add_argument("--force", nargs="*", default=[], choices=STAGES,
                     help="re-run these stages even if cached")
    run.add_argument("--transcript", help="SRT to use instead of running ASR")
    run.add_argument("--set", nargs="*", default=[], metavar="a.b=c",
                     help="config overrides, e.g. --set tts.provider=mock")

    rep = sub.add_parser("report", help="print the report for a finished run")
    rep.add_argument("workdir")

    doc = sub.add_parser("doctor", help="check this machine can run a mode")
    doc.add_argument("--config", default="config/default.yaml")
    doc.add_argument("--mode", choices=modes.MODES, default="video")
    doc.add_argument("--voice", choices=modes.VOICES, default="preset")
    doc.add_argument("--set", nargs="*", default=[], metavar="a.b=c")

    voi = sub.add_parser("voices", help="list free Telugu edge-tts voices")
    voi.add_argument("--locale", default="te")

    mod = sub.add_parser("modes", help="explain the use-case matrix")
    sub.add_parser("stages", help="list pipeline stages")
    return p


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    raw = cfg.to_dict()
    for item in overrides:
        key, _, value = item.partition("=")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return Config.from_dict(raw)


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":
        return None
    # NB: "none" is NOT coerced to None — it is a real provider name in this
    # schema (lipsync.provider: none, mix.separator: none).
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "stages":
        print("\n".join(STAGES))
        return 0

    if args.cmd == "modes":
        for name, spec in modes.MODE_SPECS.items():
            print(f"--mode {name}")
            print(f"    {spec.summary}")
            print(f"    stages: {' -> '.join(spec.stages)}")
            print(f"    output: .{spec.container}\n")
        for name, preset in modes.VOICE_PRESETS.items():
            print(f"--voice {name:8s} default TTS: {preset['provider']}")
        return 0

    if args.cmd == "voices":
        from .stages.tts import list_edge_voices
        try:
            found = list_edge_voices(args.locale)
        except Exception as exc:
            print(f"could not reach the edge-tts voice list: {exc}")
            print("Known Telugu voices: te-IN-MohanNeural (male), "
                  "te-IN-ShrutiNeural (female)")
            return 1
        for v in found:
            print(f"{v['name']:24s} {v['gender']:7s} {v['locale']}")
        return 0

    if args.cmd == "doctor":
        cfg = Config.load(args.config if Path(args.config).exists() else None)
        cfg = modes.apply(cfg, args.mode, args.voice)
        cfg = apply_overrides(cfg, args.set)
        checks = run_checks(cfg)
        print(format_checks(checks, cfg))
        return 1 if blocking(checks) else 0

    if args.cmd == "report":
        m = Manifest.load(Path(args.workdir) / "manifest.json")
        print(json.dumps(m.stats, indent=2, ensure_ascii=False))
        return 0

    cfg = Config.load(args.config if Path(args.config).exists() else None)
    if args.workdir:
        cfg.workdir = args.workdir
    if args.transcript:
        cfg.asr.provider = "srt"
        cfg.asr.transcript = args.transcript
    if args.audio and not args.mode:
        args.mode = "audio"          # an audio source cannot produce video
    cfg = modes.apply(cfg, args.mode, args.voice)
    cfg = apply_overrides(cfg, args.set)

    checks = run_checks(cfg)
    if (problems := blocking(checks)):
        print(format_checks(checks, cfg))
        if not args.skip_checks:
            print("\nFix the blockers above, or pass --skip-checks to try anyway.")
            return 1

    clip = None
    if args.clip:
        start, _, end = args.clip.partition(":")
        clip = (float(start), float(end))

    pipe = Pipeline(cfg)
    pipe.run(source=args.video or args.audio or "", url=args.url, clip=clip,
             until=args.until, force=args.force)
    print(Path(cfg.workdir, "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
