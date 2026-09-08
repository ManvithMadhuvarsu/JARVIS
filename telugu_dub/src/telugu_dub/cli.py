"""Command line entry point:  python -m telugu_dub ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
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
    run.add_argument("--config", default="config/default.yaml")
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
    if low in ("none", "null"):
        return None
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
    cfg = apply_overrides(cfg, args.set)

    clip = None
    if args.clip:
        start, _, end = args.clip.partition(":")
        clip = (float(start), float(end))

    pipe = Pipeline(cfg)
    pipe.run(source=args.video or "", url=args.url, clip=clip,
             until=args.until, force=args.force)
    print(Path(cfg.workdir, "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
