"""Preflight: tell the user exactly what is missing for the mode they picked.

The most common way a first run fails is not a bug — it is a missing binary, a
missing key, or a GPU that isn't there. This checks all of it in two seconds and
prints the install command for whatever is absent.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass

from .config import Config
from .modes import MODE_SPECS

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_cuda() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            return True, f"{torch.cuda.get_device_name(0)} ({gb:.0f} GB)"
        return False, "torch installed, no CUDA device"
    except Exception:
        return False, "torch not installed"


def run_checks(cfg: Config) -> list[Check]:
    checks: list[Check] = []
    spec = MODE_SPECS[cfg.mode]

    # ---- media tooling ---------------------------------------------------
    if shutil.which("ffmpeg"):
        checks.append(Check("ffmpeg", OK, shutil.which("ffmpeg")))
    elif _module("imageio_ffmpeg"):
        checks.append(Check("ffmpeg", WARN, "using the imageio-ffmpeg build",
                            "a system ffmpeg is preferred: apt install ffmpeg"))
    else:
        checks.append(Check("ffmpeg", FAIL, "not found",
                            "apt install ffmpeg  (or: pip install imageio-ffmpeg)"))

    checks.append(Check("yt-dlp", OK, "found") if shutil.which("yt-dlp")
                  else Check("yt-dlp", WARN, "not found — local files only",
                             "pip install yt-dlp"))

    # ---- ASR -------------------------------------------------------------
    provider = cfg.asr.provider
    if provider == "srt":
        exists = bool(cfg.asr.transcript and os.path.exists(cfg.asr.transcript))
        checks.append(Check("asr:srt", OK if exists else FAIL,
                            cfg.asr.transcript or "no transcript set",
                            "pass --transcript <file.srt>"))
    elif provider == "faster_whisper":
        checks.append(Check("asr:faster-whisper", OK if _module("faster_whisper")
                            else FAIL, "python package",
                            "pip install faster-whisper"))
    elif provider == "whisperx":
        checks.append(Check("asr:whisperx", OK if _module("whisperx") else FAIL,
                            "python package", "pip install whisperx"))

    # ---- translation -----------------------------------------------------
    tp = cfg.translate.provider
    if tp == "google_free":
        checks.append(Check("translate:google_free",
                            OK if _module("deep_translator") else FAIL,
                            "keyless Google backend",
                            "pip install deep-translator"))
    elif tp == "llm":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        checks.append(Check("translate:llm",
                            OK if (_module("anthropic") and has_key) else FAIL,
                            "anthropic package" if _module("anthropic")
                            else "package missing",
                            "pip install anthropic && export ANTHROPIC_API_KEY=..."))
    elif tp == "indictrans2":
        checks.append(Check("translate:indictrans2",
                            OK if _module("transformers") else FAIL,
                            "transformers + IndicTransToolkit",
                            "pip install transformers "
                            "git+https://github.com/VarunGumma/IndicTransToolkit"))

    # ---- TTS -------------------------------------------------------------
    tts = cfg.tts.provider
    if tts == "edge_tts":
        checks.append(Check("tts:edge_tts", OK if _module("edge_tts") else FAIL,
                            f"voice {cfg.tts.voice or 'te-IN-MohanNeural'} "
                            f"(free, needs network)", "pip install edge-tts"))
    elif tts == "gtts":
        checks.append(Check("tts:gtts", OK if _module("gtts") else FAIL,
                            "free, needs network", "pip install gtts"))
    elif tts == "indicf5":
        ready = _module("transformers") and cfg.tts.ref_audio and cfg.tts.ref_text
        checks.append(Check("tts:indicf5", OK if ready else FAIL,
                            "needs transformers + ref_audio + ref_text",
                            "pip install transformers soundfile; set "
                            "tts.ref_audio and tts.ref_text in the config"))
        cuda, detail = _has_cuda()
        checks.append(Check("gpu (for IndicF5)", OK if cuda else WARN, detail,
                            "CPU works but is slow (~5-10x slower)"))
    elif tts == "sarvam":
        checks.append(Check("tts:sarvam",
                            OK if os.environ.get("SARVAM_API_KEY") else FAIL,
                            "hosted", "export SARVAM_API_KEY=..."))
    elif tts == "elevenlabs":
        checks.append(Check("tts:elevenlabs",
                            OK if os.environ.get("ELEVENLABS_API_KEY") else FAIL,
                            "hosted", "export ELEVENLABS_API_KEY=..."))

    # ---- separation ------------------------------------------------------
    if cfg.mix.keep_background and cfg.mix.separator == "demucs":
        checks.append(Check("mix:demucs", OK if _module("demucs") else WARN,
                            "vocal separation",
                            "pip install demucs  (or set mix.separator: ducking)"))

    # ---- lip-sync --------------------------------------------------------
    if "lipsync" in spec.stages and cfg.lipsync.provider not in ("none", None):
        if cfg.lipsync.provider == "sync_api":
            checks.append(Check("lipsync:sync_api",
                                OK if os.environ.get(cfg.lipsync.api_key_env)
                                else FAIL, "hosted",
                                f"export {cfg.lipsync.api_key_env}=..."))
        else:
            repo = cfg.lipsync.repo_path
            present = bool(repo and os.path.isdir(repo))
            checks.append(Check(f"lipsync:{cfg.lipsync.provider}",
                                OK if present else FAIL,
                                repo or "no repo_path set",
                                "see docs/RUNBOOK.md section 2"))
            cuda, detail = _has_cuda()
            checks.append(Check("gpu (for lip-sync)", OK if cuda else FAIL, detail,
                                "lip-sync needs a CUDA GPU (18 GB for "
                                "LatentSync 1.6); use --mode video instead"))
    return checks


def format_checks(checks: list[Check], cfg: Config) -> str:
    icon = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
    lines = [f"mode={cfg.mode}  voice={cfg.voice}", ""]
    for c in checks:
        lines.append(f"[{icon[c.status]}] {c.name:26s} {c.detail}")
        if c.status != OK and c.fix:
            lines.append(f"{'':10s} -> {c.fix}")
    blockers = [c for c in checks if c.status == FAIL]
    lines.append("")
    lines.append(f"{len(blockers)} blocker(s)." if blockers
                 else "Ready to run.")
    return "\n".join(lines)


def blocking(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.status == FAIL]
