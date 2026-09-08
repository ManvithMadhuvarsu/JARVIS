"""Stage 7 — re-generate the mouth so the face matches the Telugu audio.

Providers (all take: original video + dubbed audio -> new video)

  latentsync : ByteDance LatentSync 1.6, audio-conditioned latent diffusion.
               Best open-source visual fidelity at 512x512; ~18 GB VRAM.
               Apache-2.0.
  musetalk   : real-time-ish inpainting model, ~30 fps on a V100, lower VRAM.
               The right default for long videos and iteration.
  wav2lip    : the 2020 baseline. Runs on anything, strongest raw sync score,
               but a visibly soft 96x96 mouth patch — needs a face-restoration
               pass to be watchable at 720p+.
  sync_api   : hosted API (sync.so). No GPU to own; per-minute cost.
  none       : passthrough, keep the original picture (voice-over style).

All local providers are invoked as subprocesses against their own repo+venv,
because their dependency pins conflict with each other and with this package.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from ..config import LipsyncCfg
from ..media import ffmpeg


def run_lipsync(video: str, audio: str, out_video: str, cfg: LipsyncCfg) -> dict:
    started = time.time()
    provider = cfg.provider or "none"
    if provider == "none":
        _mux_only(video, audio, out_video)
    elif provider == "latentsync":
        _latentsync(video, audio, out_video, cfg)
    elif provider == "musetalk":
        _musetalk(video, audio, out_video, cfg)
    elif provider == "wav2lip":
        _wav2lip(video, audio, out_video, cfg)
    elif provider == "sync_api":
        _sync_api(video, audio, out_video, cfg)
    else:
        raise ValueError(f"unknown lipsync provider: {provider}")
    return {"provider": provider, "seconds": round(time.time() - started, 1),
            "output": out_video}


def _mux_only(video: str, audio: str, out_video: str) -> None:
    ffmpeg(["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_video])


def _require_repo(cfg: LipsyncCfg, name: str) -> Path:
    if not cfg.repo_path or not Path(cfg.repo_path).exists():
        raise RuntimeError(
            f"lipsync.repo_path must point at a {name} checkout "
            f"(see docs/RUNBOOK.md for setup)")
    return Path(cfg.repo_path)


def _python_for(repo: Path) -> str:
    """Prefer the repo's own venv; these projects pin incompatible torch."""
    for candidate in (repo / ".venv/bin/python", repo / "venv/bin/python"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("python3") or "python"


def _latentsync(video: str, audio: str, out_video: str, cfg: LipsyncCfg) -> None:
    repo = _require_repo(cfg, "LatentSync")
    ckpt = cfg.checkpoint or str(repo / "checkpoints/latentsync_unet.pt")
    cmd = [_python_for(repo), "-m", "scripts.inference",
           "--unet_config_path", str(repo / "configs/unet/stage2_512.yaml"),
           "--inference_ckpt_path", ckpt,
           "--video_path", os.path.abspath(video),
           "--audio_path", os.path.abspath(audio),
           "--video_out_path", os.path.abspath(out_video),
           "--inference_steps", str(cfg.inference_steps),
           "--guidance_scale", str(cfg.guidance_scale)]
    subprocess.run(cmd, cwd=repo, check=True)


def _musetalk(video: str, audio: str, out_video: str, cfg: LipsyncCfg) -> None:
    repo = _require_repo(cfg, "MuseTalk")
    cfg_file = Path(out_video).with_suffix(".musetalk.yaml")
    cfg_file.write_text(
        f"task_0:\n  video_path: {os.path.abspath(video)}\n"
        f"  audio_path: {os.path.abspath(audio)}\n", encoding="utf-8")
    subprocess.run([_python_for(repo), "-m", "scripts.inference",
                    "--inference_config", str(cfg_file),
                    "--result_dir", str(Path(out_video).parent)],
                   cwd=repo, check=True)


def _wav2lip(video: str, audio: str, out_video: str, cfg: LipsyncCfg) -> None:
    repo = _require_repo(cfg, "Wav2Lip")
    ckpt = cfg.checkpoint or str(repo / "checkpoints/wav2lip_gan.pth")
    subprocess.run([_python_for(repo), "inference.py",
                    "--checkpoint_path", ckpt,
                    "--face", os.path.abspath(video),
                    "--audio", os.path.abspath(audio),
                    "--outfile", os.path.abspath(out_video),
                    "--resize_factor", "1", "--nosmooth"],
                   cwd=repo, check=True)


def _sync_api(video: str, audio: str, out_video: str, cfg: LipsyncCfg) -> None:
    """Hosted lip-sync. Inputs must be reachable URLs, not local paths."""
    import requests

    key = os.environ[cfg.api_key_env]
    if not (video.startswith("http") and audio.startswith("http")):
        raise ValueError("sync_api needs public URLs for video and audio; "
                         "upload them (e.g. S3 presigned) first")
    r = requests.post("https://api.sync.so/v2/generate",
                      headers={"x-api-key": key},
                      json={"model": "lipsync-2",
                            "input": [{"type": "video", "url": video},
                                      {"type": "audio", "url": audio}],
                            "options": {"sync_mode": "bounce"}},
                      timeout=60)
    r.raise_for_status()
    job = r.json()["id"]
    while True:
        time.sleep(10)
        st = requests.get(f"https://api.sync.so/v2/generate/{job}",
                          headers={"x-api-key": key}, timeout=60).json()
        if st["status"] == "COMPLETED":
            data = requests.get(st["outputUrl"], timeout=600).content
            Path(out_video).write_bytes(data)
            return
        if st["status"] in ("FAILED", "CANCELED", "REJECTED"):
            raise RuntimeError(f"sync.so job {job}: {st}")
