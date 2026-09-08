"""ffmpeg/ffprobe helpers plus a dependency-free WAV timeline assembler.

Audio placement is done with the stdlib ``wave`` module on 16-bit PCM so the
timeline maths stays exact and testable; anything that needs real DSP
(resampling, time-stretch, loudness) is delegated to ffmpeg.
"""
from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path


# ---------------------------------------------------------------- binaries
def _resolve(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    try:                                  # pip-installed static build fallback
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if name == "ffmpeg":
            return exe
        probe = str(Path(exe).with_name("ffprobe"))
        return probe if Path(probe).exists() else exe
    except Exception:
        pass
    raise RuntimeError(f"{name} not found. Install ffmpeg, or `pip install imageio-ffmpeg`.")


FFMPEG = _resolve("ffmpeg")
try:
    FFPROBE = _resolve("ffprobe")
except RuntimeError:
    FFPROBE = ""


def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.PIPE,
        text=True,
    )


def ffmpeg(args: list[str], quiet: bool = True) -> None:
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], quiet=quiet)


def probe(path: str | Path) -> dict:
    """Media metadata. Uses ffprobe when present, else parses ffmpeg -i."""
    path = str(path)
    if FFPROBE and Path(FFPROBE).name.startswith("ffprobe"):
        out = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            check=True, capture_output=True, text=True).stdout
        data = json.loads(out)
        info = {"duration": float(data["format"].get("duration", 0.0)),
                "width": 0, "height": 0, "fps": 0.0, "has_video": False}
        for st in data["streams"]:
            if st["codec_type"] == "video" and not info["has_video"]:
                info.update(has_video=True, width=int(st.get("width", 0)),
                            height=int(st.get("height", 0)),
                            fps=_parse_fps(st.get("r_frame_rate", "0/1")))
        return info
    return _probe_via_ffmpeg(path)


def _parse_fps(rate: str) -> float:
    try:
        num, den = rate.split("/")
        return round(float(num) / float(den), 3) if float(den) else 0.0
    except Exception:
        return 0.0


def _probe_via_ffmpeg(path: str) -> dict:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                          capture_output=True, text=True)
    txt = proc.stderr
    info = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "has_video": False}
    import re
    if (m := re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", txt)):
        h, mn, s = m.groups()
        info["duration"] = int(h) * 3600 + int(mn) * 60 + float(s)
    if (m := re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", txt)):
        info.update(has_video=True, width=int(m.group(1)), height=int(m.group(2)))
    if (m := re.search(r"(\d+(?:\.\d+)?) fps", txt)):
        info["fps"] = float(m.group(1))
    return info


def duration_of(path: str | Path) -> float:
    return probe(path)["duration"]


# ---------------------------------------------------------------- audio ops
def extract_audio(video: str | Path, out_wav: str | Path, sr: int = 16000) -> str:
    ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
            "-c:a", "pcm_s16le", str(out_wav)])
    return str(out_wav)


def atempo_chain(factor: float) -> str:
    """ffmpeg's atempo accepts 0.5-2.0 per instance; chain for anything wider."""
    factor = max(factor, 1e-3)
    parts: list[str] = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 1e-4:
        parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts) if parts else "anull"


def time_stretch(src: str | Path, dst: str | Path, factor: float, sr: int) -> str:
    """Change duration by `factor` (>1 = shorter/faster) preserving pitch."""
    ffmpeg(["-i", str(src), "-filter:a", f"{atempo_chain(factor)},aresample={sr}",
            "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])
    return str(dst)


def to_pcm16(src: str | Path, dst: str | Path, sr: int) -> str:
    ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])
    return str(dst)


# ------------------------------------------------------- pure-python mixing
def read_pcm16(path: str | Path) -> tuple[array.array, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        samples = array.array("h")
        samples.frombytes(raw)
        if w.getnchannels() == 2:                        # downmix
            samples = array.array("h", [
                (samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples) - 1, 2)])
    return samples, sr


def write_pcm16(path: str | Path, samples: array.array, sr: int) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return str(path)


def assemble_timeline(clips: list[tuple[float, str]], total_seconds: float,
                      sr: int, fade_ms: int = 8) -> array.array:
    """Place each (start_seconds, wav_path) on a silent bed and sum them.

    A short fade in/out on every clip prevents the clicks you otherwise get
    when a TTS chunk starts on a non-zero sample.
    """
    n_total = int(math.ceil(total_seconds * sr)) + sr    # +1 s of tail room
    bed = array.array("i", bytes(4 * n_total))           # 32-bit accumulator
    fade_n = max(1, int(sr * fade_ms / 1000))
    for start, path in clips:
        samples, clip_sr = read_pcm16(path)
        if clip_sr != sr:
            raise ValueError(f"{path}: {clip_sr} Hz, expected {sr} Hz")
        offset = int(round(start * sr))
        n = len(samples)
        for i in range(n):
            idx = offset + i
            if idx < 0 or idx >= n_total:
                continue
            v = samples[i]
            if i < fade_n:
                v = int(v * i / fade_n)
            elif i > n - fade_n:
                v = int(v * max(0, n - i) / fade_n)
            bed[idx] += v
    out = array.array("h", bytes(2 * n_total))
    peak = max((abs(v) for v in bed), default=0)
    scale = 1.0 if peak <= 32767 else 32767.0 / peak     # only if we clipped
    for i, v in enumerate(bed):
        out[i] = int(max(-32768, min(32767, v * scale)))
    return out


# ------------------------------------------------------- capability probes
_FILTERS: set[str] | None = None


def has_filter(name: str) -> bool:
    """Not every ffmpeg build ships every filter (static builds often drop
    drawtext/libass). Probe once so callers can degrade gracefully."""
    global _FILTERS
    if _FILTERS is None:
        out = subprocess.run([FFMPEG, "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
        _FILTERS = {line.split()[1] for line in out.splitlines()
                    if len(line.split()) > 2 and line.startswith(" ")}
    return name in _FILTERS


def find_font(*keywords: str) -> str | None:
    """Locate an installed font whose filename matches all keywords."""
    roots = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
             Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.tt[fc]"):
            name = path.name.lower()
            if all(k.lower() in name for k in keywords):
                return str(path)
    return None
