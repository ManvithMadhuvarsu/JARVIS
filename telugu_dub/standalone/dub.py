#!/usr/bin/env python3
"""
English audio -> Telugu audio, native voice, original background untouched.

    python dub.py talk.mp3

    1. Demucs splits the recording   -> background stem (KEPT, untouched)
                                        vocals stem     (discarded)
    2. Whisper large-v3 transcribes the isolated vocals
    3. Claude rewrites it as spoken Telugu, with a syllable budget per line
    4. Sarvam Bulbul v3 speaks it    -> native Telugu voice
    5. Fit to the original timings, lay it over the untouched background

Self-contained: one file, no local package to install. Every stage caches to
the work directory, so a failed API call costs you that stage and nothing else.

Environment (put these in a `.env` file next to this script — see below):
    SARVAM_API_KEY      https://dashboard.sarvam.ai   (Rs.100 free credits)
    ANTHROPIC_API_KEY   https://console.anthropic.com

--------------------------------------------------------------------------
KEYS GO IN A LOCAL .env FILE, NEVER IN THIS SCRIPT AND NEVER IN GIT.

    Create standalone/.env (same folder as this file) containing:

        SARVAM_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxx
        ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxx

    dub.py loads it automatically on every run — no `export` / `$env:` needed.
    .env is listed in .gitignore (both the repo root and this folder), and
    `git status` will not show it as a change to commit. Never paste real keys
    into a chat, a commit, or a file that isn't .env — treat a key that was
    ever pasted anywhere outside your own machine as compromised and rotate
    it (delete + recreate) on the provider's dashboard.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import wave
from array import array
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no extra dependency for two lines of KEY=value.

    Only sets a variable if it is not already set in the real environment, so
    `$env:SARVAM_API_KEY=...` in your shell still wins over the file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Telugu prosody: how long a line will take to say, before we synthesise it.
#
# Telugu is an abugida. Every consonant carries an inherent /a/; a vowel sign
# replaces it; the virama (U+0C4D) suppresses it to build a cluster. So:
#     syllables = independent vowels + consonants - viramas
# which predicts spoken length far better than counting characters or words.
# ---------------------------------------------------------------------------
TE_VOWELS = set(range(0x0C05, 0x0C15))
TE_CONSONANTS = set(range(0x0C15, 0x0C3A))
TE_VIRAMA = 0x0C4D
TE_RANGE = range(0x0C00, 0x0C80)
_LATIN_VOWELS = re.compile(r"[aeiouy]+", re.I)


def count_syllables(text: str) -> int:
    text = unicodedata.normalize("NFC", text)
    letters = [c for c in text if c.isalpha()]
    telugu = sum(1 for c in letters if ord(c) in TE_RANGE)
    if letters and telugu / len(letters) > 0.5:
        n = 0
        for ch in text:
            cp = ord(ch)
            if cp in TE_VOWELS or cp in TE_CONSONANTS:
                n += 1
            elif cp == TE_VIRAMA:
                n -= 1
        return max(n, 0)
    n = 0
    for word in re.findall(r"[A-Za-z']+", text):
        groups = len(_LATIN_VOWELS.findall(word))
        if word.lower().endswith("e") and groups > 1:
            groups -= 1
        n += max(groups, 1)
    return n


def estimate_seconds(text: str, rate: float) -> float:
    syl = count_syllables(text)
    if not syl:
        return 0.0
    return round(syl / rate
                 + 0.18 * len(re.findall(r"[,;:—-]", text))
                 + 0.34 * len(re.findall(r"[.!?]", text)), 3)


def syllable_budget(seconds: float, rate: float, headroom: float = 0.95) -> int:
    return max(1, int(seconds * rate * headroom))


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("ffmpeg not found.  apt install ffmpeg   (or: pip install imageio-ffmpeg)")


FFMPEG = _ffmpeg_exe()


def ff(args: list[str]) -> None:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")


def duration_of(path: str | Path) -> float:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not m:
        raise RuntimeError(f"could not read duration of {path}")
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def to_wav(src: str | Path, dst: str | Path, sr: int) -> None:
    ff(["-i", str(src), "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])


def atempo_chain(factor: float) -> str:
    """ffmpeg's atempo takes 0.5-2.0 per instance; chain for anything wider."""
    parts, remaining = [], max(factor, 1e-3)
    while remaining > 2.0:
        parts.append("atempo=2.0"); remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5"); remaining /= 0.5
    if abs(remaining - 1.0) > 1e-4:
        parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts) if parts else "anull"


def time_stretch(src: Path, dst: Path, factor: float, sr: int) -> None:
    """Change duration by `factor` (>1 = faster) without changing pitch."""
    ff(["-i", str(src), "-filter:a", f"{atempo_chain(factor)},aresample={sr}",
        "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])


def read_pcm16(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as w:
        sr, raw = w.getframerate(), w.readframes(w.getnframes())
        samples = array("h"); samples.frombytes(raw)
        if w.getnchannels() == 2:
            samples = array("h", [(samples[i] + samples[i + 1]) // 2
                                  for i in range(0, len(samples) - 1, 2)])
    return samples, sr


def write_pcm16(path: Path, samples: array, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(samples.tobytes())


def assemble(clips: list[tuple[float, Path]], total: float, sr: int) -> array:
    """Drop each clip at its start time on a silent bed and sum."""
    n = int(math.ceil(total * sr)) + sr
    bed = array("i", bytes(4 * n))
    fade = max(1, sr // 125)                       # 8 ms, kills edge clicks
    for start, path in clips:
        samples, clip_sr = read_pcm16(path)
        if clip_sr != sr:
            raise RuntimeError(f"{path}: {clip_sr} Hz, expected {sr}")
        offset, count = int(round(start * sr)), len(samples)
        for i in range(count):
            idx = offset + i
            if 0 <= idx < n:
                v = samples[i]
                if i < fade:
                    v = int(v * i / fade)
                elif i > count - fade:
                    v = int(v * max(0, count - i) / fade)
                bed[idx] += v
    peak = max((abs(v) for v in bed), default=0)
    scale = 1.0 if peak <= 32767 else 32767.0 / peak
    out = array("h", bytes(2 * n))
    for i, v in enumerate(bed):
        out[i] = int(max(-32768, min(32767, v * scale)))
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Line:
    id: int
    start: float
    end: float
    en: str = ""
    te: str = ""
    wav: str | None = None
    tts_seconds: float | None = None
    fit_start: float | None = None
    fit_seconds: float | None = None
    speed: float = 1.0
    overflow: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def slot(self) -> float:
        return self.end - self.start


class State:
    """Every stage caches here, so a crash costs one stage, not the run."""

    def __init__(self, work: Path):
        self.work = work
        self.path = work / "state.json"
        self.data: dict = {"done": [], "lines": [], "paths": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def lines(self) -> list[Line]:
        return [Line(**d) for d in self.data["lines"]]

    @lines.setter
    def lines(self, value: list[Line]) -> None:
        self.data["lines"] = [asdict(x) for x in value]

    def done(self, stage: str) -> bool:
        return stage in self.data["done"]

    def mark(self, stage: str) -> None:
        if stage not in self.data["done"]:
            self.data["done"].append(stage)
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")


def log(msg: str) -> None:
    print(f"[dub] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Demucs — split the recording, keep the background exactly as it is
# ---------------------------------------------------------------------------
def separate(src: Path, work: Path, state: State, stems: int) -> tuple[Path, Path]:
    """Returns (vocals, background).

    Two stems is the right default: `vocals` is what we replace, `no_vocals` is
    everything else and it is kept bit-for-bit. Four stems (--stems 4) keeps
    drums/bass/other separately, which matters when the "background" includes
    audience response that the vocals stem would otherwise swallow.
    """
    out = work / "stems"
    vocals, background = out / "vocals.wav", out / "background.wav"
    if state.done("separate") and vocals.exists() and background.exists():
        log("separate: cached")
        return vocals, background

    out.mkdir(parents=True, exist_ok=True)
    args = ["-n", "htdemucs", "-o", str(work / "_demucs"), str(src)]
    if stems == 2:
        args = ["--two-stems", "vocals"] + args
    log(f"separating with demucs ({stems} stems) — this is the slow step on CPU")
    proc = subprocess.run([sys.executable, "-m", "demucs", *args],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"demucs failed:\n{proc.stderr[-2000:]}")

    stem_dir = next((work / "_demucs" / "htdemucs").iterdir())
    shutil.copy(stem_dir / "vocals.wav", vocals)

    if stems == 2:
        shutil.copy(stem_dir / "no_vocals.wav", background)
    else:
        # sum everything that is not the voice
        others = [stem_dir / f"{n}.wav" for n in ("drums", "bass", "other")]
        present = [p for p in others if p.exists()]
        inputs: list[str] = []
        for p in present:
            inputs += ["-i", str(p)]
        ff([*inputs, "-filter_complex",
            f"amix=inputs={len(present)}:duration=longest:normalize=0",
            "-c:a", "pcm_s16le", str(background)])

    state.data["paths"].update(vocals=str(vocals), background=str(background))
    state.mark("separate")
    log(f"background kept at {background}")
    return vocals, background


# ---------------------------------------------------------------------------
# 2. Whisper — transcribe the ISOLATED vocals (fewer errors than the mix)
# ---------------------------------------------------------------------------
def transcribe(vocals: Path, work: Path, state: State, model_size: str,
               max_line: float, min_line: float, merge_gap: float) -> list[Line]:
    if state.done("transcribe"):
        log(f"transcribe: cached ({len(state.lines)} lines)")
        return state.lines

    from faster_whisper import WhisperModel

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    log(f"whisper {model_size} on {device} ({compute})")

    wav = work / "vocals_16k.wav"
    to_wav(vocals, wav, 16000)

    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments, _ = model.transcribe(
        str(wav), language="en", beam_size=5, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    chunks = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
              for s in segments if s.text.strip()]

    lines = _build_lines(chunks, max_line, min_line, merge_gap)
    state.lines = lines
    state.mark("transcribe")
    log(f"{len(lines)} lines, {sum(l.slot for l in lines):.1f}s of speech")
    return lines


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _build_lines(chunks: list[dict], max_line: float, min_line: float,
                 merge_gap: float) -> list[Line]:
    """Turn ASR chunks into dubbing units: one breath group each.

    ASR splits for transcription convenience, not for dubbing. Merge across
    short gaps, then split anything still too long at a sentence boundary —
    a 15-second unit has to be delivered in one breath and fitted as a whole,
    so one overrun drags the entire span out of sync.
    """
    merged: list[dict] = []
    for ch in chunks:
        if merged and ch["start"] - merged[-1]["end"] <= merge_gap \
                and ch["end"] - merged[-1]["start"] <= max_line:
            merged[-1]["text"] += " " + ch["text"]
            merged[-1]["end"] = ch["end"]
        else:
            merged.append(dict(ch))

    split: list[dict] = []
    for u in merged:
        span = u["end"] - u["start"]
        parts = [p.strip() for p in _SENTENCE_END.split(u["text"]) if p.strip()]
        if span <= max_line or len(parts) < 2:
            split.append(u)
            continue
        weights = [max(1, count_syllables(p)) for p in parts]
        total, cursor = sum(weights), u["start"]
        for part, w in zip(parts, weights):
            piece = span * w / total
            split.append({"start": round(cursor, 3), "end": round(cursor + piece, 3),
                          "text": part})
            cursor += piece

    out: list[Line] = []
    for u in split:
        if out and (u["end"] - u["start"]) < min_line \
                and u["start"] - out[-1].end < merge_gap * 2:
            out[-1].en += " " + u["text"]
            out[-1].end = u["end"]
            continue
        out.append(Line(id=len(out), start=round(u["start"], 3),
                        end=round(u["end"], 3), en=u["text"]))
    for i, l in enumerate(out):
        l.id = i
    return out


# ---------------------------------------------------------------------------
# 3. Claude — a dubbing translator, not a translation engine
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a professional Telugu dubbing translator. Your output \
is spoken aloud over the original video, so it has to work as speech and it has \
to fit the speaker's timing.

1. Translate the MEANING and INTENT, never the English sentence structure. \
Rebuild the sentence the way a Telugu speaker would actually say it.
2. Use natural spoken Telugu grammar and word order. If it reads like \
translated English, it is wrong.
3. Preserve tone exactly: sarcasm stays sarcastic, humour stays funny, urgency \
stays urgent, warmth stays warm. A flat rendering of a joke is a mistranslation.
4. Spoken register, not written prose — the words a Telugu speaker uses at \
home, not textbook Telugu.
5. DO NOT translate proper names, place names, organisation names, product \
names, technical terms, or Sanskrit/yogic vocabulary that already exists in \
Telugu. KEEP the English loanwords Telugu speakers actually use — టైం, ఆఫీసు, \
ఫోన్, స్కూల్, హాస్పిటల్. Over-Sanskritising sounds foreign.
6. LENGTH IS A HARD CONSTRAINT. Each line gives max_syllables. Over budget means \
the dub runs past the speaker's mouth. Cut filler ("you know", "see", "so", \
"I mean"), prefer shorter synonyms. Never pad a short line to fill the budget.
7. Keep sentence-final punctuation — it becomes a pause in the dub.
8. Telugu script only. No transliteration, no English gloss, no commentary.

Return strict JSON: {"lines": [{"id": <int>, "te": "<telugu>"}]}"""


def translate(lines: list[Line], state: State, model: str, rate: float,
              style: str, batch_size: int, tolerance: float) -> list[Line]:
    if state.done("translate"):
        log("translate: cached")
        return state.lines

    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in standalone/.env — "
                 "https://console.anthropic.com")
    client = anthropic.Anthropic()

    def ask(batch: list[Line], tighten: bool) -> dict[int, str]:
        payload = {
            "style": style,
            "context_before": [l.en for l in lines
                               if batch[0].id - 2 <= l.id < batch[0].id],
            "context_after": [l.en for l in lines
                              if batch[-1].id < l.id <= batch[-1].id + 2],
            "lines": [{"id": l.id, "en": l.en, "seconds": round(l.slot, 2),
                       "max_syllables": syllable_budget(
                           l.slot, rate, 0.85 if tighten else 0.95)}
                      for l in batch],
        }
        prefix = ("The previous attempt was TOO LONG. Cut every non-essential "
                  "word and stay under max_syllables.\n" if tighten else "")
        msg = client.messages.create(
            model=model, max_tokens=8000, system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": prefix + json.dumps(payload, ensure_ascii=False)}])
        text = "".join(b.text for b in msg.content if b.type == "text")
        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0) if match else text)
        return {int(x["id"]): x["te"].strip() for x in data["lines"]}

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        got = ask(batch, tighten=False)
        for l in batch:
            l.te = got.get(l.id, "")

        # Only re-ask for the lines that will actually overrun their slot.
        long = [l for l in batch
                if estimate_seconds(l.te, rate) > l.slot * (1 + tolerance)]
        if long:
            retry = ask(long, tighten=True)
            for l in long:
                cand = retry.get(l.id, "")
                if cand and count_syllables(cand) < count_syllables(l.te):
                    l.te, _ = cand, l.notes.append("shortened to fit")
        log(f"translated {min(i + batch_size, len(lines))}/{len(lines)}")

    state.lines = lines
    state.mark("translate")
    return lines


# ---------------------------------------------------------------------------
# 4. Sarvam Bulbul v3 — a Telugu voice trained on Telugu speakers
# ---------------------------------------------------------------------------
SARVAM_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_CHAR_LIMIT = 900


def synthesize(lines: list[Line], work: Path, state: State, speaker: str,
               pace: float, sr: int) -> list[Line]:
    if state.done("tts"):
        log("tts: cached")
        return state.lines

    import requests

    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        sys.exit("SARVAM_API_KEY is not set. Put it in standalone/.env — "
                 "https://dashboard.sarvam.ai (Rs.100 free credits, no card)")

    out = work / "tts"
    out.mkdir(parents=True, exist_ok=True)

    for l in lines:
        if not l.te.strip():
            continue
        dst = out / f"line_{l.id:04d}.wav"
        if not dst.exists():
            text = l.te.strip()[:SARVAM_CHAR_LIMIT]
            for attempt in range(4):
                try:
                    r = requests.post(
                        SARVAM_URL, headers={"api-subscription-key": key},
                        json={"text": text, "target_language_code": "te-IN",
                              "speaker": speaker.lower(),   # names are lowercase
                              "model": "bulbul:v3", "pace": pace,
                              "output_audio_codec": "wav"},
                        timeout=120)
                    r.raise_for_status()
                    body = r.json()
                    audio = (body.get("audios") or [body.get("audio")])[0]
                    raw = out / f"line_{l.id:04d}.raw.wav"
                    raw.write_bytes(base64.b64decode(audio))
                    to_wav(raw, dst, sr)
                    raw.unlink(missing_ok=True)
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise RuntimeError(
                            f"Sarvam failed on line {l.id}: {exc}") from exc
                    time.sleep(2 ** attempt)
        l.wav, l.tts_seconds = str(dst), round(duration_of(dst), 3)
        print(f"  {l.id + 1}/{len(lines)}", end="\r", flush=True)

    spoken = sum(l.tts_seconds or 0 for l in lines)
    source = sum(l.slot for l in lines)
    log(f"\nTelugu {spoken:.1f}s vs English {source:.1f}s (ratio {spoken / source:.2f})")

    measured = _measure_rate(lines)
    if measured:
        log(f"this voice speaks at {measured:.2f} syllables/s — put that in "
            f"--rate for the next run and the budgets get sharper")
    state.lines = lines
    state.mark("tts")
    return lines


def _measure_rate(lines: list[Line]) -> float | None:
    syl = sum(count_syllables(l.te) for l in lines if l.tts_seconds)
    dur = sum(l.tts_seconds for l in lines if l.tts_seconds)
    return round(syl / dur, 2) if dur else None


# ---------------------------------------------------------------------------
# 5. Fit to the original timings, then lay it over the untouched background
#
# Two languages never take the same time to say the same thing. Priority:
#   1. natural tempo inside the original slot
#   2. stretch within a band nobody hears (0.90-1.15x)
#   3. borrow from the pause that follows
#   4. compress hard, flag it
#   5. overflow, bounded, so one bad line cannot drag the rest out of sync
#
# It never starts a line EARLY. Viewers detect audio leading picture at 45 ms
# but tolerate 125 ms of lag (ITU-R BT.1359-1) — late reads as natural, early
# reads as broken.
# ---------------------------------------------------------------------------
MIN_GAP = 0.06


def fit(lines: list[Line], total: float, max_up: float, max_down: float,
        hard_up: float, borrow: float, max_shift: float) -> dict:
    cursor, stats = 0.0, {"natural": 0, "stretched": 0, "borrowed": 0,
                          "hard": 0, "overflow": 0}
    for i, l in enumerate(lines):
        if not l.tts_seconds:
            l.fit_start, l.fit_seconds = l.start, 0.0
            continue

        start = l.start
        if cursor > start:
            start = min(l.start + max_shift, cursor)

        nxt = lines[i + 1].start if i + 1 < len(lines) else total
        pause = max(0.0, nxt - l.end)
        base = max(0.05, l.end - start)
        extended = base + pause * borrow
        d = l.tts_seconds

        if d <= base:
            # It already fits. Leave the tempo alone — stretching a line that
            # fits only makes it less natural.
            speed = 1.0
            stats["natural"] += 1
        elif d / base <= max_up:
            speed = d / base
            stats["stretched"] += 1
        elif d / extended <= max_up:
            # Borrowing the pause: take the slowest tempo that still fits, but
            # never below natural speed. `d / extended` can be < 1 when the
            # pause is generous, and stretching *out* to fill it would slow a
            # line that was too long in the first place.
            speed = max(1.0, d / extended)
            stats["borrowed"] += 1
        elif d / extended <= hard_up:
            speed = d / extended
            stats["hard"] += 1
            l.notes.append(f"compressed {speed:.2f}x")
        else:
            speed = hard_up
            stats["hard"] += 1
            stats["overflow"] += 1
            l.overflow = round(d / speed - extended, 3)
            l.notes.append(f"overflow {l.overflow:.2f}s — shorten this line")

        l.speed = round(speed, 4)
        l.fit_start = round(start, 3)
        l.fit_seconds = round(d / speed, 3)
        cursor = l.fit_start + l.fit_seconds + MIN_GAP
    return stats


def render(lines: list[Line], work: Path, total: float, sr: int) -> Path:
    stretched = work / "stretched"
    stretched.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[float, Path]] = []
    for l in lines:
        if not l.wav or l.fit_start is None:
            continue
        dst = stretched / f"line_{l.id:04d}.wav"
        if abs(l.speed - 1.0) < 1e-3:
            to_wav(Path(l.wav), dst, sr)
        else:
            time_stretch(Path(l.wav), dst, l.speed, sr)
        clips.append((l.fit_start, dst))
    track = work / "telugu_track.wav"
    write_pcm16(track, assemble(clips, total, sr), sr)
    return track


def mix(track: Path, background: Path, out: Path, sr: int, bg_db: float,
        lufs: float, bitrate: str) -> None:
    """The English voice is gone, not hidden — so the background sits high."""
    ff(["-i", str(background), "-i", str(track), "-filter_complex",
        f"[0:a]aresample={sr},volume={bg_db}dB[bg];"
        f"[1:a]aresample={sr}[te];"
        f"[bg][te]amix=inputs=2:duration=longest:dropout_transition=0,"
        f"loudnorm=I={lufs}:TP=-1.5:LRA=11[out]",
        "-map", "[out]", "-c:a", "libmp3lame", "-b:a", bitrate, str(out)])


def write_srt(lines: list[Line], path: Path) -> None:
    def ts(t: float) -> str:
        h, rem = divmod(max(0.0, t), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"

    blocks = []
    for n, l in enumerate([x for x in lines if x.te.strip() and x.fit_start is not None], 1):
        blocks.append(f"{n}\n{ts(l.fit_start)} --> "
                      f"{ts(l.fit_start + (l.fit_seconds or 0))}\n{l.te}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="English audio -> Telugu, native voice, background kept.")
    p.add_argument("audio", help="input audio or video file")
    p.add_argument("-o", "--out", default=None, help="output mp3 (default: <name>_te.mp3)")
    p.add_argument("--work", default=None, help="work dir (default: <name>_work)")
    p.add_argument("--speaker", default="shubh",
                   help="Sarvam voice: shubh aditya rahul rohan (m) / ritu priya neha kavya (f)")
    p.add_argument("--whisper", default="large-v3",
                   help="large-v3 | medium.en | small.en (small on a laptop)")
    p.add_argument("--claude", default="claude-sonnet-5")
    p.add_argument("--rate", type=float, default=6.5,
                   help="Telugu syllables/sec for this voice; the run reports the real one")
    p.add_argument("--pace", type=float, default=1.0, help="Sarvam speaking pace")
    p.add_argument("--stems", type=int, default=2, choices=(2, 4),
                   help="4 keeps drums/bass/other separately — use it when the "
                        "background contains audience noise")
    p.add_argument("--bg-db", type=float, default=-3.0,
                   help="background level; the English voice is gone so this can be high")
    p.add_argument("--style", default=(
        "Natural spoken Telugu for a general audience. Keep the speaker's tone."))
    p.add_argument("--max-line", type=float, default=8.0)
    p.add_argument("--min-line", type=float, default=1.0)
    p.add_argument("--merge-gap", type=float, default=0.35)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--redo", nargs="*", default=[],
                   choices=["separate", "transcribe", "translate", "tts"],
                   help="re-run these stages instead of using the cache")
    args = p.parse_args()

    src = Path(args.audio).expanduser().resolve()
    if not src.exists():
        sys.exit(f"no such file: {src}")
    work = Path(args.work or f"{src.stem}_work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out or f"{src.stem}_te.mp3").resolve()

    state = State(work)
    for stage in args.redo:
        if stage in state.data["done"]:
            state.data["done"].remove(stage)
    state.save()

    started = time.time()
    total = duration_of(src)
    log(f"{src.name} — {total:.1f}s")

    vocals, background = separate(src, work, state, args.stems)
    lines = transcribe(vocals, work, state, args.whisper,
                       args.max_line, args.min_line, args.merge_gap)
    lines = translate(lines, state, args.claude, args.rate, args.style,
                      args.batch, 0.12)
    lines = synthesize(lines, work, state, args.speaker, args.pace, args.sr)

    stats = fit(lines, total, 1.15, 0.90, 1.30, 0.80, 0.30)
    state.lines = lines
    state.save()
    log(f"fit: {stats}")

    track = render(lines, work, total, args.sr)
    mix(track, background, out, args.sr, args.bg_db, -16.0, "192k")
    write_srt(lines, out.with_suffix(".srt"))
    ff(["-i", str(track), "-c:a", "libmp3lame", "-b:a", "192k",
        str(out.with_name(out.stem + "_voice_only.mp3"))])

    bad = [l.id for l in lines if l.overflow > 0]
    print(f"""
done in {time.time() - started:.0f}s

  {out}                          Telugu over your original background
  {out.with_name(out.stem + '_voice_only.mp3')}   Telugu voice alone
  {out.with_suffix('.srt')}      Telugu subtitles
  {work}/state.json              every line, editable

{'Lines that overran their slot: ' + str(bad) if bad else 'Every line fits its slot.'}
To fix a line: edit its "te" in state.json, then
  python {Path(sys.argv[0]).name} {src.name} --redo tts""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
