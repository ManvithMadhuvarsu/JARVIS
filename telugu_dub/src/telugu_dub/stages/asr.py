"""Stage 2 — transcribe English with word-level timing, then build dub units.

Providers
  faster_whisper : CTranslate2 Whisper (large-v3). Fast, word timestamps, VAD.
  whisperx       : faster-whisper + wav2vec2 forced alignment + diarization.
                   Use when the video has more than one speaker on camera.
  srt            : reuse a human transcript / YouTube caption file. Free, exact,
                   and the right choice when an official transcript exists.
"""
from __future__ import annotations

from pathlib import Path

from ..config import AsrCfg
from ..schema import Segment
from . import srt as srt_io


def transcribe(audio_path: str, cfg: AsrCfg) -> list[dict]:
    """Return raw ASR chunks: [{start, end, text, speaker}]."""
    if cfg.provider == "srt":
        if not cfg.transcript:
            raise ValueError("asr.provider=srt requires asr.transcript=<file.srt>")
        return [{**c, "speaker": "SPEAKER_00"} for c in srt_io.read_srt(cfg.transcript)]
    if cfg.provider == "faster_whisper":
        return _faster_whisper(audio_path, cfg)
    if cfg.provider == "whisperx":
        return _whisperx(audio_path, cfg)
    raise ValueError(f"unknown asr provider: {cfg.provider}")


def _device(cfg: AsrCfg) -> tuple[str, str]:
    if cfg.device != "auto":
        return cfg.device, cfg.compute_type
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", cfg.compute_type
    except Exception:
        pass
    return "cpu", "int8"


def _faster_whisper(audio_path: str, cfg: AsrCfg) -> list[dict]:
    from faster_whisper import WhisperModel

    device, compute = _device(cfg)
    model = WhisperModel(cfg.model, device=device, compute_type=compute)
    segments, _info = model.transcribe(
        audio_path, language=cfg.language, beam_size=cfg.beam_size,
        word_timestamps=True, vad_filter=cfg.vad,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    out = []
    for s in segments:
        out.append({"start": float(s.start), "end": float(s.end),
                    "text": s.text.strip(), "speaker": "SPEAKER_00"})
    return out


def _whisperx(audio_path: str, cfg: AsrCfg) -> list[dict]:
    import whisperx

    device, compute = _device(cfg)
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(cfg.model, device, compute_type=compute,
                                language=cfg.language)
    result = model.transcribe(audio, batch_size=8 if device == "cuda" else 1)
    align_model, meta = whisperx.load_align_model(language_code=cfg.language,
                                                  device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device,
                            return_char_alignments=False)
    if cfg.diarize:
        import os
        dia = whisperx.DiarizationPipeline(use_auth_token=os.environ.get("HF_TOKEN"),
                                           device=device)
        result = whisperx.assign_word_speakers(dia(audio), result)
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "text": s["text"].strip(), "speaker": s.get("speaker", "SPEAKER_00")}
            for s in result["segments"]]


def build_units(chunks: list[dict], cfg: AsrCfg) -> list[Segment]:
    """Merge ASR chunks into dubbing units.

    ASR segments are cut for transcription convenience, not for dubbing. A unit
    should be one breath group / sentence: long enough that the TTS gets prosody
    right, short enough that a timing error stays local. We merge across gaps
    shorter than `merge_gap_seconds` and never cross a long pause — pauses are
    the elastic we need later in the align stage.
    """
    units: list[Segment] = []
    cur: dict | None = None

    for ch in chunks:
        text = ch["text"].strip()
        if not text:
            continue
        if cur is None:
            cur = {**ch, "text": text}
            continue
        gap = ch["start"] - cur["end"]
        merged_len = ch["end"] - cur["start"]
        same_speaker = ch.get("speaker") == cur.get("speaker")
        ends_sentence = cur["text"].endswith((".", "?", "!", "…"))
        if (same_speaker and gap <= cfg.merge_gap_seconds
                and merged_len <= cfg.max_segment_seconds
                and not (ends_sentence and merged_len > cfg.min_segment_seconds * 2)):
            cur["text"] = f"{cur['text']} {text}".strip()
            cur["end"] = ch["end"]
        else:
            units.append(cur)
            cur = {**ch, "text": text}
    if cur:
        units.append(cur)

    # absorb slivers into their neighbour
    out: list[Segment] = []
    for u in units:
        if (out and (u["end"] - u["start"]) < cfg.min_segment_seconds
                and u["start"] - out[-1].end < cfg.merge_gap_seconds * 2):
            out[-1].text_src = f"{out[-1].text_src} {u['text']}".strip()
            out[-1].end = u["end"]
            continue
        out.append(Segment(id=len(out), start=round(u["start"], 3),
                           end=round(u["end"], 3), text_src=u["text"],
                           speaker=u.get("speaker", "SPEAKER_00")))
    for i, s in enumerate(out):
        s.id = i
    return out
