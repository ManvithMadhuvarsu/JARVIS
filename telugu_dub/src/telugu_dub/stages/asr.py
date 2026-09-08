"""Stage 2 — transcribe English with word-level timing, then build dub units.

Providers
  faster_whisper : CTranslate2 Whisper (large-v3). Fast, word timestamps, VAD.
  whisperx       : faster-whisper + wav2vec2 forced alignment + diarization.
                   Use when the video has more than one speaker on camera.
  sherpa         : sherpa-onnx running Whisper as ONNX, with Silero VAD doing
                   the segmentation. Fully offline once the models are on disk,
                   installs from PyPI with no HuggingFace login, and the models
                   are ordinary GitHub release downloads — the path of least
                   resistance on a locked-down machine or an air-gapped box.
  srt            : reuse a human transcript / YouTube caption file. Free, exact,
                   and the right choice when an official transcript exists.
"""
from __future__ import annotations

import re
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
    if cfg.provider == "sherpa":
        return _sherpa_onnx(audio_path, cfg)
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


def _sherpa_onnx(audio_path: str, cfg: AsrCfg) -> list[dict]:
    """Offline Whisper-as-ONNX with Silero VAD segmentation.

    The VAD is what produces timings here: Whisper decodes each speech run
    independently, so the segment boundaries come from where the speaker
    actually paused rather than from the decoder's own guesses. That is exactly
    the unit the dubbing pipeline wants.
    """
    import numpy as np
    import sherpa_onnx

    from ..media import read_pcm16

    model_dir = Path(cfg.model_dir or "models/sherpa-onnx-whisper-base.en")
    if not model_dir.exists():
        raise RuntimeError(
            f"{model_dir} not found. Download a sherpa-onnx Whisper model:\n"
            f"  curl -L -o m.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/"
            f"releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2"
            f" && tar xjf m.tar.bz2")

    stem = model_dir.name.replace("sherpa-onnx-whisper-", "")
    encoder = next(model_dir.glob("*-encoder.int8.onnx"),
                   model_dir / f"{stem}-encoder.onnx")
    decoder = next(model_dir.glob("*-decoder.int8.onnx"),
                   model_dir / f"{stem}-decoder.onnx")
    tokens = next(model_dir.glob("*-tokens.txt"))

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(encoder), decoder=str(decoder), tokens=str(tokens),
        num_threads=cfg.num_threads, decoding_method="greedy_search",
        language=cfg.language, task="transcribe")

    vad_path = cfg.vad_model or str(model_dir.parent / "silero_vad.onnx")
    if not Path(vad_path).exists():
        raise RuntimeError(
            f"{vad_path} not found. Download it:\n  curl -L -O https://github."
            f"com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx")

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_path
    vad_config.silero_vad.threshold = 0.5
    vad_config.silero_vad.min_silence_duration = cfg.merge_gap_seconds
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.silero_vad.max_speech_duration = cfg.max_segment_seconds
    vad_config.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(vad_config,
                                            buffer_size_in_seconds=180)

    samples16, sr = read_pcm16(audio_path)
    if sr != 16000:
        raise ValueError(f"{audio_path}: expected 16 kHz, got {sr}")
    audio = np.asarray(samples16, dtype=np.float32) / 32768.0

    out: list[dict] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, seg.samples)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                start = seg.start / 16000
                out.append({"start": round(start, 3),
                            "end": round(start + len(seg.samples) / 16000, 3),
                            "text": text, "speaker": "SPEAKER_00"})
            vad.pop()

    window = 8192
    for i in range(0, len(audio), window):
        vad.accept_waveform(audio[i:i + window])
        drain()
    vad.flush()
    drain()
    return out


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_long_units(units: list[dict], cfg: AsrCfg) -> list[dict]:
    """Break over-long units at sentence boundaries.

    A VAD can run 15+ seconds without a qualifying pause when someone speaks in
    long, connected clauses. That is one dubbing unit the TTS must deliver in a
    single breath and the aligner can only fit as a whole — so one overrunning
    clause drags the entire span out of sync. Splitting on sentence boundaries
    and allocating the span by syllable count keeps the timing local, and the
    cut lands where the speaker was going to pause anyway.
    """
    from ..telugu_prosody import count_syllables

    out: list[dict] = []
    for u in units:
        span = u["end"] - u["start"]
        parts = [p.strip() for p in _SENTENCE_END.split(u["text"].strip()) if p.strip()]
        if span <= cfg.max_segment_seconds or len(parts) < 2:
            out.append(u)
            continue
        weights = [max(1, count_syllables(p)) for p in parts]
        total = sum(weights)
        cursor = u["start"]
        for part, weight in zip(parts, weights):
            piece = span * weight / total
            out.append({**u, "text": part, "start": round(cursor, 3),
                        "end": round(cursor + piece, 3)})
            cursor += piece
    return out


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

    units = split_long_units(units, cfg)

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
