"""Typed configuration loaded from YAML (see config/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AsrCfg:
    provider: str = "faster_whisper"      # faster_whisper | whisperx | srt | mock
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "float16"
    language: str = "en"
    vad: bool = True
    diarize: bool = False
    beam_size: int = 5
    transcript: str | None = None          # for provider=srt
    model_dir: str | None = None           # for provider=sherpa (ONNX model dir)
    vad_model: str | None = None           # silero_vad.onnx, for provider=sherpa
    num_threads: int = 4
    # merge ASR chunks into dubbing units
    max_segment_seconds: float = 12.0
    min_segment_seconds: float = 1.2
    merge_gap_seconds: float = 0.35        # gaps shorter than this are absorbed


@dataclass
class TranslateCfg:
    provider: str = "llm"                  # llm | indictrans2 | mock
    model: str = "claude-sonnet-5"
    endpoint: str | None = None
    length_control: bool = True            # ask for a duration-fitting rendering
    length_tolerance: float = 0.12         # ±12 % of the source slot
    context_window: int = 2                # neighbouring segments given as context
    glossary: str | None = "config/glossary_te.yaml"
    style: str = (
        "Contemplative spiritual discourse. Warm, simple, spoken Telugu that a "
        "village listener and a city listener both understand. Never bookish."
    )


@dataclass
class TtsCfg:
    provider: str = "indicf5"              # indicf5 | sarvam | elevenlabs | mock
    voice: str | None = None
    ref_audio: str | None = None           # speaker prompt for voice cloning
    ref_text: str | None = None            # transcript of ref_audio (IndicF5 needs it)
    sample_rate: int = 24000
    speaking_rate: float = 1.0
    seed: int = 1234


@dataclass
class ProsodyCfg:
    """Speaking-rate model. Calibrate with scripts/calibrate_rate.py."""

    target_syllables_per_second: float = 6.5   # Telugu, measured on your voice
    pause_per_comma: float = 0.18
    pause_per_sentence: float = 0.34
    headroom: float = 0.95                     # budget slack given to the translator


@dataclass
class AlignCfg:
    """Isochrony: how aggressively we may bend time to make the dub fit."""

    max_speedup: float = 1.15              # comfortable band
    max_slowdown: float = 0.90
    hard_max_speedup: float = 1.30         # last resort before we let it overflow
    borrow_gap_fraction: float = 0.80      # share of the following pause we may use
    max_shift: float = 0.30                # how far a segment may start late (s)
    keep_start_anchor: bool = True         # pad at the end, not the start
    pad_symmetric: bool = False


@dataclass
class MixCfg:
    keep_background: bool = True
    separator: str = "ducking"             # demucs | ducking | none
    background_gain_db: float = -6.0
    duck_gain_db: float = -14.0
    dub_gain_db: float = 0.0
    loudness_target_lufs: float = -16.0


@dataclass
class LipsyncCfg:
    provider: str = "latentsync"           # latentsync | musetalk | wav2lip | sync_api | none
    repo_path: str | None = None
    checkpoint: str | None = None
    inference_steps: int = 20
    guidance_scale: float = 1.5
    resolution: int = 512
    face_restore: bool = True              # GFPGAN/CodeFormer pass after paste-back
    only_when_face: bool = True            # skip shots with no visible speaker
    api_key_env: str = "SYNC_API_KEY"


@dataclass
class OutputCfg:
    container: str = "mp4"                 # mp4 for video modes, mp3 for audio
    burn_subtitles: bool = False
    write_srt: bool = True
    crf: int = 18
    preset: str = "medium"
    audio_bitrate: str = "192k"


@dataclass
class Config:
    workdir: str = "runs/default"
    source_lang: str = "en"
    target_lang: str = "te"
    mode: str = "video"                    # audio | video | video-lipsync
    voice: str = "preset"                  # preset | clone
    asr: AsrCfg = field(default_factory=AsrCfg)
    translate: TranslateCfg = field(default_factory=TranslateCfg)
    tts: TtsCfg = field(default_factory=TtsCfg)
    prosody: ProsodyCfg = field(default_factory=ProsodyCfg)
    align: AlignCfg = field(default_factory=AlignCfg)
    mix: MixCfg = field(default_factory=MixCfg)
    lipsync: LipsyncCfg = field(default_factory=LipsyncCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        sub = {
            "asr": AsrCfg, "translate": TranslateCfg, "tts": TtsCfg,
            "prosody": ProsodyCfg, "align": AlignCfg, "mix": MixCfg,
            "lipsync": LipsyncCfg,
            "output": OutputCfg,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sub:
                kwargs[key] = sub[key](**(value or {}))
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)
