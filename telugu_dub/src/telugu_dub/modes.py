"""The use-case matrix: what to produce, and in whose voice.

Two independent choices:

  OUTPUT MODE                          VOICE
  ----------------------------------   ----------------------------------
  audio          just the Telugu       preset   a stock Telugu voice
                 audio track                    (no consent needed)
  video          original picture,     clone    the speaker's own voice
                 dubbed audio                   (needs their consent and a
  video-lipsync  picture regenerated            clean reference recording)
                 to match the dub

Six combinations. They are not six pipelines — they are the same pipeline with
later stages switched off, which is why `audio` costs a minute per video-minute
and `video-lipsync` costs twenty.

    ingest → asr → translate → tts → align → render → mix ─┬─► audio
                                                           ├─► + mux ► video
                                                           └─► + lipsync + mux
                                                                 ► video-lipsync

Cost and requirements per combination live in docs/USE_CASES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config

MODES = ("audio", "video", "video-lipsync")
VOICES = ("preset", "clone")


@dataclass
class ModeSpec:
    stages: tuple[str, ...]
    needs_video: bool
    container: str
    summary: str


MODE_SPECS: dict[str, ModeSpec] = {
    "audio": ModeSpec(
        stages=("ingest", "asr", "translate", "tts", "align", "render", "mix", "mux"),
        needs_video=False, container="mp3",
        summary="Telugu audio track only — podcast, radio, or a separate audio "
                "language track alongside the original video."),
    "video": ModeSpec(
        stages=("ingest", "asr", "translate", "tts", "align", "render", "mix", "mux"),
        needs_video=True, container="mp4",
        summary="Original picture with the Telugu dub over it. Lips stay "
                "English — the standard voice-over look, and what most dubbed "
                "content on television actually is."),
    "video-lipsync": ModeSpec(
        stages=("ingest", "asr", "translate", "tts", "align", "render", "mix",
                "lipsync", "mux"),
        needs_video=True, container="mp4",
        summary="Picture regenerated so the mouth matches the Telugu. "
                "Needs a GPU and 10-25 minutes per video-minute."),
}

# Provider defaults per voice choice. `preset` deliberately requires nothing:
# edge-tts uses Microsoft's free neural Telugu voices, no key, no GPU.
VOICE_PRESETS: dict[str, dict] = {
    "preset": {"provider": "edge_tts", "voice": "te-IN-MohanNeural"},
    "clone": {"provider": "indicf5", "voice": None},
}


def apply(cfg: Config, mode: str | None = None, voice: str | None = None) -> Config:
    """Switch the config into one of the six use cases."""
    if mode:
        if mode not in MODE_SPECS:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        cfg.mode = mode
        spec = MODE_SPECS[mode]
        cfg.output.container = spec.container
        if mode != "video-lipsync":
            cfg.lipsync.provider = "none"
        elif cfg.lipsync.provider == "none":
            cfg.lipsync.provider = "latentsync"

    if voice:
        if voice not in VOICES:
            raise ValueError(f"voice must be one of {VOICES}, got {voice!r}")
        cfg.voice = voice
        preset = VOICE_PRESETS[voice]
        # Only override the provider if the current one belongs to the other
        # camp; an explicit `tts.provider: sarvam` in the config is respected.
        cloning = {"indicf5", "elevenlabs"}
        current_is_cloning = cfg.tts.provider in cloning
        if voice == "preset" and current_is_cloning:
            cfg.tts.provider = preset["provider"]
            cfg.tts.voice = preset["voice"]
            cfg.tts.ref_audio = None
            cfg.tts.ref_text = None
        elif voice == "clone" and not current_is_cloning:
            cfg.tts.provider = preset["provider"]
    return cfg


def stages_for(cfg: Config) -> tuple[str, ...]:
    return MODE_SPECS[cfg.mode].stages


def describe(cfg: Config) -> str:
    spec = MODE_SPECS[cfg.mode]
    voice_note = ("stock Telugu voice" if cfg.voice == "preset"
                  else "cloned voice (consent required)")
    return (f"mode={cfg.mode} ({spec.container}), voice={cfg.voice} — {voice_note}\n"
            f"  {spec.summary}")
