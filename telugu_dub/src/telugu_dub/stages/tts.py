"""Stage 4 — Telugu speech synthesis, one wav per segment.

Providers
  edge_tts   : Microsoft Edge's neural voices. Free, no API key, no GPU, real
               Telugu (te-IN-MohanNeural male / te-IN-ShrutiNeural female).
               This is the default for the `preset` voice — it is what makes a
               first run possible on a laptop with nothing installed.
  gtts       : Google Translate TTS. Also free and keyless, lower quality, but
               a useful fallback when edge-tts is blocked on your network.
  indicf5    : AI4Bharat IndicF5 (open weights, 11 Indic languages incl. Telugu).
               Zero-shot voice cloning from a reference clip + its transcript,
               so the dub can carry the speaker's own timbre. 24 kHz output.
  sarvam     : Sarvam Bulbul (hosted, Indian-language TTS, streaming).
  elevenlabs : multilingual v2/v3 + instant voice cloning. Best prosody control,
               per-character pricing, cloud only.
  espeak     : libespeak-ng via ctypes. Robotic formant synthesis, but real
               Telugu, fully offline, no GPU and no model download — the only
               provider that always works. Use it as a preview voice.
  mock       : procedural "speech-shaped" audio whose *duration* follows the
               Telugu prosody model. Not intelligible — it exists so the timing,
               mixing and muxing stages can be exercised offline.

Voice cloning of a real person requires that person's permission. Keep the
reference clip out of version control and record the consent basis in
config; see docs/RESEARCH.md#rights-and-consent.
"""
from __future__ import annotations

import array
import math
import os
import random
from pathlib import Path

from ..config import ProsodyCfg, TtsCfg
from ..media import to_pcm16, write_pcm16, duration_of
from ..schema import Segment
from ..telugu_prosody import count_syllables, estimate_speech_duration


def synthesize(segments: list[Segment], cfg: TtsCfg, outdir: str | Path,
               prosody: ProsodyCfg | None = None) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    engine = _get_engine(cfg, prosody or ProsodyCfg())
    for seg in segments:
        if not seg.text_tgt.strip():
            continue
        dst = outdir / f"seg_{seg.id:04d}.wav"
        engine(seg, dst)
        seg.tts_path = str(dst)
        seg.tts_duration = round(duration_of(dst), 3)


def _get_engine(cfg: TtsCfg, prosody: ProsodyCfg):
    if cfg.provider == "edge_tts":
        return _edge_engine(cfg)
    if cfg.provider == "gtts":
        return _gtts_engine(cfg)
    if cfg.provider == "espeak":
        return _espeak_engine(cfg, prosody)
    if cfg.provider == "indicf5":
        return _indicf5_engine(cfg)
    if cfg.provider == "sarvam":
        return _sarvam_engine(cfg)
    if cfg.provider == "elevenlabs":
        return _elevenlabs_engine(cfg)
    if cfg.provider == "mock":
        return _mock_engine(cfg, prosody)
    raise ValueError(f"unknown tts provider: {cfg.provider}")


# ----------------------------------------------------------------- edge-tts
TELUGU_EDGE_VOICES = ("te-IN-MohanNeural", "te-IN-ShrutiNeural")


def list_edge_voices(locale_prefix: str = "te") -> list[dict]:
    """Ask Microsoft which voices exist. Handy when a voice name is rejected."""
    import asyncio

    import edge_tts

    voices = asyncio.run(edge_tts.list_voices())
    return [{"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in voices if v["Locale"].lower().startswith(locale_prefix)]


def _edge_engine(cfg: TtsCfg):
    """Free neural Telugu TTS. Needs network, but no key and no GPU.

    edge-tts takes a rate as a percentage rather than a multiplier; we pass the
    configured speaking_rate through so a whole run can be nudged faster or
    slower without touching the aligner.
    """
    import asyncio

    import edge_tts

    voice = cfg.voice or TELUGU_EDGE_VOICES[0]
    pct = int(round((cfg.speaking_rate - 1.0) * 100))
    rate = f"{pct:+d}%"

    def run(seg: Segment, dst: Path) -> None:
        tmp = dst.with_suffix(".mp3")

        async def synth() -> None:
            communicate = edge_tts.Communicate(seg.text_tgt, voice, rate=rate)
            await communicate.save(str(tmp))

        asyncio.run(synth())
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(
                f"edge-tts returned no audio for segment {seg.id}. "
                f"Check the voice name (see `telugu-dub voices`) and that "
                f"outbound HTTPS to Microsoft is not blocked.")
        to_pcm16(tmp, dst, cfg.sample_rate)
        tmp.unlink(missing_ok=True)

    return run


# --------------------------------------------------------------------- gTTS
def _gtts_engine(cfg: TtsCfg):
    from gtts import gTTS

    def run(seg: Segment, dst: Path) -> None:
        tmp = dst.with_suffix(".mp3")
        gTTS(text=seg.text_tgt, lang="te", slow=False).save(str(tmp))
        to_pcm16(tmp, dst, cfg.sample_rate)
        tmp.unlink(missing_ok=True)

    return run


# ------------------------------------------------------------------ IndicF5
def _indicf5_engine(cfg: TtsCfg):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel

    if not cfg.ref_audio or not cfg.ref_text:
        raise ValueError("indicf5 needs tts.ref_audio and tts.ref_text "
                         "(a clean 5-15 s reference clip and its transcript)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("ai4bharat/IndicF5",
                                      trust_remote_code=True).to(device)

    def run(seg: Segment, dst: Path) -> None:
        audio = model(seg.text_tgt, ref_audio_path=cfg.ref_audio,
                      ref_text=cfg.ref_text)
        audio = np.asarray(audio, dtype=np.float32)
        if np.max(np.abs(audio)) > 1.0:                 # model returns int16 range
            audio = audio / 32768.0
        sf.write(str(dst), audio, 24000)
        if cfg.sample_rate != 24000:
            to_pcm16(dst, dst, cfg.sample_rate)

    return run


# ------------------------------------------------------------------- Sarvam
def _sarvam_engine(cfg: TtsCfg):
    import base64
    import requests

    key = os.environ["SARVAM_API_KEY"]

    def run(seg: Segment, dst: Path) -> None:
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": key},
            json={"inputs": [seg.text_tgt], "target_language_code": "te-IN",
                  "speaker": cfg.voice or "meera", "speech_sample_rate": 22050,
                  "enable_preprocessing": True, "model": "bulbul:v2"},
            timeout=120)
        r.raise_for_status()
        raw = base64.b64decode(r.json()["audios"][0])
        tmp = dst.with_suffix(".raw.wav")
        tmp.write_bytes(raw)
        to_pcm16(tmp, dst, cfg.sample_rate)
        tmp.unlink(missing_ok=True)

    return run


# --------------------------------------------------------------- ElevenLabs
def _elevenlabs_engine(cfg: TtsCfg):
    import requests

    key = os.environ["ELEVENLABS_API_KEY"]
    voice = cfg.voice or "21m00Tcm4TlvDq8ikWAM"

    def run(seg: Segment, dst: Path) -> None:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
            headers={"xi-api-key": key, "accept": "audio/mpeg"},
            json={"text": seg.text_tgt, "model_id": "eleven_multilingual_v2",
                  "voice_settings": {"stability": 0.45, "similarity_boost": 0.8}},
            timeout=180)
        r.raise_for_status()
        tmp = dst.with_suffix(".mp3")
        tmp.write_bytes(r.content)
        to_pcm16(tmp, dst, cfg.sample_rate)
        tmp.unlink(missing_ok=True)

    return run


# ------------------------------------------------------------------- espeak
def _espeak_engine(cfg: TtsCfg, prosody: ProsodyCfg):
    """Offline Telugu through libespeak-ng.

    espeak's rate is words-per-minute, which does not map onto Telugu syllable
    rate; 195 wpm measures at roughly 6.5 syllables/second on the Telugu voice,
    matching the pipeline default. `speaking_rate` scales that.
    """
    from ..espeak import DEFAULT_RATE_WPM, Espeak

    engine = Espeak(voice=cfg.voice or "te",
                    rate_wpm=int(DEFAULT_RATE_WPM * cfg.speaking_rate))

    def run(seg: Segment, dst: Path) -> None:
        tmp = dst.with_suffix(".raw.wav")
        engine.to_wav(seg.text_tgt, tmp)
        to_pcm16(tmp, dst, cfg.sample_rate)
        tmp.unlink(missing_ok=True)

    return run


# --------------------------------------------------------------------- mock
def _mock_engine(cfg: TtsCfg, prosody: ProsodyCfg | None = None):
    """Speech-shaped noise with the duration the prosody model predicts.

    Two formant-ish tones amplitude-modulated at the syllable rate, plus a
    little jitter so the aligner sees realistic over/undershoot rather than a
    perfect fit.
    """
    prosody = prosody or ProsodyCfg()
    rng = random.Random(cfg.seed)
    sr = cfg.sample_rate

    def run(seg: Segment, dst: Path) -> None:
        syl = max(1, count_syllables(seg.text_tgt))
        base = estimate_speech_duration(
            seg.text_tgt, prosody.target_syllables_per_second,
            prosody.pause_per_comma, prosody.pause_per_sentence
        ) / max(cfg.speaking_rate, 0.1)
        dur = max(0.25, base * rng.uniform(0.92, 1.12))
        n = int(dur * sr)
        syl_rate = syl / dur
        f0 = 118.0 + rng.uniform(-8, 8)
        samples = array.array("h", bytes(2 * n))
        for i in range(n):
            t = i / sr
            env = 0.5 * (1 - math.cos(2 * math.pi * syl_rate * t))   # syllable beat
            attack = min(1.0, t / 0.03)
            release = min(1.0, (dur - t) / 0.05)
            v = (math.sin(2 * math.pi * f0 * t)
                 + 0.5 * math.sin(2 * math.pi * 2 * f0 * t)
                 + 0.25 * math.sin(2 * math.pi * 700 * t))
            samples[i] = int(max(-1.0, min(1.0, v / 1.75)) * env * attack
                             * release * 9000)
        write_pcm16(dst, samples, sr)

    return run
