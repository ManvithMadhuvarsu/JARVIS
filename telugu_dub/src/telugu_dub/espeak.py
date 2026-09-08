"""Offline Telugu speech via libespeak-ng, driven through ctypes.

espeak-ng is a formant synthesiser, not a neural model: the result is robotic
and unmistakably synthetic. It earns its place for two reasons —

  * it is the only Telugu TTS that runs with no GPU, no API key and no model
    download (the shared library ships in the `espeakng-loader` wheel), so a
    pipeline can always produce *something* audible; and
  * it is deterministic and instant, which makes it the right voice for testing
    timing, mixing and QC without waiting on a neural model.

Ship it as a preview voice. For anything a listener will actually hear, use
edge-tts, Sarvam or IndicF5.
"""
from __future__ import annotations

import ctypes
from array import array
from pathlib import Path

AUDIO_OUTPUT_RETRIEVAL = 1
espeakCHARS_UTF8 = 1
espeakENDPAUSE = 0x1000

# espeak_PARAMETER enum. Note SILENCE is 0, so RATE is 1 — passing 0 for rate
# silently does nothing, which looks like "the rate control is broken".
espeakRATE, espeakVOLUME, espeakPITCH, espeakRANGE = 1, 2, 3, 4

# Words-per-minute is espeak's own unit and does not map cleanly onto Telugu
# syllable rate. Measured on the Telugu voice: 175 wpm ~ 5.7 syl/s,
# 250 wpm ~ 8.6 syl/s. 195 lands near the 6.5 syl/s the pipeline assumes.
DEFAULT_RATE_WPM = 195

_SAMPLE_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short),
                              ctypes.c_int, ctypes.c_void_p)


class Espeak:
    """Thin wrapper around the espeak-ng C API.

    The library keeps global state, so one instance per process is the
    supported pattern; constructing a second one re-initialises the first.
    """

    def __init__(self, voice: str = "te", rate_wpm: int = DEFAULT_RATE_WPM,
                 pitch: int = 50, volume: int = 100):
        self.lib, data_path = _load_library()
        self.lib.espeak_SetParameter.argtypes = [ctypes.c_int] * 3
        self.lib.espeak_SetParameter.restype = ctypes.c_int
        self.sample_rate = self.lib.espeak_Initialize(
            AUDIO_OUTPUT_RETRIEVAL, 1000, data_path.encode(), 0)
        if self.sample_rate <= 0:
            raise RuntimeError("espeak_Initialize failed")

        if self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise RuntimeError(
                f"espeak-ng has no voice {voice!r}. Telugu is 'te'.")
        for param, value in ((espeakRATE, rate_wpm), (espeakVOLUME, volume),
                             (espeakPITCH, pitch)):
            self.lib.espeak_SetParameter(param, int(value), 0)  # 0 = absolute

        self._buffer = array("h")
        self._cb = _SAMPLE_CB(self._on_samples)
        self.lib.espeak_SetSynthCallback(self._cb)

    def _on_samples(self, wav, numsamples, events) -> int:
        if wav and numsamples > 0:
            self._buffer.extend(wav[i] for i in range(numsamples))
        return 0                                   # 0 = keep going

    def set_rate(self, wpm: int) -> None:
        self.lib.espeak_SetParameter(espeakRATE, int(wpm), 0)

    def synth(self, text: str) -> array:
        """Synthesise one utterance and return 16-bit PCM at self.sample_rate."""
        self._buffer = array("h")
        payload = text.encode("utf-8")
        rc = self.lib.espeak_Synth(payload, len(payload) + 1, 0, 0, 0,
                                   espeakCHARS_UTF8 | espeakENDPAUSE,
                                   None, None)
        if rc != 0:
            raise RuntimeError(f"espeak_Synth returned {rc}")
        self.lib.espeak_Synchronize()
        return self._buffer

    def to_wav(self, text: str, path: str | Path) -> str:
        from .media import write_pcm16
        return write_pcm16(path, self.synth(text), self.sample_rate)


def _load_library() -> tuple[ctypes.CDLL, str]:
    """Prefer the pip-installed library so no system package is required."""
    try:
        import espeakng_loader
        lib = ctypes.CDLL(espeakng_loader.get_library_path())
        return lib, str(espeakng_loader.get_data_path())
    except Exception:
        pass
    for name in ("libespeak-ng.so.1", "libespeak-ng.so", "libespeak.so.1"):
        try:
            return ctypes.CDLL(name), "/usr/share/espeak-ng-data"
        except OSError:
            continue
    raise RuntimeError(
        "libespeak-ng not found. Install it with: pip install espeakng-loader")
