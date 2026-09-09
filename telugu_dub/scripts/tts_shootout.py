#!/usr/bin/env python3
"""Generate the SAME Telugu lines through every TTS provider, for a blind A/B.

    export SARVAM_API_KEY=...            # ₹100 free credits on signup
    export ELEVENLABS_API_KEY=...        # optional
    export GOOGLE_APPLICATION_CREDENTIALS=...  # optional
    export AZURE_SPEECH_KEY=... AZURE_SPEECH_REGION=centralindia  # optional

    python scripts/tts_shootout.py --providers sarvam edge_tts espeak
    python scripts/tts_shootout.py --text-file samples/where_is_time_for_yoga_te.json

Why bother: vendor demos are cherry-picked, and word-error-rate — the number
most vendors quote — cannot see the failure that actually matters here. The PSP
benchmark (arXiv 2604.25476) found ElevenLabs v3 posts an excellent Hindi WER
while its Telugu output has a pitch range far narrower than a native speaker's;
you hear that as flat and mechanical, and no accuracy metric reports it. The
only reliable test is a Telugu speaker listening to the same sentences from
each system without knowing which is which.

Outputs one wav per provider per line, plus a shuffled blind-test folder whose
filenames do not reveal the provider, and a key file to score against.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_dub.media import duration_of, to_pcm16  # noqa: E402
from telugu_dub.telugu_prosody import count_syllables  # noqa: E402

# Sentences chosen to stress the things Western TTS gets wrong on Telugu:
# retroflex consonants (ట ఠ డ ఢ ణ ళ), long/short vowel contrast, gemination,
# a question contour, and an English loanword mid-sentence (code-switching).
DEFAULT_LINES = [
    "మనసు స్వభావం అలాంటిది, అది ఎప్పుడూ దేనికోసమో వెతుకుతూనే ఉంటుంది.",
    "పిల్లలకి కావాల్సింది బొమ్మలు కాదు, కొంచెం ప్రేమ, కొంచెం సమయం.",
    "మీరు నిజంగా ఏం చేయాలనుకుంటున్నారో మీకు తెలుసా?",
    "రోజుకి ముప్పై నిమిషాలు పెట్టండి, మీ జీవితం మారిపోతుంది.",
    "ఆఫీసు నుంచి వచ్చాక ఫోన్ పక్కన పెట్టి, ఒక్క క్షణం కూర్చోండి.",
    "కళ్ళు మూసుకుని, గట్టిగా ఊపిరి తీసుకోండి — అంతే చాలు.",
]


# --------------------------------------------------------------- providers
def gen_sarvam(text: str, dst: Path, voice: str | None) -> None:
    import requests

    r = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]},
        json={"text": text, "target_language_code": "te-IN",
              "speaker": (voice or "shubh").lower(), "model": "bulbul:v3",
              "output_audio_codec": "wav"},
        timeout=120)
    r.raise_for_status()
    payload = r.json()
    audios = payload.get("audios") or [payload.get("audio")]
    tmp = dst.with_suffix(".raw.wav")
    tmp.write_bytes(base64.b64decode(audios[0]))
    to_pcm16(tmp, dst, 24000)
    tmp.unlink(missing_ok=True)


def gen_elevenlabs(text: str, dst: Path, voice: str | None) -> None:
    import requests

    vid = voice or "21m00Tcm4TlvDq8ikWAM"
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                 "accept": "audio/mpeg"},
        json={"text": text, "model_id": "eleven_v3",
              "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}},
        timeout=180)
    r.raise_for_status()
    tmp = dst.with_suffix(".mp3")
    tmp.write_bytes(r.content)
    to_pcm16(tmp, dst, 24000)
    tmp.unlink(missing_ok=True)


def gen_google(text: str, dst: Path, voice: str | None) -> None:
    from google.cloud import texttospeech as tts

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=tts.SynthesisInput(text=text),
        voice=tts.VoiceSelectionParams(language_code="te-IN",
                                       name=voice or "te-IN-Standard-A"),
        audio_config=tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=24000))
    dst.write_bytes(response.audio_content)


def gen_azure(text: str, dst: Path, voice: str | None) -> None:
    import requests

    region = os.environ.get("AZURE_SPEECH_REGION", "centralindia")
    token = requests.post(
        f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
        headers={"Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"]},
        timeout=30).text
    name = voice or "te-IN-MohanNeural"
    ssml = (f"<speak version='1.0' xml:lang='te-IN'><voice name='{name}'>"
            f"{text}</voice></speak>")
    r = requests.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm"},
        data=ssml.encode("utf-8"), timeout=120)
    r.raise_for_status()
    dst.write_bytes(r.content)


def gen_edge(text: str, dst: Path, voice: str | None) -> None:
    import asyncio

    import edge_tts

    tmp = dst.with_suffix(".mp3")

    async def run() -> None:
        await edge_tts.Communicate(text, voice or "te-IN-MohanNeural").save(str(tmp))

    asyncio.run(run())
    to_pcm16(tmp, dst, 24000)
    tmp.unlink(missing_ok=True)


_ESPEAK: dict[str, object] = {}


def gen_espeak(text: str, dst: Path, voice: str | None) -> None:
    # espeak-ng keeps global state in the shared library: constructing a second
    # engine re-initialises the first, and doing it in a loop hangs. One engine
    # per voice, reused.
    from telugu_dub.espeak import Espeak

    key = voice or "te"
    if key not in _ESPEAK:
        _ESPEAK[key] = Espeak(voice=key)
    _ESPEAK[key].to_wav(text, dst)


PROVIDERS = {
    "sarvam": (gen_sarvam, "Bulbul v3 — trained from scratch on Indian speech"),
    "elevenlabs": (gen_elevenlabs, "v3 — global-first model, 90+ languages"),
    "google": (gen_google, "Cloud TTS te-IN"),
    "azure": (gen_azure, "Neural te-IN-MohanNeural / ShrutiNeural"),
    "edge_tts": (gen_edge, "Free Microsoft neural, same voices as Azure"),
    "espeak": (gen_espeak, "Offline formant synthesis — the floor, for reference"),
}


# -------------------------------------------------------------------- main
def load_lines(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_LINES
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    table = raw.get("translations", raw)
    lines = [v.strip() for v in table.values() if v and v.strip()]
    # longer lines expose prosody failures that short ones hide
    lines.sort(key=count_syllables, reverse=True)
    return lines[:8]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["sarvam", "edge_tts", "espeak"],
                    choices=sorted(PROVIDERS))
    ap.add_argument("--voice", nargs="*", default=[], metavar="provider=voice",
                    help="e.g. --voice sarvam=ritu azure=te-IN-ShrutiNeural")
    ap.add_argument("--text-file", help="a translations JSON from a dub run")
    ap.add_argument("--out", default="runs/shootout")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    voices = dict(v.split("=", 1) for v in args.voice)
    lines = load_lines(args.text_file)
    out = Path(args.out)
    blind = out / "blind"
    shutil.rmtree(out, ignore_errors=True)
    blind.mkdir(parents=True, exist_ok=True)

    key, produced = [], 0
    for provider in args.providers:
        fn, blurb = PROVIDERS[provider]
        pdir = out / provider
        pdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{provider}  — {blurb}")
        for i, line in enumerate(lines):
            dst = pdir / f"line_{i:02d}.wav"
            try:
                t0 = time.time()
                fn(line, dst, voices.get(provider))
                secs, took = duration_of(dst), time.time() - t0
                syl = count_syllables(line)
                print(f"  line {i}: {secs:5.2f}s  {syl/secs:4.2f} syl/s  "
                      f"({took:.1f}s to generate)")
                token = hashlib.sha1(
                    f"{args.seed}{provider}{i}".encode()).hexdigest()[:8]
                shutil.copy(dst, blind / f"{i:02d}_{token}.wav")
                key.append({"file": f"{i:02d}_{token}.wav", "provider": provider,
                            "line": i, "text": line,
                            "seconds": round(secs, 2),
                            "syllables_per_second": round(syl / secs, 2)})
                produced += 1
            except KeyError as exc:
                print(f"  skipped: missing environment variable {exc}")
                break
            except Exception as exc:
                print(f"  line {i} FAILED: {type(exc).__name__}: {exc}")

    random.Random(args.seed).shuffle(key)
    (out / "KEY.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

    scorecard = out / "SCORECARD.md"
    scorecard.write_text(_scorecard(lines), encoding="utf-8")
    print(f"""
{produced} clips -> {out}

Now run the test properly:
  1. Open {blind}/ — filenames do not reveal the provider.
  2. Have a Telugu speaker (not you) listen to all clips for line 00, then 01…
     and score each on the sheet in {scorecard}.
  3. Only then open {out}/KEY.json.

Score these separately — a system can nail one and fail another:
  RETROFLEX  ట ఠ డ ఢ ణ ళ said properly, or collapsed to dental sounds
  PITCH      does it move like a person, or drone flat
  RHYTHM     Telugu timing, or English timing with Telugu words on top
  LOANWORDS  "ఆఫీసు", "ఫోన్" — natural, or suddenly a different accent
  OVERALL    would you put this in front of a Telugu audience""")
    return 0


def _scorecard(lines: list[str]) -> str:
    rows = ["# Blind listening scorecard", "",
            "Score 1-5 (5 = indistinguishable from a native speaker).",
            "Do not look at KEY.json until every row is filled in.", ""]
    for i, line in enumerate(lines):
        rows += [f"## Line {i:02d}", "", f"> {line}", "",
                 "| clip | retroflex | pitch | rhythm | loanwords | overall |",
                 "|---|---|---|---|---|---|",
                 "|  |  |  |  |  |  |", "|  |  |  |  |  |  |",
                 "|  |  |  |  |  |  |", ""]
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
