#!/usr/bin/env python3
"""Measure the articulation rate of the configured Telugu TTS voice.

Everything downstream (the translator's syllable budget, overflow warnings, the
tempo the aligner picks) is derived from one constant: how many Telugu
syllables per second this particular voice actually speaks. Published figures
for Indian-language narration span 6-8 syl/s, which is far too wide to guess
with — so synthesise a calibration set once per voice and write the measured
number into config.prosody.target_syllables_per_second.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telugu_dub.config import Config                     # noqa: E402
from telugu_dub.media import duration_of                 # noqa: E402
from telugu_dub.schema import Segment                    # noqa: E402
from telugu_dub.stages import tts as tts_stage           # noqa: E402
from telugu_dub.telugu_prosody import measure_rate       # noqa: E402

# Short/medium/long, statement/question, with and without internal pauses.
CALIBRATION_SET = [
    "నమస్కారం.",
    "మనసు ఎప్పుడూ దేనికోసమో వెతుకుతూ ఉంటుంది.",
    "మీరు ఎవరు అన్నది మీకు తెలుసా?",
    "ఈ ఒక్క క్షణం మీద శ్రద్ధ పెడితే, మిగతావన్నీ వాటంతట అవే కుదురుకుంటాయి.",
    "శరీరం, మనసు, భావోద్వేగాలు — ఇవన్నీ మీరు పోగుచేసుకున్నవి మాత్రమే.",
    "అవగాహన పెరిగేకొద్దీ బాధ తగ్గుతుంది; ఇది తత్వశాస్త్రం కాదు, ఒక శాస్త్రం.",
    "ధ్యానం అంటే ఏదో చేయడం కాదు, ఏదీ చేయకుండా ఉండగలగడం.",
    "మీ జీవితాన్ని మీరే ఎలా సృష్టించుకుంటున్నారో ఒకసారి చూడండి.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="runs/calibration")
    ap.add_argument("--write", action="store_true",
                    help="write the measured rate back into the config file")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    outdir = Path(args.outdir)
    segments = [Segment(id=i, start=0, end=1, text_tgt=t)
                for i, t in enumerate(CALIBRATION_SET)]
    tts_stage.synthesize(segments, cfg.tts, outdir, cfg.prosody)

    samples = [(s.text_tgt, duration_of(s.tts_path)) for s in segments if s.tts_path]
    result = measure_rate(samples)
    result["provider"] = cfg.tts.provider
    result["voice"] = cfg.tts.voice or cfg.tts.ref_audio or "default"
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.write and result["n"]:
        path = Path(args.config)
        text = path.read_text(encoding="utf-8")
        import re
        new = re.sub(r"(target_syllables_per_second:\s*)[\d.]+",
                     rf"\g<1>{result['rate']}", text)
        path.write_text(new, encoding="utf-8")
        print(f"wrote target_syllables_per_second: {result['rate']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
