# telugu-dub — English → Telugu dubbing with lip sync

Takes an English talk (the working example is a Sadhguru discourse) and produces
a Telugu version where the **voice and the mouth both match**: translated,
re-voiced, time-fitted to the original speech, and lip-synced.

- **What to build and why:** [`docs/RESEARCH.md`](docs/RESEARCH.md) — model
  choices, hardware, timings, costs, failure modes, rights.
- **How to run it for real:** [`docs/RUNBOOK.md`](docs/RUNBOOK.md)

## The idea in one paragraph

Lip-sync models get the attention, but sync is won earlier: Telugu takes ~1.2×
longer to say the same thing as English, so if you translate first and worry
about timing later, every sentence overruns and the video drifts. This pipeline
gives the translator a **syllable budget per line**, predicts each line's spoken
duration before synthesising it, and then fits the synthesised speech into the
original speech slots — stretching only within a band the ear cannot hear,
borrowing from pauses when it must, and **never starting a line before the
original mouth opened**, because audio that leads the picture is perceptually
three times worse than audio that lags it.

## Try it now (no GPU, no network, no API keys)

```bash
pip install pyyaml imageio-ffmpeg pytest
export PYTHONPATH=src

python scripts/make_sample_video.py                 # stand-in talking-head clip
python -m telugu_dub run --video samples/demo_source.mp4 \
    --config config/poc_offline.yaml \
    --transcript samples/demo_en.srt

pytest tests -q
```

Output lands in `runs/poc/`: `final_te.mp4`, `dub_te.srt`, `report.md`,
`qc.json`, `sync_report.json`.

The offline preset swaps in stub providers (`srt` transcript, phrase-table
translation, procedural TTS, no lip-sync) so the whole pipeline — timing,
mixing, muxing, QC — is exercised without weights. Real runs use
`config/default.yaml`.

## Real run

```bash
./scripts/fetch_youtube.sh "https://www.youtube.com/watch?v=<ID>" 120 180
python scripts/calibrate_rate.py --config config/default.yaml --write
python -m telugu_dub run --video samples/source.mp4 \
    --config config/default.yaml --transcript samples/source.en.srt
```

## Pipeline

| # | Stage | Default provider | Alternatives |
|---|---|---|---|
| 1 | ingest | yt-dlp + ffmpeg | local file |
| 2 | ASR | faster-whisper large-v3 | WhisperX (+diarization), official captions |
| 3 | translate | LLM with syllable budget | IndicTrans2, mock |
| 4 | TTS | IndicF5 (voice cloning) | Sarvam Bulbul, ElevenLabs, mock |
| 5 | **align** | isochrony fitter (this repo) | — |
| 6 | render | ffmpeg atempo + timeline assembly | — |
| 7 | mix | Demucs stems | side-chain ducking, none |
| 8 | lip-sync | LatentSync 1.6 | MuseTalk, Wav2Lip, sync.so, none |
| 9 | mux | ffmpeg | + burned-in Telugu subtitles |

Every stage is cached in the run directory, so you iterate on the cheap stages
and run lip-sync — 90 % of the compute — only once the audio is final.

## Quality gates

`qc.py` grades every segment against ITU-R BT.1359-1 onset tolerances
(−45 ms audio-lead / +125 ms audio-lag) and reports tempo pressure and overflow.
A run that does not pass QC should not be sent to lip-sync.

```
QC REVIEW: 83.3% of 6 segments inside the ITU-R BT.1359 window (-45 ms .. +125 ms)
  mean |drift| 0.05s, p95 0.3s, max 0.3s
  50.0% needed tempo above the comfortable band, 16.7% overflowed their slot
  segments to fix: [1] (shorten the Telugu, or split the slot)
```

## Layout

```
src/telugu_dub/
  align.py            isochrony fitting — where sync is decided
  telugu_prosody.py   Telugu syllable counting + duration prediction
  qc.py               AV-sync grading against broadcast thresholds
  pipeline.py         staged, cached, resumable orchestrator
  media.py            ffmpeg helpers + dependency-free WAV timeline assembly
  stages/             ingest, asr, translate, tts, render, lipsync, mux
config/               default (GPU) and poc_offline presets, glossary
scripts/              sample generator, YouTube fetcher, rate calibration
docs/                 RESEARCH.md, RUNBOOK.md
tests/                24 tests incl. an end-to-end offline run
```

## Rights

The source footage is copyrighted, and cloning a real person's voice needs that
person's consent. Label output as an AI translation. See
[`docs/RESEARCH.md` §7](docs/RESEARCH.md#7-rights-and-consent).
