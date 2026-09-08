# telugu-dub — English → Telugu dubbing with lip sync

Takes an English talk (the working example is a Sadhguru discourse) and produces
a Telugu version where the **voice and the mouth both match**: translated,
re-voiced, time-fitted to the original speech, and lip-synced.

- **Run it on your own clip in ten minutes:** [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
- **The six use cases:** [`docs/USE_CASES.md`](docs/USE_CASES.md)
- **What to build and why:** [`docs/RESEARCH.md`](docs/RESEARCH.md) — model
  choices, hardware, timings, costs, failure modes, rights.
- **GPU / lip-sync setup:** [`docs/RUNBOOK.md`](docs/RUNBOOK.md)

## Six use cases, one pipeline

| `--mode` | `--voice` | Output | GPU | Keys | Per video-minute |
|---|---|---|---|---|---|
| `audio` | `preset` | Telugu `.mp3` | no | no | ~40 s |
| `audio` | `clone` | Telugu `.mp3` | yes* | maybe | ~1.5 min |
| `video` | `preset` | `.mp4`, original picture | no | no | ~1 min |
| `video` | `clone` | `.mp4`, original picture | yes* | maybe | ~2 min |
| `video-lipsync` | `preset` | `.mp4`, mouth regenerated | yes | no | 10-25 min |
| `video-lipsync` | `clone` | `.mp4`, mouth regenerated | yes | maybe | 10-25 min |

The two `preset` rows without a GPU are what you can run **right now, free, on a
laptop** — free Google translation and Microsoft's free Telugu neural voice. The
cloned-voice rows need the speaker's consent; see
[RESEARCH.md §7](docs/RESEARCH.md#7-rights-and-consent).

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

## Dub your own clip (free, CPU, no keys)

```bash
./scripts/setup.sh && source .venv/bin/activate && export PYTHONPATH=src

python -m telugu_dub doctor --config config/free_cpu.yaml --mode video
./scripts/fetch_youtube.sh "https://www.youtube.com/watch?v=<ID>" 120 210

python -m telugu_dub run --video samples/source.mp4 \
    --config config/free_cpu.yaml --mode audio --voice preset   # listen first
python -m telugu_dub run --video samples/source.mp4 \
    --config config/free_cpu.yaml --mode video --voice preset   # then the video
```

`doctor` checks every requirement for the mode you picked and prints the exact
install command for anything missing. Start with a **60-90 second clip** — you
will iterate three or four times before you like the result.

Full walkthrough, including how to fix a bad line and what to upgrade first:
[`docs/QUICKSTART.md`](docs/QUICKSTART.md).

Run every mode your machine supports on one clip:

```bash
./scripts/run_all_modes.sh samples/source.mp4
```

## Try it with nothing installed at all

```bash
pip install pyyaml imageio-ffmpeg pytest && export PYTHONPATH=src
python scripts/make_sample_video.py                 # stand-in talking-head clip
python -m telugu_dub run --video samples/demo_source.mp4 \
    --config config/poc_offline.yaml --transcript samples/demo_en.srt
pytest tests -q
```

The offline preset swaps in stub providers (SRT transcript, phrase-table
translation, procedural TTS, no lip-sync) so the whole pipeline — timing,
mixing, muxing, QC — is exercised without weights or network.

## Pipeline

| # | Stage | Default provider | Alternatives |
|---|---|---|---|
| 1 | ingest | yt-dlp + ffmpeg | local file |
| 2 | ASR | faster-whisper (CPU `small` → GPU `large-v3`) | WhisperX (+diarization), official captions |
| 3 | translate | Google free (keyless) → LLM with syllable budget | IndicTrans2, Argos, mock |
| 4 | TTS | edge-tts (free Telugu neural) | Sarvam, ElevenLabs, IndicF5 (cloning), gTTS, mock |
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
  modes.py            the six use cases: mode x voice
  doctor.py           preflight — what is missing for the mode you picked
  align.py            isochrony fitting — where sync is decided
  telugu_prosody.py   Telugu syllable counting + duration prediction
  qc.py               AV-sync grading against broadcast thresholds
  pipeline.py         staged, cached, resumable orchestrator
  media.py            ffmpeg helpers + dependency-free WAV timeline assembly
  stages/             ingest, asr, translate, tts, render, lipsync, mux
config/               free_cpu (start here), default (GPU), poc_offline, glossary
scripts/              setup, YouTube fetcher, sample generator, rate calibration
docs/                 QUICKSTART, USE_CASES, RESEARCH, RUNBOOK
tests/                47 tests incl. end-to-end offline runs
```

## Commands

```
telugu-dub run       dub a video      --mode/--voice/--until/--force/--set
telugu-dub doctor    preflight check for a given mode
telugu-dub voices    list free Telugu edge-tts voices
telugu-dub modes     explain the use-case matrix
telugu-dub report    print stats for a finished run
```

## Rights

The source footage is copyrighted, and cloning a real person's voice needs that
person's consent. Label output as an AI translation. See
[`docs/RESEARCH.md` §7](docs/RESEARCH.md#7-rights-and-consent).
