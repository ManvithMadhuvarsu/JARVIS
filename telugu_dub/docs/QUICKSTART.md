# Quickstart — dub your own clip in about ten minutes

No GPU. No API keys. No HuggingFace login. Works on Linux, macOS and Windows
(WSL or native).

---

## 1. Install

```bash
git clone <this repo> && cd telugu_dub
./scripts/setup.sh              # creates .venv and installs the free stack
source .venv/bin/activate
export PYTHONPATH=src           # Windows PowerShell: $env:PYTHONPATH="src"
```

`ffmpeg` is the one thing worth installing system-wide:

| OS | Command |
|---|---|
| Ubuntu/Debian | `sudo apt-get install -y ffmpeg fonts-noto-telugu` |
| macOS | `brew install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` |

(If you skip it, the bundled `imageio-ffmpeg` build is used automatically — it
works, but it cannot burn in subtitles.)

## 2. Check the machine

```bash
python -m telugu_dub doctor --config config/free_cpu.yaml --mode video
```

It prints one line per requirement and the exact `pip install` for anything
missing. Fix the `FAIL` lines; `warn` lines are fine.

## 3. Get a clip

**Start with 60–90 seconds.** Not the full talk. You will want three or four
iterations before you like the result, and a long video turns a two-minute loop
into an hour.

```bash
./scripts/fetch_youtube.sh "https://www.youtube.com/watch?v=VIDEO_ID" 120 210
```

That downloads seconds 120–210 as `samples/source.mp4`, and grabs the official
English captions as `samples/source.en.srt` when they exist.

Pick a clip where the speaker is **talking to camera for most of it** — a
single continuous take. Avoid clips with heavy background music (it makes the
mix harder) or rapid cuts to the audience.

No video handy? Generate a synthetic one to test the plumbing:

```bash
python scripts/make_sample_video.py     # writes samples/demo_source.mp4
```

## 4. Run it — audio first

Audio mode is the fastest loop and it is where you judge the translation.

```bash
python -m telugu_dub run \
    --video samples/source.mp4 \
    --config config/free_cpu.yaml \
    --mode audio --voice preset
```

First run downloads the Whisper model (~500 MB for `small`) once.

Output in `runs/free/`:

| File | What it is |
|---|---|
| `final_te.mp3` | the Telugu audio track |
| `dub_te.srt` | Telugu subtitles on the fitted timings |
| `report.md` | stage timings, length pressure, QC |
| `qc.json` | per-segment sync grading |
| `manifest.json` | every segment: English, Telugu, timing, tempo |

## 5. Read the QC before you look at anything else

```
QC PASS: 96.2% of 53 segments inside the ITU-R BT.1359 window (-45 ms .. +125 ms)
  mean |drift| 0.02s, p95 0.06s, max 0.11s
  9.4% needed tempo above the comfortable band, 0.0% overflowed their slot
```

| If you see | It means | Do this |
|---|---|---|
| high `overflow_pct` | Telugu is running past its slots | calibrate the rate (step 7), or switch to `translate.provider: llm` |
| high `over_speed_pct` | speech is being compressed to fit | same |
| `failing_segments` | those lines are audibly out of sync | edit them (step 8) |

## 6. Then the video

```bash
python -m telugu_dub run --video samples/source.mp4 \
    --config config/free_cpu.yaml --mode video --voice preset
```

ASR, translation and TTS are cached from the audio run, so this only does the
mux — a couple of seconds.

## 7. Calibrate the voice (once)

Everything downstream depends on how fast your chosen Telugu voice actually
speaks. Measure it instead of guessing:

```bash
python scripts/calibrate_rate.py --config config/free_cpu.yaml --write
```

Then re-run with `--force translate tts align render mix`.

## 8. Fixing a bad line

`runs/free/manifest.json` holds every segment. Edit the `text_tgt` of the one
you dislike, then re-run only what depends on it:

```bash
python -m telugu_dub run --video samples/source.mp4 \
    --config config/free_cpu.yaml --mode audio \
    --force tts align render mix
```

## 9. Upgrading quality

```bash
pip install anthropic && export ANTHROPIC_API_KEY=...
python -m telugu_dub run --video samples/source.mp4 \
    --config config/free_cpu.yaml --mode video \
    --set translate.provider=llm asr.model=medium \
    --force translate tts align render mix
```

The LLM translator is the single biggest jump — it obeys a syllable budget per
line, keeps the yogic glossary, and reads neighbouring lines for context.

## 10. Lip sync

Different league: needs a CUDA GPU and a model checkout. See
[RUNBOOK.md](RUNBOOK.md) §2 and [USE_CASES.md](USE_CASES.md) cases 5–6. Get the
audio right first — lip-syncing badly timed audio wastes hours of GPU time.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `edge-tts returned no audio` | network blocks Microsoft's TTS endpoint, or a bad voice name | `python -m telugu_dub voices`; or `--set tts.provider=gtts` |
| translation dies partway | free Google endpoint rate-limited you | rerun (stages are cached); or switch to `llm` |
| Whisper is very slow | large model on CPU | `--set asr.model=base`, or pass `--transcript` with the YouTube captions |
| Telugu subtitles show as boxes | no Telugu font | `apt install fonts-noto-telugu`, or keep `burn_subtitles: false` |
| everything overflows its slot | uncalibrated speaking rate | step 7 |
| `ffmpeg not found` | not installed | see step 1, or `pip install imageio-ffmpeg` |
| dub drowns the original music | ducking too gentle | `--set mix.background_gain_db=-14` |
