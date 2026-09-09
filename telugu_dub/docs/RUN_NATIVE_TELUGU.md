# English audio → native Telugu, background untouched

No GPU required. No lip-sync. Two API keys, about 20 minutes of setup, then one
command per file.

---

## What you are building

```
your_talk.mp3
   │
   ├─ Demucs ──► background.wav   kept, untouched, mixed back at the end
   │             vocals.wav       English speech, isolated
   │                 │
   │                 ▼
   │          Whisper large-v3    transcript + timings (reads clean speech)
   │                 │
   │                 ▼
   │          Claude              rewrite as spoken Telugu, one syllable
   │                 │            budget per line
   │                 ▼
   │          Sarvam Bulbul v3    native Telugu voice
   │                 │
   │                 ▼
   │          isochrony fit       lands on his original timings
   │                 │
   └─────────────────┴──────────► final_te.mp3 + Telugu .srt + QC report
```

---

## 1 · Get the two keys

### Sarvam (the Telugu voice) — ₹100 free, no card

1. Go to **https://dashboard.sarvam.ai** and sign up.
2. **API Keys → Create**. Copy it.
3. New accounts get ₹100 of credits. Bulbul v3 is ₹30 per 10,000 characters,
   so that is roughly **33,000 characters ≈ three ten-minute talks, free**.

```bash
export SARVAM_API_KEY=sk_xxxxxxxx
```

### Anthropic (the translation)

1. **https://console.anthropic.com** → API Keys → Create Key.
2. Add a few dollars of credit. Translation costs cents per talk.

```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

Put both in `~/.bashrc` (or `~/.zshrc`) so you do not re-export them every
session.

---

## 2 · Install

```bash
git clone <this repo> && cd telugu_dub
python3 -m venv .venv && source .venv/bin/activate
export PYTHONPATH=src

pip install -r requirements.txt
pip install faster-whisper anthropic requests demucs torch
```

`ffmpeg` is worth having system-wide:

| OS | Command |
|---|---|
| Ubuntu/Debian | `sudo apt-get install -y ffmpeg` |
| macOS | `brew install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` |

**CPU is fine.** Demucs on CPU runs roughly 1–2× real time (a ten-minute talk
takes ten to twenty minutes) and Whisper `large-v3` on CPU is slow — see step 6
if you want it faster.

Check everything before you spend an hour finding out:

```bash
python -m telugu_dub doctor --config config/native_telugu.yaml --mode audio
```

It prints one line per requirement and the exact install command for whatever
is missing. Fix every `FAIL`; `warn` lines are fine.

---

## 3 · Test the voice first (5 minutes, free)

Do not dub a whole talk before you know you like the voice.

```bash
python scripts/tts_shootout.py --providers sarvam --voice sarvam=shubh
```

Listen to `runs/shootout/sarvam/`. Try other speakers — `ritu`, `priya`,
`neha`, `kavya` (female), `aditya`, `rahul`, `rohan` (male). Names are
lowercase. Pick one and put it in the config:

```yaml
tts:
  voice: shubh      # ← your pick
```

Those default test sentences are chosen to expose the failure modes: retroflex
consonants (ట ఠ డ ఢ ణ ళ), long vowels, a question contour, and a mid-sentence
English loanword. If a voice survives those six lines it will survive your talk.

---

## 4 · Calibrate the voice (once, 2 minutes)

Every timing decision downstream derives from one number: how fast your chosen
Telugu voice actually speaks. Measure it rather than assume it.

```bash
python scripts/calibrate_rate.py --config config/native_telugu.yaml --write
```

Skipping this is the most common reason a first run comes out with every line
overrunning.

---

## 5 · Dub it

Start with a **2–3 minute excerpt**, not the whole file:

```bash
ffmpeg -ss 120 -t 180 -i talk.mp3 -c copy excerpt.mp3

python -m telugu_dub run --audio excerpt.mp3 \
    --config config/native_telugu.yaml --mode audio
```

Output lands in `runs/native_te/`:

| File | What it is |
|---|---|
| `final_te.mp3` | Telugu voice over your original background |
| `dub_track.wav` | the Telugu voice alone |
| `stems/…/no_vocals.wav` | your background, untouched — check it |
| `dub_te.srt` | the full Telugu script |
| `qc.json` | per-segment sync grading |
| `report.md` | stage timings and length pressure |
| `manifest.json` | every segment: English, Telugu, timing, tempo |

### Read the QC before you listen

```
QC PASS: 98.7% of 75 segments inside the ITU-R BT.1359 window (-45 ms .. +125 ms)
  mean |drift| 0.014s, p95 0.06s, max 0.168s
  4.0% needed tempo above the comfortable band, 2.7% overflowed their slot
```

| If you see | Meaning | Fix |
|---|---|---|
| high `overflow_pct` | Telugu running past its slots | redo step 4, or tighten `prosody.headroom` to 0.9 |
| high `over_speed_pct` | speech being compressed to fit | same |
| named `failing_segments` | those lines are audibly out of sync | step 7 |

---

## 6 · If it is too slow

| Symptom | Fix |
|---|---|
| Whisper crawling on CPU | `--set asr.model=medium.en` (or `small.en`) |
| Demucs taking forever | it is once per file and cached — the second run reuses the stems |
| You have an NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu121`; both steps become minutes |
| No patience for either | `--set mix.separator=ducking` skips Demucs entirely, at the cost of a faint English bed |

---

## 7 · Fixing individual lines

`runs/native_te/manifest.json` holds every segment with its English and Telugu.
Edit the `text_tgt` of anything you dislike, then re-run only what depends on it:

```bash
python -m telugu_dub run --audio excerpt.mp3 \
    --config config/native_telugu.yaml --mode audio \
    --force tts align render mix
```

ASR, separation and translation stay cached — this takes seconds.

To keep those corrections permanently, save them and switch the translator to
the reviewed file so re-runs never regenerate them:

```bash
python -c "import json;m=json.load(open('runs/native_te/manifest.json'));\
json.dump({'translations':{str(s['id']):s['text_tgt'] for s in m['segments']}},\
open('reviewed_te.json','w'),ensure_ascii=False,indent=1)"

python -m telugu_dub run --audio talk.mp3 --config config/native_telugu.yaml \
    --mode audio --set translate.provider=file translate.model=reviewed_te.json
```

---

## 8 · Then the full file

```bash
python -m telugu_dub run --audio talk.mp3 \
    --config config/native_telugu.yaml --mode audio
```

### Cost per ten-minute talk

| | |
|---|---|
| Sarvam Bulbul v3 | ~₹36 (~$0.43) |
| Claude translation | ~$0.20 |
| Whisper + Demucs | free, your CPU |
| **Total** | **under $1** |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARNING: demucs is not installed` | it fell back to ducking; the English voice will stay faintly audible | `pip install demucs torch` |
| Background sounds thin | you ducked instead of separating | check the log actually says "vocals and background separated" |
| Background voices disappeared | Demucs treats *all* human voice as vocals — audience and second speakers go with it | `--set mix.separator=ducking` for audience-heavy recordings |
| Telugu sounds flat | wrong speaker for the content | try another `tts.voice`; step 3 |
| `401` from Sarvam | key not exported, or spent credits | `echo $SARVAM_API_KEY`; check the dashboard |
| Speaker name rejected | names are case-sensitive lowercase | `shubh` not `Shubh` |
| Everything overruns | uncalibrated rate | step 4 |

## What to check before you publish

1. A Telugu speaker listens end to end. Nothing substitutes for this.
2. Scrub to each `hard_compressed` segment — those are where it sounds rushed.
3. Listen to `stems/…/no_vocals.wav` on its own and confirm the background is
   what you expected to keep.
4. Label the output as an AI translation.
