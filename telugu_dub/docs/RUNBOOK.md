# Runbook — GPU setup for lip-sync and cloned voices

This is the heavyweight path: a GPU box, real weights, lip-sync.

If you only want a Telugu dub over the original picture — use cases 1-4, no GPU,
no API keys — you do not need any of this. Go to
[QUICKSTART.md](QUICKSTART.md) instead; it takes about ten minutes.

---

## 1. Machine

- Linux, NVIDIA GPU with **24 GB VRAM** (RTX 3090/4090, L40S, A100)
- CUDA 12.x, Python 3.10+
- `ffmpeg` **with libass** (subtitle burn-in) and `fonts-noto-telugu`
- ~10 GB free disk per hour of 720p footage

```bash
sudo apt-get install -y ffmpeg fonts-noto-telugu
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pipeline core
pip install -r requirements-gpu.txt    # ASR / TTS / separation
```

## 2. Lip-sync models (each in its own venv)

These repos pin mutually incompatible torch versions. Do not fight it — give
each its own environment; the pipeline shells out to them.

```bash
mkdir -p third_party && cd third_party

git clone https://github.com/bytedance/LatentSync
cd LatentSync
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# weights: follow the repo's checkpoint instructions (HF: ByteDance/LatentSync-1.6)
cd ..

git clone https://github.com/TMElyralab/MuseTalk      # fast iteration
cd MuseTalk && python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Point the config at whichever you use:

```yaml
lipsync:
  provider: latentsync
  repo_path: third_party/LatentSync
  checkpoint: third_party/LatentSync/checkpoints/latentsync_unet.pt
```

## 3. Keys (only for hosted providers)

```bash
export ANTHROPIC_API_KEY=...   # translate.provider: llm
export SARVAM_API_KEY=...      # tts.provider: sarvam
export ELEVENLABS_API_KEY=...  # tts.provider: elevenlabs
export SYNC_API_KEY=...        # lipsync.provider: sync_api
```

## 4. Get the clip

Start with 60 seconds. Always. A full talk is an overnight job and you will
want three iterations before you like the result.

```bash
./scripts/fetch_youtube.sh "https://www.youtube.com/watch?v=<ID>" 120 180
```

That also pulls official English captions when they exist — free, and better
than any ASR model. Pass the `.srt` with `--transcript` to skip stage 2.

Pick a clip where the speaker is **on camera, frontal, and talking** for most of
its length. A clip that cuts to the audience teaches you nothing about lip-sync
quality.

## 5. Calibrate the voice — do this once per voice

Everything downstream inherits one constant: how many Telugu syllables per
second your TTS voice actually speaks.

```bash
python scripts/calibrate_rate.py --config config/default.yaml --write
```

Skipping this is the most common reason a first run comes out with every line
overflowing.

## 6. Prepare the reference voice (only if cloning)

IndicF5 wants a 5–15 s clip plus its exact transcript:

```bash
ffmpeg -ss 00:01:12 -t 9 -i samples/source.mp4 -vn -ac 1 -ar 24000 \
       samples/reference_voice.wav
```

Clean speech, no music, no applause, neutral tone — the model copies the
prosody of the prompt. Then set `tts.ref_audio` and `tts.ref_text`.

Read `docs/RESEARCH.md` §7 before cloning a real person's voice.

## 7. Run it

Audio first — it is fast, and it is where sync is decided:

```bash
python -m telugu_dub run \
  --video samples/source.mp4 \
  --config config/default.yaml \
  --transcript samples/source.en.srt \
  --until mix
```

Read `runs/default/report.md` and `runs/default/qc.json`. **Do not proceed to
lip-sync until QC passes.** If it does not:

| QC says | Do this |
|---|---|
| segments overflowing | tighten `prosody.headroom` to 0.9, or hand-edit those lines |
| many segments above 1.15× | your speaking rate is miscalibrated — rerun §5 |
| `fail-early` segments | should be impossible; file a bug with `sync_report.json` |
| a few bad lines only | edit `manifest.json` text, then `--force tts align render mix` |

Then the expensive part:

```bash
python -m telugu_dub run --video samples/source.mp4 \
  --config config/default.yaml --transcript samples/source.en.srt
```

Cached stages are skipped, so this only runs lip-sync and mux.

## 8. Iterating on one bad line

```bash
# edit runs/default/manifest.json -> segments[N].text_tgt
python -m telugu_dub run --video samples/source.mp4 \
  --config config/default.yaml --force tts align render mix --until mix
```

Everything else stays cached.

## 9. What to check before showing anyone

1. A Telugu speaker watches it end to end. Nothing substitutes for this.
2. Scrub to each `hard_compressed` segment — those are where it sounds rushed.
3. Watch the mouth in close-ups, at 100 % zoom, at normal speed. Beard artefacts
   only show at full resolution.
4. Confirm the AI-translation label is present.
