# The six use cases

Two independent switches: **what you output** (`--mode`) and **whose voice**
(`--voice`). Same pipeline, later stages switched off.

```
ingest → asr → translate → tts → align → render → mix ─┬──────────────► --mode audio
                                                       ├─ mux ────────► --mode video
                                                       └─ lipsync ─mux► --mode video-lipsync
                                          ▲
                                          └── --voice preset (stock) | clone (his voice)
```

## The matrix

| # | Mode | Voice | Output | GPU | Keys | Time per video-minute | Consent needed |
|---|---|---|---|---|---|---|---|
| 1 | `audio` | `preset` | `final_te.mp3` | no | no | **~40 s** | no |
| 2 | `audio` | `clone` | `final_te.mp3` | yes* | maybe | ~1.5 min | **yes** |
| 3 | `video` | `preset` | `final_te.mp4` | no | no | **~1 min** | no |
| 4 | `video` | `clone` | `final_te.mp4` | yes* | maybe | ~2 min | **yes** |
| 5 | `video-lipsync` | `preset` | `final_te.mp4` | **yes** | no | 10–25 min | no |
| 6 | `video-lipsync` | `clone` | `final_te.mp4` | **yes** | maybe | 10–25 min | **yes** |

\* IndicF5 runs on CPU, roughly 5–10× slower. ElevenLabs/Sarvam cloning needs no
GPU but does need an API key.

**Cases 1 and 3 are what you can run today**, on a laptop, for free. Everything
in this repo's default free preset targets them.

---

## 1 & 3 — preset voice (what to build first)

```bash
python -m telugu_dub run --video clip.mp4 --config config/free_cpu.yaml \
    --mode audio --voice preset      # Telugu audio track
python -m telugu_dub run --video clip.mp4 --config config/free_cpu.yaml \
    --mode video --voice preset      # original picture + Telugu dub
```

Stack: faster-whisper (CPU) → keyless Google translate → edge-tts
`te-IN-MohanNeural` → isochrony fit → ducked mix → mux.

What you get: a watchable Telugu version where the timing is right and the
meaning is right, in a stock Telugu male voice. The lips still say English, the
same as most television dubbing.

What is *not* good enough to publish, and the upgrade order:

1. **Translation.** Google-free is literal and misses register — it will render
   discourse metaphor flatly and mangle yogic terms. Switch
   `translate.provider: llm` (needs `ANTHROPIC_API_KEY`). Biggest quality jump
   available, costs cents per minute.
2. **ASR.** `small` on CPU mishears names and Sanskrit terms. Use `large-v3` on
   a GPU, or better, the official English captions if the video has them.
3. **Voice.** edge-tts is clear but flat and has no control over emphasis.
   Sarvam Bulbul is the best-measured Telugu voice; ElevenLabs gives the most
   prosody control.

## 2 & 4 — cloned voice

Same pipeline, `--voice clone`, plus a reference recording:

```yaml
tts:
  provider: indicf5
  ref_audio: samples/reference_voice.wav   # 5-15 s, clean, no music
  ref_text: "the exact transcript of that clip"
```

Read [RESEARCH.md §7](RESEARCH.md#7-rights-and-consent) first. Cloning a
recognisable person's voice without their permission is the one step in this
project that can go badly wrong, and it is not a technical problem.

Note that a cloned voice raises the stakes on everything else: a flat stock
voice reads as "this is a dub", while his own voice saying a mistranslated
sentence reads as "he said that".

## 5 & 6 — lip sync

Needs a CUDA GPU (18 GB for LatentSync 1.6, less for MuseTalk) and a
third-party checkout — see [RUNBOOK.md §2](RUNBOOK.md). This is 90 % of the
compute in the whole project, which is why the pipeline caches every earlier
stage: get the audio right in mode 3, then switch on mode 5 and only lip-sync
runs.

```bash
python -m telugu_dub run --video clip.mp4 --config config/default.yaml \
    --mode video --until mix          # iterate here, seconds per run
python -m telugu_dub run --video clip.mp4 --config config/default.yaml \
    --mode video-lipsync              # then this, once
```

Do not run mode 5 until QC passes in mode 3. Lip-syncing badly timed audio just
produces an expensive video of a mouth moving at the wrong moments.

---

## Choosing

- **Testing whether the translation is any good** → mode 1 (`audio`). Fastest
  loop; you can listen at 2× while reading the SRT.
- **Showing someone what this project does** → mode 3 (`video`). It is
  convincing, it costs nothing, and it runs on the machine you have.
- **A real product for Telugu viewers** → mode 3 is genuinely shippable with an
  LLM translator and a good TTS voice. Most dubbed content people watch is
  exactly this.
- **The "wow" demo** → mode 5/6, on a 30-second close-up clip, once, on a
  rented GPU.
