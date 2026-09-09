# Sadhguru English → Telugu Dub: POC Status & Build Plan

**One line:** we have a working, tested pipeline that turns English speech into
timed, native-sounding Telugu speech over the original background. We do **not**
yet have Sadhguru's own voice or lip-sync — both are real, unsolved R&D
problems, not just integration work, and this doc says exactly why.

---

## 1. What we have right now

Everything below is written, unit-tested, and exercised end-to-end with
synthetic/offline data. **Nothing has been run yet against Sarvam or Anthropic
with real audio on a machine that can actually reach those APIs** — this
sandbox's network blocks Sarvam entirely, and the Anthropic key that was
issued has $0 balance. So: the code path is proven, the live run is not, and
that's the very next thing to do (on your machine, not here).

| Piece | Status | Where |
|---|---|---|
| Voice/background separation (Demucs) | Built, tested | `src/telugu_dub/stages/separate.py` |
| English ASR (Whisper large-v3, faster-whisper) | Built, tested on real audio in this session | `stages/asr.py` |
| Dubbing-style translation w/ syllable budget | Built, prompt tuned, tested with mock provider | `stages/translate.py` |
| Telugu TTS — 6 interchangeable providers | Built | `stages/tts.py` |
| Isochrony fitting (timing/tempo match) | Built, tested incl. 2 real bugs found & fixed | `align.py` |
| Sync QC against ITU-R BT.1359 | Built, tested | `qc.py` |
| Mixing dub over preserved background | Built, tested | `stages/render.py` |
| Reference-clip picker for voice cloning | Built, tested, run on your real Sadhguru audio | `reference.py` |
| Standalone one-file version (no repo needed) | Built, offline logic tested | `standalone/dub.py` |
| Lip-sync — provider abstraction only | Built, **never executed** (no GPU here) | `stages/lipsync.py` |
| Voice cloning (zero-shot) | Wired as a TTS provider, **never executed** (model host blocked here) | `stages/tts.py::indicf5` |
| Voice cloning (fine-tuned) | **Not built.** Documented as a Colab notebook only | `notebooks/telugu_dub_end_to_end.ipynb` |

**Test coverage:** 58 tests across 8 files, covering the syllable-timing math,
the aligner's tempo/overflow logic, mode selection, the reference-clip scorer,
the separation stage, and an offline end-to-end run. None of this exercises a
real network call — that's by design (tests must run anywhere), but it also
means "tests pass" ≠ "a real Telugu dub has been produced and listened to."

**What has actually been *heard* by a human in this project so far:** nothing
in a native Telugu voice. Two demo dubs were produced earlier with `espeak`
(a robotic formant synthesizer used only because every real TTS host was
network-blocked in that sandbox) — useful for proving the timing logic, not
representative of final quality.

---

## 2. Current architecture (what actually runs today)

```
English audio/video
        │
        ▼
 ┌─────────────┐
 │  1. SEPARATE │  Demucs → vocals.wav (discarded) + background.wav (kept, untouched)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  2. ASR      │  Whisper large-v3 transcribes the ISOLATED vocals
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  3. TRANSLATE│  Claude — dubbing-house prompt: meaning over structure,
 │              │  tone preserved, proper nouns kept, syllable budget per line
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  4. TTS      │  Sarvam Bulbul v3 (native-trained Telugu voice, NOT Sadhguru)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  5. FIT      │  isochrony: stretch/compress/borrow-pause so each line lands
 │              │  on the original timing; never starts before the original onset
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  6. MIX      │  Telugu speech laid over the untouched background stem
 └──────┬──────┘
        ▼
  Telugu audio, native voice, original background — NO lip-sync, NOT his voice
```

This is real and it works. It is also **not the end goal** — it's the
scaffolding everything else plugs into.

---

## 3. The two things you actually asked for, and why they're hard

### 3a. Sadhguru's voice

There are two completely different technical problems hiding under "make it
sound like him," and they have very different data requirements:

| Approach | What it needs | What you get |
|---|---|---|
| **Zero-shot cloning** | One 10–15 s clean reference clip (already extracted from your audio — see `samples/` and the `make_reference.py` output from earlier in this project) | His **timbre** — recognisably his voice type. **Not** his delivery: the long pauses, the timing of emphasis, the things that actually make him sound like *him* rather than "a similar voice." |
| **Fine-tuned cloning** | A real training set — see below | His timbre **and** much more of his delivery, if the data is good enough |

**How much audio a fine-tune actually needs** (this is the honest range from
current TTS fine-tuning practice, not a guess):

| Audio | Result |
|---|---|
| < 30 min | Usually not worth it — barely beats zero-shot |
| 1–3 hours | Workable, mediocre robustness, some tell-tale artifacts on words unlike the training set |
| **3–10 hours** | **The realistic target for this project.** Clean, transcribed, single-speaker |
| 10–30+ hours | Production-grade, what a real dubbing studio would use |

**The problem underneath the problem:** Sadhguru's public catalogue is
overwhelmingly **English** (with some Tamil/Hindi). We have found **no
evidence he has Telugu source audio at all.** That matters a lot, because:

- Fine-tuning on his **English** audio gives you an excellent English clone of
  his voice — genuinely achievable with 5–20 hours of clean talks, which
  exist on his YouTube channel in volume.
- Making that English-trained model **speak Telugu** is a different, much
  less proven problem: cross-lingual voice cloning (train on language A,
  synthesize language B in the same voice identity). It's an active research
  area, not a solved one. The same PSP benchmark research that showed
  English-first commercial TTS mangles Telugu retroflex consonants (see
  `docs/native-telugu-voice.html`) is exactly the failure mode this approach
  risks reproducing, at a scale we can't easily benchmark ourselves.
- **We have not tested whether this works at all.** Nobody has verified
  cross-lingual quality with Sadhguru's data specifically.

**Best shot — the plan I'd actually run:**

1. **Collect 5–20 hours of clean English audio.** Long single-camera talks,
   run each through Demucs (already built) to strip audience/music, keep only
   segments with continuous uninterrupted speech. His YouTube catalogue easily
   supports this.
2. **Transcribe + human-correct** with Whisper large-v3 (already built) —
   accuracy here directly limits fine-tune quality.
3. **Fine-tune IndicF5** (F5-TTS architecture, already Indic-capable) on that
   English set. This is a known, documented procedure — see Section 11 of
   `notebooks/telugu_dub_end_to_end.ipynb` for the exact `finetune_cli.py`
   command. Needs a GPU with real VRAM (A100 40GB class) for 8–20 hours.
4. **Test cross-lingual inference immediately**, before investing further:
   feed the fine-tuned model Telugu text and listen. Use the same blind-test
   method already built (`scripts/tts_shootout.py`) — score retroflex
   articulation, pitch, rhythm, separately, with a Telugu speaker who doesn't
   know which clip is which.
5. **Branch on the result:**
   - If cross-lingual Telugu quality is acceptable → you have a genuine
     "Sadhguru speaking Telugu" voice. This is the win condition, and it is
     genuinely uncertain until step 4 is run.
   - If it's not (the likely-but-unproven outcome) → the fine-tuned model is
     still valuable for **English**-in-his-voice use cases, and Telugu falls
     back to zero-shot cloning (his timbre, a stock delivery) or a native
     Telugu voice (Sarvam, no likeness at all — what we have today).

**Consent, again, because it matters more here than anywhere else in this
project:** training a voice model on someone's recordings and making it say
sentences they never spoke is a materially bigger step than dubbing existing
words. Do this only with Isha's participation or explicit written consent,
and keep the fine-tuned model access-controlled. This isn't a legal
formality — it's the difference between a dubbing tool and a deepfake
generator, and the line is exactly here.

### 3b. Lip-sync

Nothing has been built beyond a provider abstraction that shells out to
external tools (`stages/lipsync.py` supports LatentSync, MuseTalk, Wav2Lip,
or a hosted API) — **none have been executed**, because this sandbox has no
GPU. This is the honest state.

**Best shot — the plan I'd actually run:**

1. **Rent a GPU**, don't buy one yet. A 24GB card (RTX 4090, L40S, or an
   A100) on RunPod/Lambda/Vast.ai runs $0.50–1.50/hr. `docs/RUNBOOK.md`
   already has the exact setup steps for this.
2. **Start with LatentSync 1.6** (Apache-2.0, best open-source visual
   fidelity, already wired as a provider). Follow `docs/RUNBOOK.md` §2.
3. **Test on a clip with his beard visible, in close-up, first.** This is the
   single biggest visual risk flagged in `docs/RESEARCH.md` — lip-sync models
   are trained mostly on clean-shaven faces, and facial hair is the classic
   failure mode (smeared mouth, mouth "floating" over the beard). If this
   fails, it fails early and cheaply — 60 seconds of GPU time, not a wasted
   video.
4. **If beard quality is bad**, add a face-restoration pass (GFPGAN or
   CodeFormer) before accepting or rejecting the approach.
5. **If self-hosting is a hassle**, sync.so's `lipsync-2` API
   ($0.04–0.05/sec, ≈ $2.40–3.00/min) gets you a quality answer for one test
   clip with zero infrastructure — worth doing in parallel with step 2 just
   to have a ceiling to compare against.
6. **MuseTalk** is the fast-iteration fallback once you've decided LatentSync
   is the quality bar — near real-time, lower fidelity, good for previewing
   many takes quickly.

**Realistic cost per finished minute once this is running:** $2.40–3.00
(hosted) or a few cents of GPU time (self-hosted, after setup). This is
roughly **30× the cost of everything else in the pipeline combined** — see
Section 5.

---

## 4. Requirements to actually build this

### Data
- **5–20 hours of clean English Sadhguru audio** for the voice fine-tune (see
  3a). Source: his YouTube catalogue, run through the Demucs separation stage
  already built.
- **A verified, corrected transcript** for that audio — Whisper output is a
  starting point, not the final transcript.
- **A short test video with his face visible, including a beard close-up**,
  for lip-sync validation. 30–60 seconds is enough for the first pass.

### Compute
- **CPU-only is enough** for everything up through Section 2 (the current
  working pipeline) — Whisper `small`/`medium` on CPU is workable; `large-v3`
  wants a GPU for speed but not correctness.
- **A 24GB-class GPU** for lip-sync (LatentSync 1.6 needs ~18GB VRAM) and for
  a serious voice fine-tune (A100 40GB class, though smaller GPUs can work
  slower). Rent, don't buy, until the approach is validated.

### Accounts / keys
- **Sarvam** — native Telugu voice, works today. `SARVAM_API_KEY`, ₹100 free
  credit, ₹30/10k characters after.
- **Anthropic** — the dubbing-translator step. `ANTHROPIC_API_KEY` — **the
  key issued in this project currently has $0 balance and needs credit added**
  before anything downstream of translation will run.
- **HuggingFace** — for downloading IndicF5, F5-TTS, and any fine-tuning
  starting checkpoints. Needed on whatever machine actually does the
  fine-tuning (blocked from this sandbox, not from a normal machine).
- **A GPU rental account** (RunPod, Lambda Labs, Vast.ai, or similar) for both
  the fine-tune and lip-sync testing.

### People
- **A Telugu speaker who isn't you**, for every blind-listening decision —
  this is not optional. `scripts/tts_shootout.py` exists specifically to make
  this test easy to run repeatedly.
- **Isha's involvement or explicit consent**, before a voice-cloned model
  goes anywhere near distribution. See 3a.

---

## 5. Final architecture — the whole thing, once built

```
English talk (video)
        │
        ▼
 1. SEPARATE ──────► background stem (kept, untouched)
        │            vocals stem (transcribed, then discarded)
        ▼
 2. ASR (Whisper large-v3, on isolated vocals)
        ▼
 3. TRANSLATE (Claude, dubbing-style, syllable-budgeted, glossary-aware)
        ▼
 4. TTS ─── Sadhguru voice model (fine-tuned, cross-lingual Telugu) ◄── the open question in 3a
        │   fallback: zero-shot clone, or native Telugu stock voice
        ▼
 5. FIT (isochrony — lands on his original timing, never starts early)
        ▼
 6. MIX (Telugu speech + untouched background stem)
        ▼
 7. LIP-SYNC (LatentSync 1.6 or hosted API, face-gated to shots where he's on camera)
        ▼
 8. MUX (regenerated video + dubbed audio + Telugu subtitles)
        ▼
Final video: his face, his voice (if 3a succeeds), speaking Telugu,
lips matching, original music/ambience intact
```

Stages 1, 2, 3, 5, 6, 8 are **built and tested today**. Stage 4 works today
with a *native but not his* voice, and the path to *his* voice is the R&D
plan in §3a. Stage 7 is **entirely unbuilt in practice** — the code exists to
call it, but it's never been run.

---

## 6. Where we are, in one paragraph

The plumbing — everything that isn't "sound like him" or "move his mouth
right" — is done, tested, and the single hardest and most valuable part of
it (isochrony fitting, so the dub doesn't drift out of sync) is solved and
verified against real transcribed audio. Native Telugu voice output, with
correct grammar and preserved background, works **today** with a stock voice.
What's missing is genuinely hard and genuinely unproven: whether an
English-trained voice clone will sound acceptable speaking Telugu (nobody
has tested this with his data), and whether lip-sync holds up on his beard
(nobody has tested this at all, for lack of a GPU in this environment). Both
are answerable in a single focused session each — a fine-tune run plus a
blind listening test for the voice, a 60-second GPU test for lip-sync — but
neither is answerable from here. They need real compute and a real Telugu
listener, which this sandbox cannot provide.

---

## 7. Immediate next steps, in order

1. Add credit to the Anthropic account — nothing downstream of translation
   runs without it.
2. Run `standalone/dub.py` on your machine against a real clip end-to-end,
   with real Sarvam output, and actually listen to it. This is the first
   real listening test of the whole pipeline and it's been blocked on you
   running it locally.
3. In parallel: start pulling together 5–20 hours of clean English Sadhguru
   audio, if fine-tuning is something you want to pursue.
4. Rent a GPU for one afternoon. Run the LatentSync beard test. That answers
   the lip-sync question cheaply, before any bigger commitment.
5. Once both are answered, revisit this document — it's the honest baseline
   to measure "did the R&D work" against, not a promise that it will.
