# Sadhguru English → Telugu Dub: Plan

**Goal:** take an English talk and produce a Telugu version — his voice,
his mouth moving correctly, the original music/ambience intact underneath.

---

## 1. The pipeline

```
English talk (video)
        │
        ▼
 1. SEPARATE ──────► background stem (kept, untouched — music/ambience)
        │            vocals stem (used for transcription, then discarded)
        ▼
 2. TRANSCRIBE  (Whisper large-v3, on the isolated vocals)
        ▼
 3. TRANSLATE   (dubbing-style: meaning over literal structure, tone
        │        preserved, proper nouns and Sanskrit/yogic terms left
        │        alone, each line given a syllable budget so it can
        │        physically fit the original timing)
        ▼
 4. SPEAK IT    Sadhguru's own voice model, in Telugu
        ▼
 5. FIT TIMING  (isochrony: stretch/compress/borrow-pause so each line
        │        lands on his original timing; never starts before he
        │        actually opened his mouth — a dub that starts early
        │        reads as broken, one that lags slightly reads as natural)
        ▼
 6. MIX         (Telugu speech laid over the untouched background stem)
        ▼
 7. LIP-SYNC    (regenerate the mouth so it matches the Telugu audio)
        ▼
 8. FINAL VIDEO — his face, his voice, speaking Telugu, lips matching,
                  original music/ambience intact, Telugu subtitles included
```

Steps 1–3, 5, 6, 8 are ordinary, well-understood audio/video engineering.
Steps 4 and 7 — his actual voice, and lip-sync — are the two parts that
determine whether the result is convincing, and they're the focus of the
plan below.

**Timeline: 3–5 weeks end to end**, from a standing start to a full clip with
his voice and lip-sync, run with one or two people and the two tracks below
overlapped rather than done strictly in sequence. See §5 for the week-by-week
breakdown.

---

## 2. Step 4: Sadhguru's voice, in Telugu

Two approaches, in order of effort:

### Zero-shot cloning (fast path) — same day
Feed a TTS model a short clean reference clip of his voice (10–15 seconds)
alongside the Telugu text, and it imitates the *timbre* of that clip. No
training required — this can produce a result immediately, same day it's
attempted. What it will **not** capture: his delivery — the long pauses, the
way he lands emphasis, the timing that actually makes him sound like him
rather than "a similar voice." Good for a fast first pass, a ceiling on
realism.

### Fine-tuned voice model (the real target) — 2–3 weeks
Train a voice model on a substantial set of his own recordings, so it
captures timbre *and* delivery. What this takes:

| Training audio | Result |
|---|---|
| 1–3 hours | Workable, but rough on words unlike the training set |
| **5–20 hours** | **The target** — clean, single-speaker, accurately transcribed |
| 30+ hours | Studio-grade |

His public catalogue is overwhelmingly English, so the plan is:

1. **Collect 5–20 hours of clean English audio** (2–4 days elapsed, mostly
   waiting on downloads/organizing — light effort, can run alongside
   everything else) — long single-camera talks, background/audience
   stripped out (step 1 above, reused as a prep tool).
2. **Transcribe and hand-correct** the training transcripts (2–3 days —
   transcription itself is fast, the hand-correction pass is what takes the
   time) — this is the ceiling on fine-tune quality, worth getting right.
3. **Fine-tune a voice model** (F5-TTS-based architecture, Indic-capable)
   on that English audio, using the standard fine-tuning procedure for that
   model family (1–2 days — the training run itself is hours of GPU time,
   the rest is setup and re-running after issues). This step needs a GPU
   with real memory (A100-class).
4. **Generate Telugu with the fine-tuned model** and evaluate it by ear
   (1–2 days) — specifically with a native Telugu speaker, blind (they
   shouldn't know in advance which clip is the target voice), scoring
   pronunciation, pitch, and rhythm separately, since a voice can nail one
   and miss another.
5. **Decide based on that listening test** (same day as step 4):
   - If the Telugu output is convincing → that's the production voice.
   - If it isn't → the same fine-tuned model still gives a strong result for
     **English**-in-his-voice output, and Telugu falls back to zero-shot
     cloning (his timbre, without full delivery) as the interim answer while
     the cross-lingual approach is refined further.

Steps 1–2 can run in parallel with the audio-pipeline work in §5 step 1, so
they don't add to the critical path on their own — the fine-tune and test
(steps 3–4) are what actually gate the timeline, at roughly a week once the
training data is ready.

**Consent is part of the plan, not an afterthought.** Training a model that
can make him say sentences he never spoke is a bigger step than dubbing his
own words, and it should only proceed with Isha's participation or explicit
written consent, with the resulting model kept access-controlled.

---

## 3. Step 7: Lip-sync

**Timeline: 3–5 days** for the validation pass (steps 1–3 below); folding
lip-sync into a full video happens in §5 step 5, once a voice is ready to
pair it with.

Plan:

1. **Rent a GPU** for testing rather than provisioning permanent
   infrastructure up front — a 24GB-class card (RTX 4090 / L40S / A100) is
   enough, at low hourly cost from any GPU rental provider.
2. **Use an open lip-sync model with strong visual fidelity** (LatentSync-
   class) as the primary approach — regenerates the mouth region conditioned
   on the new audio.
3. **Validate on a close-up clip with his beard visible first**, before
   committing to a full video. Facial hair is the known hard case for this
   class of model, so it's the right thing to test cheaply and early rather
   than discover on a finished video.
4. **Add a face-restoration pass** if needed to sharpen the regenerated mouth
   region.
5. **Keep a hosted lip-sync API as a parallel comparison point** — useful for
   getting a quality ceiling to measure the self-hosted result against,
   without standing up infrastructure just to get that one data point.
6. **Only run lip-sync on shots where his face is actually on camera** —
   cutaways to the audience or B-roll pass through untouched.

Lip-sync is the most compute-intensive step in the pipeline by a wide
margin, so the plan is to prove it works on one short, hard clip before
spending time or money on a full video.

---

## 4. Requirements

### Data
- 5–20 hours of clean English Sadhguru audio, for the voice model.
- Accurate, hand-checked transcripts of that audio.
- A short test clip with his face visible, including a beard close-up, for
  the lip-sync validation pass.

### Compute
- Ordinary CPU is enough for transcription, translation, timing, and mixing.
- A 24GB-class GPU (rented, not owned) for the voice fine-tune and for
  lip-sync testing.

### Accounts
- A translation API account, for the dubbing-style translation step.
- A TTS account or self-hosted model, for the fast-path zero-shot voice and
  as a fallback if the Telugu fine-tune isn't ready yet.
- A GPU rental account, for the fine-tune and lip-sync runs.
- Optionally, a hosted lip-sync API account, as the comparison point in §3.

### People
- A native Telugu speaker for every listening decision — this determines
  whether the voice model and the translation actually land, and it's not a
  step that can be skipped or automated away.
- Sign-off from Isha (or explicit consent) before any voice-cloned model is
  used beyond internal testing.

---

## 5. Plan and timeline

**Total: 3–5 weeks from a standing start to one finished clip** — his voice
(or the best available fallback), lip-synced, over the original background.
That's a lean-to-medium POC timeline: fast enough to get a real answer
quickly, not a rushed weekend hack that skips the listening tests that
actually determine whether it's convincing.

Two tracks run in parallel rather than one strict sequence — the voice model
work does not wait on the audio pipeline, and vice versa:

```
Week    1          2          3          4          5
        │──────────│──────────│──────────│──────────│
Track A [ audio pipeline: translate → voice → fit → mix ]
        │──────────│
        (done — Telugu track ready with a stock/zero-shot voice)

Track B            [ collect & transcribe training audio ]
                    │──────────│
                               [ fine-tune → test → decide ]
                               │──────────│
                                          [ lip-sync beard test ]
                                          │────│
                                                     [ combine + review ]
                                                     │──────────│
```

| # | Step | Duration | Depends on |
|---|---|---|---|
| 1 | **Audio pipeline** — translation, timing fit, a stock or zero-shot Telugu voice, background preserved | **~1 week** | nothing — starts immediately |
| 2 | **Collect + transcribe training audio** for the voice model | **~1 week**, overlapped with step 1 | nothing — runs alongside step 1 |
| 3 | **Fine-tune the voice model, test it in Telugu, go/no-go call** | **~1 week** | step 2 finished |
| 4 | **Rent a GPU, validate lip-sync on the beard test clip** | **3–5 days**, overlapped with step 3 | nothing — only needs the test clip |
| 5 | **Combine into one full run and review** — voice + lip-sync + background, end to end | **~1 week** | steps 1, 3, 4 all finished |

**Where the timeline can compress:** if the fast-path zero-shot voice (§2)
turns out to be good enough on review, step 3 becomes optional and the whole
plan collapses to roughly **2 weeks** — step 1, the lip-sync test, and one
combined run.

**Where it can stretch:** if the fine-tuned voice fails the Telugu listening
test in step 3 and needs a second iteration (different training data, a
different base model), add another week. This is the one genuinely
open-ended part of the plan — everything else has a known shape.
