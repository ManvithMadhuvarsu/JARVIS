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

---

## 2. Step 4: Sadhguru's voice, in Telugu

Two approaches, in order of effort:

### Zero-shot cloning (fast path)
Feed a TTS model a short clean reference clip of his voice (10–15 seconds)
alongside the Telugu text, and it imitates the *timbre* of that clip. No
training required — this can produce a result immediately. What it will
**not** capture: his delivery — the long pauses, the way he lands emphasis,
the timing that actually makes him sound like him rather than "a similar
voice." Good for a fast first pass, a ceiling on realism.

### Fine-tuned voice model (the real target)
Train a voice model on a substantial set of his own recordings, so it
captures timbre *and* delivery. What this takes:

| Training audio | Result |
|---|---|
| 1–3 hours | Workable, but rough on words unlike the training set |
| **5–20 hours** | **The target** — clean, single-speaker, accurately transcribed |
| 30+ hours | Studio-grade |

His public catalogue is overwhelmingly English, so the plan is:

1. **Collect 5–20 hours of clean English audio** — long single-camera talks,
   background/audience stripped out (step 1 above, reused as a prep tool).
2. **Transcribe and hand-correct** the training transcripts — this is the
   ceiling on fine-tune quality, worth getting right.
3. **Fine-tune a voice model** (F5-TTS-based architecture, Indic-capable)
   on that English audio, using the standard fine-tuning procedure for that
   model family. This step needs a GPU with real memory (A100-class),
   running for several hours.
4. **Generate Telugu with the fine-tuned model** and evaluate it by ear —
   specifically with a native Telugu speaker, blind (they shouldn't know in
   advance which clip is the target voice), scoring pronunciation, pitch,
   and rhythm separately, since a voice can nail one and miss another.
5. **Decide based on that listening test:**
   - If the Telugu output is convincing → that's the production voice.
   - If it isn't → the same fine-tuned model still gives a strong result for
     **English**-in-his-voice output, and Telugu falls back to zero-shot
     cloning (his timbre, without full delivery) as the interim answer while
     the cross-lingual approach is refined further.

**Consent is part of the plan, not an afterthought.** Training a model that
can make him say sentences he never spoke is a bigger step than dubbing his
own words, and it should only proceed with Isha's participation or explicit
written consent, with the resulting model kept access-controlled.

---

## 3. Step 7: Lip-sync

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

## 5. Plan, in order

1. **Get the audio/timing pipeline producing a finished Telugu track** —
   translation, a stock or zero-shot Telugu voice, timing fit, background
   preserved. This is the fast, low-risk foundation everything else sits on.
2. **Collect and prepare the training audio** for the voice model in
   parallel — this can start immediately and doesn't block step 1.
3. **Fine-tune the voice model, test it in Telugu, and make the go/no-go
   call** on cross-lingual quality with a real Telugu listener.
4. **Rent a GPU and validate lip-sync on the beard test clip.**
5. **Combine everything into one full run** — his voice (or the best
   available fallback), timing-fitted, lip-synced, over the original
   background — and review the complete result before wider use.
