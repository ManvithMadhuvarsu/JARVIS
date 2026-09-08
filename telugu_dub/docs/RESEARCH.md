# English → Telugu dubbing with lip sync: research and build plan

Scope: take an English talk (the working example is a Sadhguru discourse on
YouTube) and produce a Telugu version where the voice *and* the mouth match.

Not every deliverable needs all of that. The six shipping configurations —
audio-only / video / video-with-lip-sync, each with a stock or a cloned voice —
are laid out in [USE_CASES.md](USE_CASES.md), and the fastest way onto your own
clip is [QUICKSTART.md](QUICKSTART.md). This document is the *why* behind those
choices.

---

## 1. The thing people get wrong

"Translate the video into Telugu" sounds like one problem. It is three, and
they fight each other:

| Problem | What it means | Who solves it |
|---|---|---|
| **Semantic** | Say the same thing in Telugu | ASR + MT |
| **Temporal** | Say it in the same number of seconds | length-controlled MT + isochrony fitting |
| **Visual** | Have the mouth make those sounds | lip-sync generation |

The third one gets all the attention and is the *least* important. A viewer
forgives a soft mouth. A viewer does not forgive audio that starts before the
mouth opens, or a sentence that is still going when the speaker has visibly
stopped. **Sync is won in the translation stage, not in the lip-sync stage** —
by the time you are running a diffusion model over frames, the timing is already
decided.

Concretely: Telugu renderings of English discourse run roughly **1.15–1.30×
longer in spoken time** in our measurements on this corpus (the POC run reports
1.21× before fitting). If you translate first and think about timing later,
every sentence overruns and the whole video drifts.

### Why this content is unusually favourable

Sadhguru's delivery has three properties that make it one of the easier dubbing
targets:

- **Slow, pause-heavy speech.** Long silences between sentences are elastic the
  aligner can borrow. Fast-cut interview content has none.
- **Static framing.** Long single-camera takes, head roughly frontal, minimal
  motion blur — the regime lip-sync models were trained on.
- **Monologue.** No overlapping speakers, so no diarization headaches and no
  cross-talk to separate.

And one property that makes it unusually *hard*:

- **A full beard.** Every open-source lip-sync model inpaints a mouth region
  learned mostly from clean-shaven faces. Beards produce the classic failure:
  a smeared, low-frequency mush around the mouth, or a mouth that "floats" over
  the facial hair. This is the single biggest visual risk in this project and it
  must be evaluated on a real clip before committing to a model. Budget a
  face-restoration pass (GFPGAN/CodeFormer) and expect to prefer higher-fidelity
  models (LatentSync 512) over faster ones for this reason.

---

## 2. Pipeline

```
YouTube URL
    │  yt-dlp
    ▼
[1] ingest ─────────► video.mp4 + 16 kHz mono wav
    │
    ▼
[2] ASR ────────────► English segments with word-level timings
    │                 faster-whisper large-v3 / WhisperX / official captions
    ▼
[3] translate ──────► Telugu text WITH A SYLLABLE BUDGET PER LINE
    │                 LLM (length-controlled) or IndicTrans2
    ▼
[4] TTS ────────────► one wav per segment, cloned voice
    │                 IndicF5 / Sarvam Bulbul / ElevenLabs
    ▼
[5] align ──────────► per-segment start time + tempo factor   ◄── the sync stage
    │                 isochrony fitting (this repo)
    ▼
[6] render ─────────► one continuous dubbed track
    │                 time-stretch (atempo) + timeline assembly
    ▼
[7] mix ────────────► dub over the original music/ambience
    │                 Demucs stem separation, or side-chain ducking
    ▼
[8] lip-sync ───────► regenerated mouth
    │                 LatentSync / MuseTalk / Wav2Lip / hosted API
    ▼
[9] mux ────────────► final_te.mp4 + Telugu SRT + QC report
```

Every stage is cached in the run directory. Lip-sync is 90 % of the compute, so
you iterate on stages 2–7 (seconds) and only run stage 8 when the audio is
final.

---

## 3. Stage-by-stage technology choice

### 3.1 ASR — English transcription with timings

| Option | Quality | Speed | Notes |
|---|---|---|---|
| **faster-whisper large-v3** | best open | ~12× realtime on an RTX 4070 | CTranslate2 build of Whisper; word timestamps, built-in VAD |
| **WhisperX** | same + precise word alignment | ~60–70× realtime (large-v2/v3, batched) | wav2vec2 forced alignment; adds pyannote diarization |
| **YouTube official captions** | human-checked where available | free, instant | Isha publishes captions on much of its catalogue — use them |
| whisper.cpp | good | CPU-viable | for laptops without a GPU |

**Recommendation:** WhisperX when you need tight word boundaries (it is the
alignment that makes pause-borrowing safe), faster-whisper otherwise, and
*always* check for an official transcript first — it is free and better than
any model.

The subtlety: ASR segments are cut for transcription, not for dubbing. This repo
re-merges them into breath groups (`stages/asr.py: build_units`), because the
pauses between groups are the raw material the aligner spends later.

### 3.2 Translation — where sync is actually decided

| Option | Quality en→te | Length control | Cost |
|---|---|---|---|
| **LLM (Claude/GPT class)** | best for discourse: register, metaphor, Sanskrit terms | **yes** — you can hand it a syllable budget and re-ask | per-token |
| **IndicTrans2 (AI4Bharat, 1B)** | strong literal MT; beats commercial baselines by 4–8 BLEU/chrF++ on en→Indic | no | free, offline |
| Google/Azure MT | solid | no | per-character |
| Bhashini (govt of India) | good Indic coverage | no | free tier |

**Recommendation:** LLM as the primary translator, with the syllable budget in
the prompt (implemented in `stages/translate.py`), and IndicTrans2 as an offline
fallback or as a cross-check. A useful hybrid seen in production Indic systems:
translate with the LLM, and use a second LLM pass as a judge that routes bad
lines to IndicTrans2 for re-translation.

Three things the prompt must carry:

1. **A syllable budget per line**, derived from the slot length and the measured
   speaking rate of your TTS voice.
2. **A glossary** (`config/glossary_te.yaml`). Yogic vocabulary — *karma*,
   *dhyana*, *mukti*, *Adiyogi* — already exists in Telugu. Translating it into
   everyday words destroys the meaning; transliterating it inconsistently
   destroys the trust. Pin the terms.
3. **Neighbouring lines as context.** Discourse is full of "this", "that",
   "it" — pronouns that a per-sentence translator will get wrong.

### 3.3 Telugu TTS — the voice

| Option | Voice cloning | Licence | Notes |
|---|---|---|---|
| **IndicF5 (AI4Bharat)** | zero-shot from a reference clip + its transcript | open weights | trained on 1,417 h across 11 Indian languages incl. Telugu; 24 kHz out |
| **Sarvam Bulbul v3** | preset voices | hosted | won a 20k-vote blind eval across 11 Indian languages over ElevenLabs v3 alpha, v2.5 flash and Cartesia Sonic-3, in both 48 kHz and telephony bands; sub-250 ms streaming |
| **ElevenLabs multilingual v2/v3** | instant + professional cloning | hosted | best prosody controls; strongest cloning; ~$0.18/min for dubbing |
| Bhashini / IndicTTS | preset | free | fallback, more robotic |

**Recommendation:** IndicF5 if the dub must carry Sadhguru's own timbre and stay
self-hosted; Sarvam if you want a hosted Telugu-native voice with the best
measured naturalness; ElevenLabs if cloning fidelity matters more than cost.

Note the reference-audio requirement: IndicF5 needs a clean 5–15 s clip *plus its
exact transcript*. Pick a clip with no music, no applause, and a neutral tone —
the model copies the prosody of the prompt, so a dramatic reference makes every
line dramatic.

### 3.4 Isochrony — fitting speech to the mouth

This is the part no off-the-shelf tool does well, and it is implemented in full
in `src/telugu_dub/align.py`.

The research consensus (VideoDubber; isochrony-aware NMT; prosodic alignment
work) is a two-layer approach:

1. **Control the length at translation time** — pick shorter synonyms when both
   are correct. Cheap and artefact-free.
2. **Bounded time-stretching afterwards** — but only within roughly ±10–15 %,
   because past that the ear hears it.

Our fitter adds a third layer that matters for talking heads specifically:

3. **Pause borrowing with a bounded cascade.** A sentence that will not fit may
   eat up to 80 % of the silence that follows it, and may start up to 300 ms
   late — but never earlier than the original onset, and one bad line can never
   push the rest of the video out of sync.

Why never earlier: ITU-R BT.1359-1 says viewers detect desync at **45 ms of
audio leading** the picture but tolerate **125 ms of audio lagging** — a 3:1
asymmetry, because in the physical world sound always arrives after the sight.
Late is natural; early is broken. (EBU R37 is stricter for broadcast production:
+40/−60 ms.) These are the thresholds `qc.py` grades against.

### 3.5 Background audio

The original track carries music, ambience, and audience response you do not
want to lose.

- **Demucs (htdemucs, two-stem)** — separate vocals from everything else, keep
  the non-vocal stem, put the Telugu on top. Clean, needs a GPU to be quick.
- **Side-chain ducking** — keep the whole original quietly underneath and duck
  it when the dub speaks. No separation artefacts, but the English voice stays
  faintly audible. Acceptable for talking-head content with little music.

### 3.6 Lip sync

| Model | VRAM | Resolution | Strength | Licence |
|---|---|---|---|---|
| **LatentSync 1.6** (ByteDance) | 18 GB | 512×512 | best visual fidelity: teeth, tongue, lip shape hold up at 720p+ | Apache-2.0 |
| LatentSync 1.5 | 8 GB | 256×256 | cheaper, softer | Apache-2.0 |
| **MuseTalk** | lower | ~256 | ~30 fps on a V100 — real-time class; the right default for long videos | check repo licence |
| **Wav2Lip** | tiny | 96×96 mouth patch | strongest raw sync score, runs on anything, visibly soft — needs face restoration | non-commercial weights |
| SadTalker | — | — | still image → talking head, not for dubbing existing footage | |
| **sync.so / HeyGen API** | none | up to 1080p | no GPU to own | ~$2/min (HeyGen standard), ~$4/min high-precision |

**Recommendation for this project:** LatentSync 1.6 at 512, with a
GFPGAN/CodeFormer restoration pass, and MuseTalk for fast iteration. Then judge
on a beard clip before you commit — see §1.

Practical notes that cost people days:

- Models resample to **25 fps and 16 kHz** internally. Feed them that directly
  so you control the resampling.
- Run lip-sync **only on shots where the speaker's face is visible**. Cutaways
  to the audience or to B-roll must pass through untouched, or you will
  hallucinate mouths onto the crowd.
- Each of these repos pins conflicting torch versions. Give each its own venv
  and call it as a subprocess (this repo does).

### 3.7 Quality control

| Metric | What it measures | Target |
|---|---|---|
| **Onset drift** (this repo, `qc.py`) | dub start vs original mouth onset | inside −45/+125 ms for ≥95 % of segments |
| **Tempo pressure** (this repo) | how far we bent time | <15 % of segments above 1.15× |
| **LSE-D** (SyncNet distance) | audio-visual sync of the rendered face | ≤8.0 general, ≤6.0 for close-ups |
| **LSE-C** (SyncNet confidence) | AV correlation | ≥0.35 for close-ups (higher is better) |
| Back-translation + LLM judge | did the meaning survive | spot-check |
| Native-speaker MOS | the only thing that actually counts | 1–5, n≥5 raters |

Caveat worth knowing: LSE-D correlates only ~0.36 with human opinion scores. It
is a regression guard, not a verdict. Ship nothing without a Telugu speaker
watching it.

---

## 4. Requirements

### Hardware

| Tier | GPU | What it gets you |
|---|---|---|
| Minimum | RTX 3090 / 4090, 24 GB | LatentSync 1.6 (18 GB), Whisper large-v3, IndicF5 — one video at a time |
| Comfortable | A100 40 GB / L40S | headroom to keep models resident between stages |
| Cheap iteration | any 8 GB GPU | LatentSync 1.5 at 256, MuseTalk, Wav2Lip |
| No GPU | — | everything except lip-sync via hosted APIs |

Also: ~10 GB disk per hour of 720p working footage (intermediates dominate),
ffmpeg with libass and a Telugu font (`fonts-noto-telugu`) if you burn subtitles.

### Software stack

Python 3.10+ · PyTorch 2.x + CUDA · faster-whisper / WhisperX · IndicTrans2 or
an LLM API · IndicF5 (transformers) · Demucs · LatentSync / MuseTalk ·
ffmpeg · yt-dlp.

---

## 5. How long does it take?

Per **one minute of source video**, on a single 24 GB GPU. Stages 1–7 are
measured or well-anchored; the lip-sync row is the one you must measure
yourself, and the pipeline prints it (`stats.lipsync.realtime_factor`).

| Stage | Time per video-minute | Basis |
|---|---|---|
| ingest (download + extract) | 5–15 s | network-bound |
| ASR (faster-whisper large-v3) | 5–10 s | ~12× realtime on a 4070; WhisperX batched is faster |
| translate (LLM, batched 12 lines) | 5–15 s | ~10 lines/min of speech, one call per batch + retries |
| TTS (IndicF5) | 15–30 s | flow-matching TTS, roughly 0.3–0.5× realtime per synthesised second |
| align + render + mix | 2–5 s | ffmpeg-bound; measured 0.1× realtime in the POC |
| Demucs separation | ~6 s | ~0.1× realtime on GPU |
| **lip-sync (LatentSync 512, 20 steps)** | **8–25 min** | *estimate*: 25 fps × 20 denoise steps = 500 UNet passes per video-second. Measure before you plan. |
| lip-sync (MuseTalk) | ~1–2 min | anchored: repo reports 30 fps+ on a V100, i.e. near real-time |
| lip-sync (hosted API) | 1.5–3 min | WaveSpeed's LatentSync deployment reports ~90 s median per request |
| mux + encode | 5–10 s | libx264 CRF 18 |

**So, end to end:**

| Source length | Audio-only dub (no lip-sync) | With MuseTalk | With LatentSync 512 |
|---|---|---|---|
| 1 min | **~1 min** | ~3 min | ~10–25 min |
| 10 min | ~8 min | ~25 min | 1.5–4 h |
| 60 min | ~45 min | ~2.5 h | 8–24 h |

Three things change these numbers a lot:

- **Face-only processing.** If the speaker is on screen 60 % of the time, you
  cut lip-sync cost by 40 % by skipping the rest. Biggest single win.
- **Batching / multi-GPU.** Lip-sync is embarrassingly parallel across shots.
- **Human review.** Budget **2–4× the source length** in a Telugu reviewer's
  time for the first videos, dropping as the glossary matures. This, not GPU
  time, is what limits throughput in practice.

### Cost, roughly

- **Self-hosted:** an L40S/A100 rents for ~$1–2/h → **$1–4 per finished minute**
  with LatentSync, well under $1 with MuseTalk.
- **Hosted:** dubbing APIs run ~$0.20–2.40 per dubbed minute in 2026;
  ElevenLabs dubbing is ~$0.18/min audio-only, HeyGen video translate with
  lip-sync ~$2/min standard and ~$4/min high-precision.
- **LLM translation:** cents per minute of video.

For a POC, hosted is cheaper than the engineer-hours to set up local weights.
For a catalogue of thousands of hours, self-hosting wins decisively.

---

## 6. Failure modes to expect on this content

| Symptom | Cause | Fix |
|---|---|---|
| Mouth smears / floats | beard, model trained on clean-shaven faces | higher-res model, face restoration, or accept voice-over-only |
| Dub finishes after the speaker stops | translation too long | tighten the syllable budget, split the segment |
| Speech sounds rushed | tempo above ~1.15× | re-translate shorter rather than compress harder |
| Mouths appear on the audience | lip-sync applied to cutaways | face detection gate per shot |
| Sanskrit terms mangled | translator "helpfully" translated them | glossary pinning |
| English loanwords sound foreign | Telugu speakers code-switch naturally | allow common loanwords; don't over-Sanskritise |
| Voice drifts between segments | TTS reference prompt inconsistency | one fixed reference clip for the whole video |
| Audio leads the picture | naive stretching that shifts onsets early | never start before the original onset (enforced here) |

---

## 7. Rights and consent

This is not a footnote; it decides whether the project can ship.

- **The footage is copyrighted** (Isha Foundation). Dubbing and republishing is
  a derivative work. Internal evaluation is one thing; distribution needs
  permission.
- **Cloning a real person's voice needs that person's consent.** Several
  jurisdictions now treat voice as a protected personal attribute, and this is
  a recognisable public figure speaking on spiritual matters, where a fabricated
  statement does real damage.
- **A synthetic likeness saying words he did not say is the core risk.** Even a
  faithful translation is a *new* utterance in his voice and face. Label the
  output as an AI translation, on the video and in the metadata.
- **Isha already publishes Telugu content.** The realistic path is a partnership
  where they supply reference audio and approve output — which also gets you
  better source material than YouTube rips.

A defensible POC: use a short excerpt, keep it internal, label it, and use a
neutral (non-cloned) Telugu voice until consent for cloning exists. The pipeline
supports exactly this — `tts.provider: sarvam` with a preset voice.

---

## 8. Roadmap

**Phase 0 — POC (this repo).** Pipeline plumbing, isochrony fitting, QC gates,
offline test run. Done.

**Phase 1 — one real clip (1–2 days on a GPU box).** 60 s of a real talk,
real models, LatentSync and MuseTalk side by side, one Telugu native reviewer.
Deliverable: the beard question answered, and measured timings that replace the
estimates in §5.

**Phase 2 — one full talk (1 week).** Shot detection, face-gated lip-sync,
Demucs, glossary hardened over ~10 videos' worth of terms, subtitle burn-in.

**Phase 3 — production (4–8 weeks).** Queue + worker on GPUs, resumable jobs,
reviewer UI showing per-segment drift and letting a human retranslate a line and
re-render only that segment, LSE-D/LSE-C regression tracking, cost dashboard.

---

## Sources

- [LatentSync (ByteDance)](https://github.com/bytedance/LatentSync) — VRAM, 512 support, Apache-2.0
- [Open-source lip-sync model comparisons, 2026](https://instavar.com/research/ai-video/open-source-lip-sync-models) · [lipsync.com](https://lipsync.com/blog/open-source-lip-sync) · [model selection guide](https://tomodahinata.com/en/blog/ai-lip-sync-talking-head-model-selection-guide-2026)
- [IndicF5 (AI4Bharat)](https://github.com/AI4Bharat/IndicF5) · [HF model card](https://huggingface.co/ai4bharat/IndicF5)
- [Sarvam AI TTS](https://www.sarvam.ai/text-to-speech) · [Open-source voice AI in India, 2026](https://caller.digital/blog/open-source-voice-ai-india-sarvam-ai4bharat-bhasini-2026)
- [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) · [paper](https://arxiv.org/pdf/2305.16307)
- [VideoDubber: MT with speech-aware length control](https://arxiv.org/pdf/2211.16934) · [Isochrony-aware NMT for dubbing](https://arxiv.org/pdf/2112.08548) · [Jointly optimizing translations and speech timing](https://arxiv.org/pdf/2302.12979)
- [AI dubbing pipelines overview](https://soniox.com/wiki/ai-dubbing)
- [WhisperX](https://github.com/m-bain/whisperX) · [Whisper variant comparison](https://modal.com/blog/choosing-whisper-variants)
- [LSE-D / LSE-C](https://www.emergentmind.com/topics/lse-d-lip-sync-error-distance) · [OTT lip-sync QA thresholds](https://www.truefan.ai/blogs/ott-lip-sync-accuracy-testing)
- [AV sync perception study citing ITU-R BT.1359-1 and EBU R37](https://arxiv.org/html/2212.01686v1)
- [Dubbing API pricing comparison](https://startpinch.com/guides/dubbing-api-pricing-comparison) · [HeyGen vs ElevenLabs vs Rask](https://www.heygen.com/blog/heygen-vs-elevenlabs-vs-rask-ai-vs-dubverse)
- [Speech rate across languages](https://languagelog.ldc.upenn.edu/nll/?p=22) · [speaking rate / information density](https://www.isca-archive.org/interspeech_2019/bradlow19b_interspeech.pdf)
