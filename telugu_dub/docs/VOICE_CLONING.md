# Getting the dub into his own voice

The short answer: **IndicF5, zero-shot, from an 11-second reference clip cut out
of your own recording.** It is the only option that both speaks Telugu and will
clone an arbitrary voice. Everything else either does not do Telugu or will not
let you clone someone who is not you.

---

## Why the obvious options don't work

| Option | Telugu? | Clones an arbitrary voice? | Verdict |
|---|---|---|---|
| **IndicF5** (AI4Bharat) | yes — 11 Indic languages, 1,417 h of training speech | **yes**, zero-shot from a reference clip + its transcript | **Use this.** Open weights, runs locally |
| XTTS-v2 (Coqui) | **no** — 17 languages, Hindi yes, Telugu no | yes | Dead end for Telugu |
| F5-TTS (base) | not out of the box | yes | IndicF5 *is* F5-TTS fine-tuned for Indic — use that |
| ElevenLabs v3 | yes (74 languages) | Instant cloning yes; **Professional cloning requires live voice verification** | Highest fidelity, but see below |
| Sarvam Bulbul | yes, best-measured Telugu naturalness | **no** — preset voices only | Best stock voice, not a clone |

### The ElevenLabs blocker worth knowing

ElevenLabs' Professional Voice Cloning is the highest-fidelity path and it does
support Telugu. But PVC requires you to pass a **live voice verification** — you
read a phrase, and it must match the audio you uploaded. It is designed
specifically to stop you cloning someone else. The documented route for another
person's voice is that *they* create and verify the clone on their own account
and share it with you.

So the quality ceiling here is not set by the technology. It is set by whether
Isha participates. If they do, ElevenLabs PVC with 30 minutes of clean audio
beats anything you can self-host. If they don't, IndicF5 zero-shot is your
ceiling — decent likeness, not indistinguishable.

---

## What zero-shot cloning will actually sound like

Set expectations before you spend a day on it. From an 11-second reference,
IndicF5 gives you:

- **timbre** — recognisably the same voice type, close to his
- **rough register** — pitch range and weight carry over
- **not the delivery** — the long pauses, the deliberate slowness, the sudden
  emphasis, the humour in the timing. Those are the things that actually make
  him sound like him, and a reference clip does not transfer them.

Expect "sounds like his voice reading something" rather than "sounds like him
speaking." That gap is the reason the QC report matters more than the cloning:
a familiar voice saying something slightly wrong is more jarring than a stranger
saying it right.

---

## Step 1 — cut the reference clip

Already automated. It scores every window in the recording for speech density,
the 3–6 Hz syllabic signature (which separates voice from music and applause),
loudness consistency and clipping, then exports the best few:

```bash
python scripts/make_reference.py --audio talk.mp3 --out samples/ref --seconds 11
```

```
#  start     end       score   speech   syllabic  level    steady
1  04:56.00  05:07.00  3.02    88     % 0.292    -24.5    4.1
2  06:21.50  06:32.50  2.75    68     % 0.320    -24.0    4.3
3  05:36.50  05:47.50  2.66    65     % 0.279    -23.5    4.9
```

Listen to all of them and pick by ear. Choose the **most ordinary** one — his
normal speaking register, calm, no laughter, no music, no dramatic emphasis.
The model copies the mood of the prompt, so a theatrical reference makes every
dubbed line theatrical.

## Step 2 — transcribe the reference exactly

IndicF5 needs the reference audio *and* its exact transcript. Every word,
including false starts and filler — the model aligns against it.

```bash
python -m telugu_dub run --audio samples/ref/ref_01.wav \
    --config config/free_cpu.yaml --until asr --workdir runs/ref
python -c "import json;print(' '.join(s['text_src'] for s in json.load(open('runs/ref/manifest.json'))['segments']))"
```

Check it by ear and correct it. A wrong transcript degrades the clone.

## Step 3 — configure and run

```yaml
# config/clone.yaml  (copy config/free_cpu.yaml and change these)
voice: clone
tts:
  provider: indicf5
  ref_audio: samples/ref/ref_01.wav
  ref_text: "the exact transcript from step 2"
  sample_rate: 24000
```

```bash
python -m telugu_dub doctor --config config/clone.yaml --mode audio --voice clone
python -m telugu_dub run --audio talk.mp3 --config config/clone.yaml \
    --mode audio --voice clone
```

Hardware: a CUDA GPU makes this comfortable; CPU works at roughly 5–10× slower.
For an 8-minute talk on CPU expect a long coffee break, not an overnight run.

## Step 4 — calibrate, because the clone changes the timing

A cloned voice speaks at a different rate from the stock one, and every timing
decision in the pipeline derives from that rate:

```bash
python scripts/calibrate_rate.py --config config/clone.yaml --write
```

Skip this and you will see overflow warnings on lines that were fine before.

---

## Fallback: no clone

```bash
python -m telugu_dub run --audio talk.mp3 --config config/free_cpu.yaml \
    --mode audio --voice preset
```

Free Microsoft neural Telugu voice (`te-IN-MohanNeural`), no GPU, no key. Or
`tts.provider: sarvam` for the best-measured Telugu naturalness — a stock voice
that is genuinely good, rather than a clone that is nearly right.

For a POC that has to be shown to people, this is often the better choice: it
is unambiguously a dub, so nobody has to wonder whether he said it.

---

## Before you distribute anything

Cloning a recognisable public figure's voice is the one step in this project
that carries real risk, and it is not a technical one.

- The recording is copyrighted; a dub is a derivative work.
- Several jurisdictions now treat voice as a protected personal attribute.
- A synthetic voice that is *his* saying words he never said — even a faithful
  translation — is a new utterance attributed to him.

Label the output as an AI translation, in the file metadata and audibly or
on screen. Keep cloned-voice output internal until you have written consent.
The stock-voice path above needs none of that and is available today.
