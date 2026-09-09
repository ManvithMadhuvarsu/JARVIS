# English audio → Telugu, native voice, background untouched

One script. Five stages.

```
English audio
  │
1 ├─ Demucs splits it ────► background stem  (KEPT, bit-for-bit)
  │                         vocals stem      (discarded)
2 ├─ Whisper large-v3 transcribes the isolated vocals
3 ├─ Claude rewrites it as spoken Telugu, with a syllable budget per line
4 ├─ Sarvam Bulbul v3 speaks it — a voice trained on Telugu speakers
5 └─ Fit to the original timings, lay it over the untouched background
```

---

## 1. Install

**ffmpeg** first — it is the only thing that is not a pip package:

| | |
|---|---|
| Ubuntu / Debian | `sudo apt-get install -y ffmpeg` |
| macOS | `brew install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` |

Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU is optional. On CPU, Demucs and Whisper are slow but they work — use
`--whisper small.en` on a laptop.

## 2. Get the two API keys

**Sarvam** — the Telugu voice.

1. Sign up at <https://dashboard.sarvam.ai>
2. You get **₹100 in free credits, no card**. At ₹30 per 10,000 characters
   that is roughly 33,000 characters — about **three ten-minute talks, free**.
3. Copy the API subscription key.

**Anthropic** — the translation.

1. Sign up at <https://console.anthropic.com>
2. Create an API key. Translating a ten-minute talk costs a few cents.

```bash
export SARVAM_API_KEY=xxxxxxxx
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

(Windows PowerShell: `$env:SARVAM_API_KEY="xxxxxxxx"`)

## 3. Run

```bash
python dub.py talk.mp3
```

That is the whole command. It also accepts a video file — it reads the audio
track and writes Telugu audio you can mux back in.

**Start with a 1–2 minute clip.** You will want two or three passes before you
like the translation, and every stage is cached so the second run is fast.

```bash
ffmpeg -ss 120 -t 90 -i talk.mp3 -c copy clip.mp3   # a 90-second excerpt
python dub.py clip.mp3
```

## 4. What you get

| File | |
|---|---|
| `talk_te.mp3` | **Telugu voice over your original background** |
| `talk_te_voice_only.mp3` | Telugu voice alone, no background |
| `talk_te.srt` | Telugu subtitles on the fitted timings |
| `talk_work/state.json` | every line: English, Telugu, timing, tempo |

The run ends by telling you which lines overran their slot, if any.

## 5. Fix a line

`state.json` is the working copy. Edit the `te` field of any line, then
re-synthesise just that stage:

```bash
python dub.py talk.mp3 --redo tts
```

`--redo` takes any of `separate transcribe translate tts`. Everything else
stays cached, so fixing one line costs seconds, not another full run.

## 6. Options worth knowing

| Flag | Default | |
|---|---|---|
| `--speaker` | `shubh` | Male: shubh aditya rahul rohan · Female: ritu priya neha kavya |
| `--whisper` | `large-v3` | `small.en` on a laptop, `large-v3` on a GPU |
| `--rate` | `6.5` | Telugu syllables/sec. The run prints the voice's real rate — put it here next time and the length budgets get sharper |
| `--stems` | `2` | `4` when the background has audience noise (see below) |
| `--bg-db` | `-3.0` | Background level. The English voice is *gone*, not ducked, so this can sit high |
| `--pace` | `1.0` | Sarvam's own speaking pace |
| `--style` | — | Steer the register, e.g. `--style "Formal Telugu for a corporate audience."` |

## 7. Two things that will bite you

**Audience voices.** Demucs sorts *all* human voice into the vocals stem — so
laughter, applause and a second speaker get discarded along with the English.
If your background is music or room tone, the default is right. If it contains
people, use `--stems 4`: it keeps drums, bass and other separately, and much of
the room comes back in `other`.

**Whisper mishears names and numbers.** It is the one stage nothing downstream
can recover from — a wrong word gets faithfully translated into wrong Telugu.
Read the `en` fields in `state.json` before you trust the output, fix them,
then `--redo translate`.

## 8. Costs

| | Per 10-minute talk |
|---|---|
| Demucs + Whisper | free (your machine) |
| Claude translation | ~$0.03 |
| Sarvam voice | ~₹36 / $0.43 |
| **Total** | **under $0.50** |

## 9. Why it is built this way

**The background is preserved by separation, not ducking.** Ducking turns the
original down under the dub — the English voice stays quietly audible forever.
Separation deletes the English voice and keeps the background stem untouched.
That is also why `--bg-db` defaults to −3 dB instead of the −15 or −20 dB that
ducking needs.

**Transcription reads the isolated vocals**, not the mix — noticeably fewer
errors when there is music underneath.

**The translator gets a syllable budget per line.** Telugu takes roughly 1.2×
longer than English to say the same thing. If you translate first and think
about timing later, every line overruns. The budget comes from the slot length
and the voice's measured speaking rate; lines predicted to overrun are re-asked
with a tighter budget.

**The dub never starts before the original line did.** Viewers detect audio
leading the picture at 45 ms but tolerate 125 ms of lag (ITU-R BT.1359-1) —
late reads as natural, early reads as broken. So the fitter will pad the end of
a line, borrow from the pause that follows, or compress up to 1.30×, but it
will not steal the start.
