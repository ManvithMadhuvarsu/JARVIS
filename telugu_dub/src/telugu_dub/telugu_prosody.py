"""Telugu text metrics used for length-controlled translation and for
predicting how long a sentence will take to speak *before* we synthesise it.

Telugu is an abugida: every consonant carries an inherent /a/, vowel signs
(matras) replace it, and the virama (U+0C4D, "halant") suppresses it to form a
consonant cluster.  So the syllable count of a word is:

    (independent vowels) + (consonants) - (viramas)

which is a very good proxy for spoken length, far better than a character or
word count.

DEFAULT_TE_RATE is a starting point, not a fact: published articulation rates
for professional Indian-language narration sit in the 6-8 syllables/second band
and matched-sentence studies across seven languages span 5.2-7.8, but there is
no Telugu-specific published figure, and the number that actually matters is
the rate of *your* TTS voice.  Run scripts/calibrate_rate.py against the voice
you ship and write the measured value into config.prosody.  Every downstream
decision - the translator's syllable budget, the overflow warnings, the tempo
the aligner picks - inherits this constant, so calibrating it is the single
highest-leverage tuning step in the pipeline.
"""
from __future__ import annotations

import re
import unicodedata

# Telugu Unicode block U+0C00-U+0C7F
TE_INDEPENDENT_VOWELS = set(range(0x0C05, 0x0C15))     # అ .. ఔ (incl. vocalic r/l)
TE_CONSONANTS = set(range(0x0C15, 0x0C3A))             # క .. హ (+ extras)
TE_VIRAMA = 0x0C4D
TE_MATRAS = set(range(0x0C3E, 0x0C4D))                 # dependent vowel signs
TE_RANGE = range(0x0C00, 0x0C80)

DEFAULT_TE_RATE = 6.5      # Telugu syllables/second, calibrate per voice
DEFAULT_EN_RATE = 4.2      # English syllables/second by the heuristic below

# Latin fallback: rough syllable count by vowel groups.
_LATIN_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.I)


def is_telugu(text: str) -> bool:
    te = sum(1 for ch in text if ord(ch) in TE_RANGE)
    letters = sum(1 for ch in text if ch.isalpha())
    return letters > 0 and te / letters > 0.5


def count_syllables(text: str) -> int:
    """Syllable count for Telugu (script-aware) or Latin (heuristic) text."""
    text = unicodedata.normalize("NFC", text)
    if is_telugu(text):
        n = 0
        for ch in text:
            cp = ord(ch)
            if cp in TE_INDEPENDENT_VOWELS:
                n += 1
            elif cp in TE_CONSONANTS:
                n += 1
            elif cp == TE_VIRAMA:
                n -= 1          # cluster: the two consonants share one syllable
        return max(n, 0)
    # Latin heuristic
    n = 0
    for word in re.findall(r"[A-Za-z']+", text):
        groups = len(_LATIN_VOWEL_GROUP.findall(word))
        if word.lower().endswith("e") and groups > 1:
            groups -= 1                      # silent final e
        n += max(groups, 1)
    return n


def estimate_speech_duration(
    text: str,
    syllables_per_second: float = DEFAULT_TE_RATE,
    pause_per_comma: float = 0.18,
    pause_per_sentence: float = 0.34,
) -> float:
    """Predict spoken duration (seconds) without running TTS.

    Used to (a) give the translator a character/syllable budget and (b) warn
    about segments that will overflow their slot before we spend GPU time.
    """
    syl = count_syllables(text)
    if syl == 0:
        return 0.0
    dur = syl / max(syllables_per_second, 0.1)
    dur += pause_per_comma * len(re.findall(r"[,;:—-]", text))
    dur += pause_per_sentence * len(re.findall(r"[.!?।]", text))
    return round(dur, 3)


def syllable_budget(slot_seconds: float,
                    syllables_per_second: float = DEFAULT_TE_RATE,
                    headroom: float = 1.0) -> int:
    """How many Telugu syllables comfortably fit in a slot of N seconds."""
    return max(1, int(slot_seconds * syllables_per_second * headroom))


def measure_rate(samples: list[tuple[str, float]]) -> dict:
    """Fit syllables/second from (text, measured_seconds) pairs.

    `samples` come from actually synthesising sentences with the target voice.
    Returns the rate plus the spread, because a voice with a wide spread needs
    a looser length tolerance in the translator.
    """
    rows = [(count_syllables(t), d) for t, d in samples if d > 0 and count_syllables(t)]
    if not rows:
        return {"rate": DEFAULT_TE_RATE, "n": 0}
    rates = [syl / dur for syl, dur in rows]
    rates.sort()
    total_syl = sum(r[0] for r in rows)
    total_dur = sum(r[1] for r in rows)
    mid = len(rates) // 2
    return {
        "rate": round(total_syl / total_dur, 3),          # syllable-weighted
        "median": round(rates[mid], 3),
        "min": round(rates[0], 3),
        "max": round(rates[-1], 3),
        "n": len(rows),
        "total_seconds": round(total_dur, 2),
    }
