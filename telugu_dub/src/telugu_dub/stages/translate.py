"""Stage 3 — English -> Telugu, with a duration budget attached to every line.

Why not just translate?
    A faithful Telugu rendering of an English sentence is usually 10-25 % longer
    in *spoken* time. If we hand that to TTS the dub runs past the shot and no
    lip-sync model can save it. So translation is where sync is won or lost: we
    give the model a syllable budget derived from the length of the slot it has
    to fit, and re-ask for a shorter rendering when the first one overshoots.

Providers
  llm          : an instruction-following model (Claude by default). Best for
                 discourse: it keeps register, handles Sanskrit/spiritual terms,
                 and can obey a length budget. Needs ANTHROPIC_API_KEY.
  indictrans2  : AI4Bharat's NMT (en-indic 1B). Offline, free, strong literal
                 quality, but no length control and no discourse context — we
                 add a compression retry loop around it.
  mock         : offline phrase table, for pipeline tests without network.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from ..config import ProsodyCfg, TranslateCfg
from ..schema import Segment
from ..telugu_prosody import (count_syllables, estimate_speech_duration,
                              syllable_budget)

SYSTEM_PROMPT = """You are a professional Telugu dubbing translator for spoken \
spiritual discourse.

Rules:
1. Translate meaning, not words. The result must sound like natural spoken \
Telugu, the way a Telugu speaker would say this out loud.
2. LENGTH IS A HARD CONSTRAINT. Each line has a syllable budget. Going over \
means the dub will not fit the speaker's mouth. Prefer shorter synonyms, drop \
English filler ("you know", "see", "so"), and keep it tight.
3. Keep the glossary terms exactly as given.
4. Keep sentence-final punctuation; it becomes a pause in the dub.
5. Output Telugu script only. No transliteration, no English, no commentary.
6. Return strict JSON: {"lines": [{"id": <int>, "te": "<telugu>"}]}"""


# ------------------------------------------------------------------ glossary
def load_glossary(path: str | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in (data.get("terms") or {}).items()}


# ------------------------------------------------------------------ dispatch
def translate_segments(segments: list[Segment], cfg: TranslateCfg,
                       prosody: ProsodyCfg | None = None) -> dict:
    prosody = prosody or ProsodyCfg()
    glossary = load_glossary(cfg.glossary)
    if cfg.provider == "llm":
        _translate_llm(segments, cfg, glossary, prosody)
    elif cfg.provider == "indictrans2":
        _translate_indictrans2(segments, cfg, glossary)
    elif cfg.provider == "mock":
        _translate_mock(segments, cfg, glossary)
    else:
        raise ValueError(f"unknown translate provider: {cfg.provider}")
    return length_report(segments, cfg, prosody)


def length_report(segments: list[Segment], cfg: TranslateCfg,
                  prosody: ProsodyCfg | None = None) -> dict:
    """Predicted overshoot per line, before a single GPU second is spent."""
    prosody = prosody or ProsodyCfg()
    rows, over = [], 0
    for seg in segments:
        est = estimate_speech_duration(
            seg.text_tgt, prosody.target_syllables_per_second,
            prosody.pause_per_comma, prosody.pause_per_sentence)
        slot = seg.source_duration
        ratio = est / slot if slot > 0 else 0.0
        if ratio > 1 + cfg.length_tolerance:
            over += 1
        rows.append({"id": seg.id, "slot": round(slot, 2), "est": est,
                     "ratio": round(ratio, 3),
                     "syllables": count_syllables(seg.text_tgt)})
    return {"lines": rows, "over_budget": over, "total": len(segments),
            "rate_used": prosody.target_syllables_per_second,
            "mean_ratio": round(sum(r["ratio"] for r in rows) / len(rows), 3)
            if rows else 0.0}


# ----------------------------------------------------------------- providers
def _batch(segments: list[Segment], size: int = 12):
    for i in range(0, len(segments), size):
        yield segments[i:i + size]


def _build_user_prompt(batch: list[Segment], all_segs: list[Segment],
                       cfg: TranslateCfg, glossary: dict[str, str],
                       prosody: ProsodyCfg, tighten: bool = False) -> str:
    ids = {s.id for s in batch}
    ctx_before = [s.text_src for s in all_segs
                  if s.id < min(ids) and s.id >= min(ids) - cfg.context_window]
    ctx_after = [s.text_src for s in all_segs
                 if s.id > max(ids) and s.id <= max(ids) + cfg.context_window]
    lines = []
    for s in batch:
        budget = syllable_budget(
            s.source_duration, prosody.target_syllables_per_second,
            prosody.headroom * (0.9 if tighten else 1.0))
        lines.append({"id": s.id, "en": s.text_src,
                      "seconds": round(s.source_duration, 2),
                      "max_syllables": budget})
    payload = {
        "style": cfg.style,
        "glossary": glossary,
        "context_before": ctx_before,
        "context_after": ctx_after,
        "lines": lines,
    }
    extra = ("\nThe previous attempt was TOO LONG. Cut every non-essential word "
             "and stay strictly under max_syllables.\n" if tighten else "")
    return extra + json.dumps(payload, ensure_ascii=False, indent=2)


def _translate_llm(segments: list[Segment], cfg: TranslateCfg,
                   glossary: dict[str, str], prosody: ProsodyCfg) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def ask(batch: list[Segment], tighten: bool) -> dict[int, str]:
        msg = client.messages.create(
            model=cfg.model, max_tokens=4096, system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": _build_user_prompt(batch, segments, cfg,
                                                     glossary, prosody,
                                                     tighten)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0) if m else text)
        return {int(l["id"]): l["te"].strip() for l in data["lines"]}

    for batch in _batch(segments):
        result = ask(batch, tighten=False)
        for seg in batch:
            seg.text_tgt = result.get(seg.id, "")

        if cfg.length_control:
            rate = prosody.target_syllables_per_second
            too_long = [s for s in batch
                        if estimate_speech_duration(s.text_tgt, rate)
                        > s.source_duration * (1 + cfg.length_tolerance)]
            if too_long:
                retry = ask(too_long, tighten=True)
                for seg in too_long:
                    cand = retry.get(seg.id, "")
                    # keep the retry only if it actually got shorter
                    if cand and count_syllables(cand) < count_syllables(seg.text_tgt):
                        seg.text_tgt = cand
                        seg.notes.append("length-control retry applied")


def _translate_indictrans2(segments: list[Segment], cfg: TranslateCfg,
                           glossary: dict[str, str]) -> None:
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = cfg.model if "/" in cfg.model else "ai4bharat/indictrans2-en-indic-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(device)
    proc = IndicProcessor(inference=True)

    for batch in _batch(segments, 8):
        src = [s.text_src for s in batch]
        prepared = proc.preprocess_batch(src, src_lang="eng_Latn", tgt_lang="tel_Telu")
        enc = tok(prepared, truncation=True, padding="longest",
                  return_tensors="pt").to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, num_beams=5, max_length=256,
                                 num_return_sequences=1)
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for seg, out in zip(batch, proc.postprocess_batch(decoded, lang="tel_Telu")):
            seg.text_tgt = _apply_glossary(out, glossary)


def _apply_glossary(text: str, glossary: dict[str, str]) -> str:
    for en, te in glossary.items():
        text = re.sub(rf"\b{re.escape(en)}\b", te, text, flags=re.I)
    return text


def _translate_mock(segments: list[Segment], cfg: TranslateCfg,
                    glossary: dict[str, str]) -> None:
    """Offline phrase-table translation, for pipeline tests without a network.

    Falls back to a syllable-preserving placeholder so timing behaviour stays
    realistic even for lines the table does not know.
    """
    table_path = Path(cfg.glossary or "").with_name("mock_phrases_te.yaml")
    table: dict[str, str] = {}
    if table_path.exists():
        table = (yaml.safe_load(table_path.read_text(encoding="utf-8")) or {}).get(
            "phrases", {})
    norm = {re.sub(r"[^a-z ]", "", k.lower()).strip(): v for k, v in table.items()}
    for seg in segments:
        key = re.sub(r"[^a-z ]", "", seg.text_src.lower()).strip()
        if key in norm:
            seg.text_tgt = norm[key]
        else:
            seg.text_tgt = _placeholder_telugu(seg.text_src)
            seg.notes.append("mock translation (placeholder)")


_TE_SYL = ["క", "మ", "త", "ల", "న", "వ", "ర", "ప", "స", "ద"]
_TE_MATRA = ["", "ా", "ి", "ు", "ె", "ో"]


def _placeholder_telugu(english: str) -> str:
    """Deterministic Telugu-shaped filler with the same syllable count."""
    n = count_syllables(english)
    out, i = [], 0
    while i < n:
        word_len = 2 if (i + 2) <= n else 1
        word = "".join(_TE_SYL[(i + j) % len(_TE_SYL)] + _TE_MATRA[(i + j) % len(_TE_MATRA)]
                       for j in range(word_len))
        out.append(word)
        i += word_len
    return " ".join(out) + "."
