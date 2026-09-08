"""Minimal SRT reader/writer (no dependency, and we need both directions)."""
from __future__ import annotations

import re
from pathlib import Path

_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def parse_timestamp(ts: str) -> float:
    m = _TS.search(ts)
    if not m:
        raise ValueError(f"bad timestamp: {ts}")
    h, mn, s, ms = m.groups()
    return int(h) * 3600 + int(mn) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def format_timestamp(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def read_srt(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    cues: list[dict] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        idx = 0
        if lines[0].strip().isdigit():
            idx = 1
        if "-->" not in lines[idx]:
            continue
        start_s, end_s = lines[idx].split("-->")
        cues.append({
            "start": parse_timestamp(start_s),
            "end": parse_timestamp(end_s),
            "text": " ".join(lines[idx + 1:]).strip(),
        })
    return cues


def write_srt(cues: list[dict], path: str | Path) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{format_timestamp(c['start'])} --> "
                   f"{format_timestamp(c['end'])}\n{c['text']}\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return str(path)
