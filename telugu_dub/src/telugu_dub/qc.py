"""Quality gates for a dubbing run.

Three things can be measured cheaply, without a human and without a GPU:

1. **Onset drift** — how far each dubbed utterance starts from the moment the
   original speaker opened their mouth. This is the number that decides whether
   a viewer perceives the video as "in sync", because AV-sync perception is
   dominated by onsets.
2. **Tempo pressure** — how hard we had to bend time. Beyond about 1.15x,
   listeners start to hear "sped up".
3. **Coverage / spill** — does the dub speak while the mouth moves, and stay
   quiet while it does not.

Thresholds come from ITU-R BT.1359-1: viewers start to *detect* desync when
audio leads picture by 45 ms or lags it by 125 ms, and call it *unacceptable*
beyond +90 ms lead / -185 ms lag. EBU R37 is stricter for broadcast production
(+40 / -60 ms). The asymmetry is the important part: ears forgive audio that
arrives late roughly three times as much as audio that arrives early, because
in the physical world sound always arrives after the sight of the event. That
is why the aligner never starts an utterance before the original onset — it
would rather pad the end than steal the start.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Segment

# Perceptual thresholds in seconds (ITU-R BT.1359-1).
# Negative drift = dub starts before the mouth moves = audio leads picture.
LEAD_DETECTABLE = -0.045
LEAD_UNACCEPTABLE = -0.090
LAG_DETECTABLE = 0.125
LAG_UNACCEPTABLE = 0.185


@dataclass
class QcResult:
    passed: bool
    rows: list[dict]
    summary: dict

    def as_dict(self) -> dict:
        return {"passed": self.passed, "summary": self.summary, "rows": self.rows}


def _pctile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def evaluate(segments: list[Segment], max_speed: float = 1.15,
             max_overflow_fraction: float = 0.10) -> QcResult:
    rows: list[dict] = []
    drifts: list[float] = []
    for seg in segments:
        if seg.fit_start is None or not seg.tts_duration:
            continue
        drift = seg.fit_start - seg.start
        drifts.append(drift)
        if drift < LEAD_UNACCEPTABLE:
            grade = "fail-early"
        elif drift < LEAD_DETECTABLE:
            grade = "noticeable-early"
        elif drift <= LAG_DETECTABLE:
            grade = "in-sync"
        elif drift <= LAG_UNACCEPTABLE:
            grade = "noticeable"
        else:
            grade = "fail-late"
        rows.append({"id": seg.id, "onset_drift": round(drift, 3), "grade": grade,
                     "speed": seg.speed, "overflow": seg.overflow})

    n = len(rows) or 1
    abs_drifts = [abs(d) for d in drifts]
    in_sync = sum(1 for r in rows if r["grade"] == "in-sync")
    failed = [r for r in rows if r["grade"].startswith("fail")]
    over_speed = [r for r in rows if r["speed"] > max_speed]
    overflowed = [r for r in rows if r["overflow"] > 0]

    summary = {
        "segments": len(rows),
        "in_sync_pct": round(100 * in_sync / n, 1),
        "mean_abs_drift": round(sum(abs_drifts) / n, 3),
        "p95_abs_drift": round(_pctile(abs_drifts, 0.95), 3),
        "max_drift": round(max(drifts, default=0.0), 3),
        "over_speed_pct": round(100 * len(over_speed) / n, 1),
        "overflow_pct": round(100 * len(overflowed) / n, 1),
        "failing_segments": [r["id"] for r in failed],
    }
    passed = (not failed
              and len(overflowed) / n <= max_overflow_fraction
              and summary["p95_abs_drift"] <= LAG_DETECTABLE)
    return QcResult(passed=passed, rows=rows, summary=summary)


def format_summary(result: QcResult) -> str:
    s = result.summary
    verdict = "PASS" if result.passed else "REVIEW"
    lines = [
        f"QC {verdict}: {s['in_sync_pct']}% of {s['segments']} segments inside "
        f"the ITU-R BT.1359 window "
        f"({int(LEAD_DETECTABLE * 1000)} ms .. +{int(LAG_DETECTABLE * 1000)} ms)",
        f"  mean |drift| {s['mean_abs_drift']}s, p95 {s['p95_abs_drift']}s, "
        f"max {s['max_drift']}s",
        f"  {s['over_speed_pct']}% needed tempo above the comfortable band, "
        f"{s['overflow_pct']}% overflowed their slot",
    ]
    if s["failing_segments"]:
        lines.append(f"  segments to fix: {s['failing_segments']} "
                     f"(shorten the Telugu, or split the slot)")
    return "\n".join(lines)
