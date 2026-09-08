"""Core data structures shared by every pipeline stage.

Everything the pipeline produces is JSON-serialisable so that each stage can be
cached on disk and the run resumed from any point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Segment:
    """One utterance: a slot in the source timeline that must be re-voiced."""

    id: int
    start: float                      # source speech start (s)
    end: float                        # source speech end (s)
    text_src: str = ""
    speaker: str = "SPEAKER_00"
    text_tgt: str = ""                # Telugu translation

    # filled in by the tts stage
    tts_path: str | None = None
    tts_duration: float | None = None

    # filled in by the align (isochrony) stage
    fit_start: float | None = None    # where the dub is actually placed (s)
    fit_duration: float | None = None # duration after time-stretching (s)
    speed: float = 1.0                # atempo factor applied (>1 = faster)
    overflow: float = 0.0             # seconds we could not absorb
    notes: list[str] = field(default_factory=list)

    @property
    def source_duration(self) -> float:
        return self.end - self.start


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = True


@dataclass
class Manifest:
    """The single artefact that travels through the pipeline."""

    source_url: str = ""
    video: MediaInfo | None = None
    source_lang: str = "en"
    target_lang: str = "te"
    segments: list[Segment] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)   # stage -> path
    stages_done: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        video = MediaInfo(**d["video"]) if d.get("video") else None
        segs = [Segment(**s) for s in d.get("segments", [])]
        m = cls(**{k: v for k, v in d.items() if k not in ("video", "segments")})
        m.video = video
        m.segments = segs
        return m

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
