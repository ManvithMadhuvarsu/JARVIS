"""The orchestrator: ingest -> ASR -> translate -> TTS -> align -> render ->
mix -> lip-sync -> mux, with every stage cached on disk so a failed or slow
run resumes instead of restarting.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import qc as qc_mod
from .align import fit_segments, sync_report
from .config import Config
from .media import duration_of
from .modes import MODE_SPECS, describe
from .schema import Manifest
from .stages import asr, ingest, lipsync, mux, render, translate, tts

STAGES = ["ingest", "asr", "translate", "tts", "align", "render", "mix",
          "lipsync", "mux"]


class Pipeline:
    def __init__(self, cfg: Config, verbose: bool = True):
        self.cfg = cfg
        self.work = Path(cfg.workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.work / "manifest.json"
        self.manifest = (Manifest.load(self.manifest_path)
                         if self.manifest_path.exists() else Manifest())
        self.verbose = verbose
        self.timings: dict[str, float] = self.manifest.stats.get("timings", {})

    # ------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[dub] {msg}", flush=True)

    def _save(self) -> None:
        self.manifest.stats["timings"] = self.timings
        self.manifest.save(self.manifest_path)

    def _done(self, stage: str) -> bool:
        return stage in self.manifest.stages_done

    def _mark(self, stage: str, t0: float) -> None:
        if stage not in self.manifest.stages_done:
            self.manifest.stages_done.append(stage)
        self.timings[stage] = round(time.time() - t0, 2)
        self._save()
        self.log(f"{stage} done in {self.timings[stage]}s")

    # --------------------------------------------------------------- stages
    def run(self, source: str, url: str | None = None,
            clip: tuple[float, float] | None = None,
            until: str = "mux", force: list[str] | None = None) -> Manifest:
        for stage in force or []:
            if stage in self.manifest.stages_done:
                self.manifest.stages_done.remove(stage)

        self.log(describe(self.cfg))
        wanted = MODE_SPECS[self.cfg.mode].stages
        stop = STAGES.index(until)
        plan = [s for s in STAGES[:stop + 1] if s in wanted]

        for stage in plan:
            if self._done(stage):
                self.log(f"{stage} cached, skipping")
                continue
            t0 = time.time()
            getattr(self, f"_stage_{stage}")(source=source, url=url, clip=clip)
            self._mark(stage, t0)

        self.write_report()
        return self.manifest

    def _stage_ingest(self, source: str, url: str | None, clip, **_) -> None:
        if url:
            self.log(f"downloading {url}")
            source = ingest.download(url, self.work / "download", clip=clip)
            clip = None                      # yt-dlp already cut the excerpt
        self.log(f"preparing {source}")
        info, audio = ingest.prepare(source, self.work, clip=clip)
        self.manifest.source_url = url or source
        self.manifest.video = info
        self.manifest.artifacts["source_video"] = info.path
        self.manifest.artifacts["source_audio"] = audio
        self.log(f"{info.duration:.1f}s  {info.width}x{info.height} @ {info.fps}fps")

    def _stage_asr(self, **_) -> None:
        chunks = asr.transcribe(self.manifest.artifacts["source_audio"], self.cfg.asr)
        segments = asr.build_units(chunks, self.cfg.asr)
        self.manifest.segments = segments
        speech = sum(s.source_duration for s in segments)
        self.manifest.stats["asr"] = {
            "chunks": len(chunks), "units": len(segments),
            "speech_seconds": round(speech, 1),
            "speech_ratio": round(speech / max(self.manifest.video.duration, 1e-6), 3),
        }
        self.log(f"{len(segments)} dubbing units, {speech:.1f}s of speech")

    def _stage_translate(self, **_) -> None:
        rate = self._source_syllable_rate()
        report = translate.translate_segments(self.manifest.segments,
                                              self.cfg.translate,
                                              self.cfg.prosody)
        self.manifest.stats["translate"] = report
        self.manifest.stats["source_syllable_rate"] = rate
        self.log(f"translated {report['total']} lines, "
                 f"{report['over_budget']} predicted over budget "
                 f"(mean length ratio {report['mean_ratio']})")

    def _source_syllable_rate(self) -> float:
        from .telugu_prosody import count_syllables
        syl = sum(count_syllables(s.text_src) for s in self.manifest.segments)
        dur = sum(s.source_duration for s in self.manifest.segments)
        return round(syl / dur, 2) if dur else 0.0

    def _stage_tts(self, **_) -> None:
        tts.synthesize(self.manifest.segments, self.cfg.tts, self.work / "tts",
                       self.cfg.prosody)
        total = sum(s.tts_duration or 0 for s in self.manifest.segments)
        src = sum(s.source_duration for s in self.manifest.segments)
        self.manifest.stats["tts"] = {
            "synth_seconds": round(total, 1), "source_seconds": round(src, 1),
            "raw_length_ratio": round(total / src, 3) if src else 0.0}
        self.log(f"synthesised {total:.1f}s vs {src:.1f}s of source speech "
                 f"(ratio {total / src:.2f})" if src else "synthesised")

    def _stage_align(self, **_) -> None:
        stats = fit_segments(self.manifest.segments, self.cfg.align,
                             self.manifest.video.duration)
        self.manifest.stats["align"] = stats.as_dict()
        rows = sync_report(self.manifest.segments)
        (self.work / "sync_report.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"fit: {stats.natural} natural, {stats.in_band} in-band, "
                 f"{stats.borrowed_pause} borrowed pause, "
                 f"{stats.hard_compressed} hard, {stats.overflowed} overflow")
        result = qc_mod.evaluate(self.manifest.segments,
                                 max_speed=self.cfg.align.max_speedup)
        self.manifest.stats["qc"] = result.as_dict()["summary"]
        self.manifest.stats["qc"]["passed"] = result.passed
        (self.work / "qc.json").write_text(
            json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        self.log(qc_mod.format_summary(result))

    def _stage_render(self, **_) -> None:
        path = render.render_dub_track(
            self.manifest.segments, self.work,
            self.manifest.video.duration, self.cfg.tts.sample_rate)
        self.manifest.artifacts["dub_track"] = path

    def _stage_mix(self, **_) -> None:
        out = str(self.work / "dub_mixed.wav")
        render.mix_with_background(
            self.manifest.artifacts["dub_track"],
            self.manifest.artifacts["source_audio"], out,
            self.cfg.mix, self.cfg.tts.sample_rate)
        self.manifest.artifacts["dub_audio"] = out

    def _stage_lipsync(self, **_) -> None:
        out = str(self.work / "lipsync.mp4")
        info = lipsync.run_lipsync(self.manifest.artifacts["source_video"],
                                   self.manifest.artifacts["dub_audio"],
                                   out, self.cfg.lipsync)
        self.manifest.artifacts["lipsync_video"] = out
        video_seconds = self.manifest.video.duration
        info["realtime_factor"] = (round(info["seconds"] / video_seconds, 2)
                                   if video_seconds else None)
        self.manifest.stats["lipsync"] = info
        self.log(f"lipsync ({info['provider']}) {info['seconds']}s "
                 f"= {info['realtime_factor']}x realtime")

    def _stage_mux(self, **_) -> None:
        srt_path = None
        if self.cfg.output.write_srt:
            srt_path = mux.write_subtitles(self.manifest.segments,
                                           self.work / "dub_te.srt")
            self.manifest.artifacts["subtitles"] = srt_path

        if self.cfg.mode == "audio":
            final = str(self.work / f"final_te.{self.cfg.output.container}")
            mux.export_audio(self.manifest.artifacts["dub_audio"], final,
                             self.cfg.output)
            self.manifest.artifacts["final"] = final
            self.manifest.stats["final_duration"] = round(duration_of(final), 2)
            self.log(f"final: {final}")
            return

        final = str(self.work / "final_te.mp4")
        video = self.manifest.artifacts.get("lipsync_video") or \
            self.manifest.artifacts["source_video"]
        mux.finalize(video, self.manifest.artifacts["dub_audio"], final,
                     self.cfg.output, srt_path)
        self.manifest.artifacts["final"] = final
        self.manifest.stats["final_duration"] = round(duration_of(final), 2)
        self.log(f"final: {final}")

    # --------------------------------------------------------------- report
    def write_report(self) -> str:
        m = self.manifest
        lines = ["# Dubbing run report", ""]
        if m.video:
            lines += [f"- source: `{m.source_url}`",
                      f"- media: {m.video.duration:.1f}s, {m.video.width}x"
                      f"{m.video.height} @ {m.video.fps} fps", ""]
        lines += ["## Stage timings (seconds)", ""]
        vid = m.video.duration if m.video else 0
        lines += ["| stage | seconds | x realtime |", "|---|---:|---:|"]
        for stage, secs in self.timings.items():
            rt = f"{secs / vid:.2f}x" if vid else "-"
            lines.append(f"| {stage} | {secs} | {rt} |")
        total = sum(self.timings.values())
        lines.append(f"| **total** | **{round(total, 1)}** | "
                     f"**{total / vid:.2f}x**|" if vid else f"| total | {total} | - |")

        if a := m.stats.get("align"):
            lines += ["", "## Sync fit", ""]
            for k, v in a.items():
                lines.append(f"- {k}: {v}")
        if q := m.stats.get("qc"):
            lines += ["", "## QC (ITU-R BT.1359 onset window)", ""]
            for k, v in q.items():
                lines.append(f"- {k}: {v}")
        if t := m.stats.get("tts"):
            lines += ["", "## Length pressure", "",
                      f"- Telugu speech before fitting: {t['synth_seconds']}s "
                      f"vs {t['source_seconds']}s of English "
                      f"(ratio {t['raw_length_ratio']})"]
        if m.stats.get("source_syllable_rate"):
            lines.append(f"- source speaking rate: "
                         f"{m.stats['source_syllable_rate']} syllables/s")

        lines += ["", "## Artifacts", ""]
        for k, v in m.artifacts.items():
            lines.append(f"- {k}: `{v}`")

        path = self.work / "report.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)
