"""Run artifacts: JSON, CSV, reports and a ZIP bundle per run."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from models.schemas import Episode, ParseReport, PromoCandidate


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _sorted(cands: list[PromoCandidate]) -> list[PromoCandidate]:
    return sorted(cands, key=lambda c: (c.rank is None, c.rank or 0, c.promo_id))


def candidate_row(c: PromoCandidate) -> dict[str, Any]:
    s = c.scores
    return {
        "rank": c.rank, "promo_id": c.promo_id, "status": c.status, "title": c.title, "origin": c.origin,
        "episodes": ", ".join(c.episode_ids),
        "segments": " | ".join(f"{x.episode_id} {x.start_timestamp}-{x.end_timestamp} ({x.duration_seconds:.2f}s)" for x in c.segments),
        "n_segments": len(c.segments), "total_duration_seconds": c.total_duration_seconds,
        "hook": s.hook if s else None, "cliffhanger": s.cliffhanger if s else None,
        "context": s.context if s else None, "pacing": s.pacing if s else None, "total": s.total if s else None,
        "ranking_score": c.ranking_score, "confidence": c.confidence, "passes_thresholds": c.passes_thresholds,
        "eligible": c.validation.eligible, "human_review_required": c.human_review_required,
        "review_reasons": "; ".join(c.review_reasons), "rejection_reasons": "; ".join(c.rejection_reasons),
        "failure_cases": "; ".join(c.failure_cases), "duplicate_of": c.duplicate_of or "",
        "hook_evidence": " / ".join(e.quote for e in (c.hook.evidence if c.hook else []) if e.verified),
        "cliffhanger_evidence": " / ".join(e.quote for e in (c.cliffhanger.evidence if c.cliffhanger else []) if e.verified),
        "why_it_works": c.why_it_works, "weaknesses": "; ".join(c.weaknesses),
        "generation_call_ids": ", ".join(c.generation_call_ids), "evaluation_call_ids": ", ".join(c.evaluation_call_ids),
    }


def candidates_frame(cands: list[PromoCandidate]) -> pd.DataFrame:
    return pd.DataFrame([candidate_row(c) for c in _sorted(cands)])


def segments_frame(cands: list[PromoCandidate]) -> pd.DataFrame:
    rows = []
    for c in _sorted(cands):
        for i, s in enumerate(c.segments, start=1):
            rows.append({
                "promo_id": c.promo_id, "rank": c.rank, "status": c.status, "segment_index": i,
                "episode_id": s.episode_id, "episode_title": s.episode_title, "source_filename": s.source_filename,
                "start_timestamp": s.start_timestamp, "end_timestamp": s.end_timestamp,
                "start_seconds": s.start_seconds, "end_seconds": s.end_seconds, "duration_seconds": s.duration_seconds,
                "start_event_id": s.start_event_id, "end_event_id": s.end_event_id,
                "scene_ids": ", ".join(s.scene_ids), "source_event_ids": ", ".join(s.source_event_ids),
            })
    return pd.DataFrame(rows, columns=[
        "promo_id", "rank", "status", "segment_index", "episode_id", "episode_title", "source_filename",
        "start_timestamp", "end_timestamp", "start_seconds", "end_seconds", "duration_seconds",
        "start_event_id", "end_event_id", "scene_ids", "source_event_ids"])


def validation_report(cands: list[PromoCandidate]) -> dict[str, Any]:
    items = [{
        "promo_id": c.promo_id, "title": c.title, "status": c.status, "origin": c.origin,
        "validation": c.validation.model_dump(), "rejection_reasons": c.rejection_reasons,
        "review_reasons": c.review_reasons, "warnings": c.warnings, "failure_cases": c.failure_cases,
        "deterministic_notes": c.deterministic_notes,
        "unverified_evidence": [e.model_dump() for k in ("hook", "cliffhanger", "context", "pacing")
                                for e in (getattr(c, k).evidence if getattr(c, k) else []) if not e.verified],
    } for c in _sorted(cands)]
    return {
        "summary": {
            "candidates": len(cands),
            "timestamps_invalid": sum(1 for c in cands if c.segments and not c.validation.timestamps_valid),
            "duration_invalid": sum(1 for c in cands if c.segments and not c.validation.duration_valid),
            "source_unverified": sum(1 for c in cands if not c.validation.source_verified),
            "evidence_unverified": sum(1 for x in items if x["unverified_evidence"]),
            "rejected": sum(1 for c in cands if c.status == "rejected"),
            "duplicates": sum(1 for c in cands if c.status == "duplicate"),
        },
        "candidates": items,
    }


def export_run(run_dir: Path, raw_dir: Path, *, run_meta: dict[str, Any], episodes: list[Episode],
               parse_report: ParseReport, inventory: list[dict[str, Any]], candidates: list[PromoCandidate],
               qa_report: dict[str, Any], qa_markdown: str) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["run_config"] = _write_json(run_dir / "run_config.json", run_meta)
    files["input_inventory"] = _write_json(run_dir / "input_inventory.json", inventory)
    files["parse_report"] = _write_json(run_dir / "parse_report.json", parse_report.model_dump(mode="json"))
    files["normalized_episodes"] = _write_json(run_dir / "normalized_episodes.json", [e.model_dump(mode="json") for e in episodes])
    files["candidates_all"] = _write_json(run_dir / "candidates_all.json", {
        "run_id": run_meta.get("run_id"), "backend": run_meta.get("backend"),
        "candidates": [c.model_dump(mode="json") for c in _sorted(candidates)],
    })
    ranked = [c for c in candidates if c.rank is not None]
    df_ranked = candidates_frame(ranked)
    files["ranked_promos_csv"] = run_dir / "ranked_promos.csv"
    df_ranked.to_csv(files["ranked_promos_csv"], index=False)
    files["candidates_summary_csv"] = run_dir / "candidates_summary.csv"
    candidates_frame(candidates).to_csv(files["candidates_summary_csv"], index=False)
    files["promo_segments_csv"] = run_dir / "promo_segments.csv"
    segments_frame(candidates).to_csv(files["promo_segments_csv"], index=False)
    files["validation_report"] = _write_json(run_dir / "validation_report.json", validation_report(candidates))
    files["qa_report"] = _write_json(run_dir / "qa_report.json", qa_report)
    (run_dir / "qa_report.md").write_text(qa_markdown, encoding="utf-8")
    files["qa_report_md"] = run_dir / "qa_report.md"
    raw = raw_dir / "raw_responses.jsonl"
    if raw.exists():
        files["raw_responses"] = raw
    zip_path = run_dir / f"{run_meta.get('run_id', 'run')}_artifacts.zip"
    zip_path.write_bytes(build_zip(run_dir, raw_dir, exclude=zip_path.name))
    files["artifacts_zip"] = zip_path
    return files


def build_zip(run_dir: Path, raw_dir: Path, exclude: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(run_dir.rglob("*")):
            if p.is_file() and p.name != exclude and not p.name.endswith("_artifacts.zip"):
                zf.write(p, f"outputs/{p.relative_to(run_dir)}")
        if raw_dir.exists():
            for p in sorted(raw_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, f"raw_outputs/{p.relative_to(raw_dir)}")
    return buf.getvalue()
