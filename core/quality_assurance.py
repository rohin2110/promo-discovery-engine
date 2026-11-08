"""Quality assurance: failure-case tagging, run-level report, reviewer feedback.

No statement here predicts views or conversions; the report only explains the
editorial evidence and the deterministic checks behind each decision.
"""
from __future__ import annotations

import json
import statistics
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from config.settings import ScoringConfig
from core.validator import EpisodeIndex
from models.schemas import EventType, PromoCandidate

FAILURE_CASES = {
    "strong_hook_weak_ending": "Strong hook but weak ending",
    "strong_cliffhanger_weak_context": "Strong cliffhanger but insufficient context",
    "good_dialogue_poor_standalone_narrative": "Good dialogue but poor standalone narrative",
    "duration_out_of_range": "Excessively long or short clip",
    "duplicate_moment": "Duplicate promotional moment",
    "timestamp_issues": "Missing or inconsistent timestamps",
    "incoherent_cross_episode": "Cross-episode segments that do not form a coherent story",
    "unverified_evidence": "Evidence not found verbatim in the source (possible hallucination)",
    "hallucinated_source": "Clip references events that do not exist in the source",
}


def tag_failure_cases(c: PromoCandidate, cfg: ScoringConfig, index: Optional[EpisodeIndex] = None) -> list[str]:
    tags = [t for t in c.failure_cases if t in ("duplicate_moment", "incoherent_cross_episode")]
    w, strong, weak = cfg.weights, cfg.failure_case_strong_ratio, cfg.failure_case_weak_ratio
    s = c.scores
    if s:
        if w.hook and w.cliffhanger and s.hook >= strong * w.hook and s.cliffhanger < weak * w.cliffhanger:
            tags.append("strong_hook_weak_ending")
        if w.cliffhanger and w.context and s.cliffhanger >= strong * w.cliffhanger and s.context < weak * w.context:
            tags.append("strong_cliffhanger_weak_context")
        if index is not None:
            n_dialogue = sum(1 for seg in c.segments for e in seg.source_event_ids
                             if e in index.events and index.events[e].event_type == EventType.dialogue)
            weak_story = (w.context and s.context < weak * w.context) or (w.pacing and s.pacing < weak * w.pacing)
            if n_dialogue >= 5 and weak_story:
                tags.append("good_dialogue_poor_standalone_narrative")
    if c.segments and not c.validation.duration_valid:
        tags.append("duration_out_of_range")
    if any("duration" in r for r in c.rejection_reasons):
        tags.append("duration_out_of_range")
    # Timestamp problems only (duration violations have their own tag above).
    ts_words = ("timestamp", "not after start", "timeline", "inconsistent", "mid-line")
    ts_texts = [x for x in c.rejection_reasons + c.warnings + c.deterministic_notes if "duration" not in x]
    if (c.segments and not c.validation.timestamps_valid) or any(any(k in x for k in ts_words) for x in ts_texts):
        tags.append("timestamp_issues")
    if c.cross_episode_coherent is False:
        tags.append("incoherent_cross_episode")
    if any("do not exist" in r or "cannot be found" in r for r in c.rejection_reasons):
        tags.append("hallucinated_source")
    if any(not e.verified for k in ("hook", "cliffhanger", "context", "pacing") for e in (getattr(c, k).evidence if getattr(c, k) else [])):
        tags.append("unverified_evidence")
    c.failure_cases = list(dict.fromkeys(tags))
    return c.failure_cases


def _fmt_segments(c: PromoCandidate) -> str:
    return "; ".join(f"{s.episode_id} {s.start_timestamp}-{s.end_timestamp}" for s in c.segments) or "(no valid segments)"


def build_qa_report(cands: list[PromoCandidate], cfg: ScoringConfig, *, backend: str, episodes: int) -> dict[str, Any]:
    evaluated = [c for c in cands if c.scores is not None]
    ranked = sorted([c for c in cands if c.rank is not None], key=lambda c: c.rank)  # type: ignore[arg-type,return-value]
    rejected = [c for c in cands if c.status == "rejected"]
    dups = [c for c in cands if c.status == "duplicate"]
    totals = [c.scores.total for c in evaluated]  # type: ignore[union-attr]
    fc = Counter(t for c in cands for t in c.failure_cases)
    status = Counter(c.status for c in cands)
    report: dict[str, Any] = {
        "backend": backend,
        "episodes_processed": episodes,
        "candidates_total": len(cands),
        "candidates_evaluated": len(evaluated),
        "status_counts": dict(status),
        "eligible_ranked": len(ranked),
        "passing_all_thresholds": sum(1 for c in ranked if c.passes_thresholds),
        "human_review_required": sum(1 for c in cands if c.human_review_required),
        "score_stats": {
            "mean": round(statistics.mean(totals), 2) if totals else None,
            "median": round(statistics.median(totals), 2) if totals else None,
            "min": min(totals) if totals else None,
            "max": max(totals) if totals else None,
        },
        "origin_counts": dict(Counter(c.origin for c in cands)),
        "failure_case_counts": {FAILURE_CASES.get(k, k): v for k, v in fc.most_common()},
        "selected": [{
            "rank": c.rank, "promo_id": c.promo_id, "title": c.title, "segments": _fmt_segments(c),
            "duration": c.total_duration_seconds, "total": c.scores.total if c.scores else None,
            "why_selected": c.why_it_works, "human_review_required": c.human_review_required,
            "review_reasons": c.review_reasons,
        } for c in ranked[: cfg.top_k_recommendations]],
        "rejected": [{
            "promo_id": c.promo_id, "title": c.title, "segments": _fmt_segments(c), "reasons": c.rejection_reasons,
        } for c in rejected],
        "duplicates": [{"promo_id": c.promo_id, "title": c.title, "duplicate_of": c.duplicate_of} for c in dups],
        "below_threshold": [{
            "promo_id": c.promo_id, "title": c.title, "total": c.scores.total if c.scores else None,
            "reasons": c.review_reasons, "failure_cases": c.failure_cases,
        } for c in ranked if not c.passes_thresholds],
        "disclaimer": "Scores are editorial quality estimates from script text. They are not predictions of views, "
                      "clicks or conversions; no performance data was used.",
    }
    return report


def qa_report_markdown(report: dict[str, Any]) -> str:
    L = ["# Promo QA report", "", f"Backend: `{report['backend']}`", ""]
    L.append(f"- Episodes processed: {report['episodes_processed']}")
    L.append(f"- Candidates: {report['candidates_total']} (evaluated {report['candidates_evaluated']})")
    L.append(f"- Eligible and ranked: {report['eligible_ranked']} (passing all thresholds: {report['passing_all_thresholds']})")
    L.append(f"- Requiring human review: {report['human_review_required']}")
    ss = report["score_stats"]
    L.append(f"- Score mean/median/min/max: {ss['mean']} / {ss['median']} / {ss['min']} / {ss['max']}")
    L += ["", "## Why these promos were selected", ""]
    for s in report["selected"]:
        flag = " (human review)" if s["human_review_required"] else ""
        L.append(f"**#{s['rank']} {s['title']}** [{s['promo_id']}] - {s['total']} pts, {s['duration']:.1f}s{flag}")
        L.append(f"  - Segments: {s['segments']}")
        L.append(f"  - Why: {s['why_selected'] or '(no rationale returned)'}")
        if s["review_reasons"]:
            L.append(f"  - Review: {'; '.join(s['review_reasons'])}")
    L += ["", "## Failure cases detected", ""]
    for k, v in report["failure_case_counts"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Why candidates were rejected", ""]
    for r in report["rejected"]:
        L.append(f"- {r['promo_id']} {r['title']} ({r['segments']}): {'; '.join(r['reasons'])}")
    if report["duplicates"]:
        L += ["", "## Near-duplicates removed from ranking", ""]
        for d in report["duplicates"]:
            L.append(f"- {d['promo_id']} {d['title']} duplicates {d['duplicate_of']}")
    if report["below_threshold"]:
        L += ["", "## Ranked but below thresholds (flagged for review)", ""]
        for b in report["below_threshold"]:
            L.append(f"- {b['promo_id']} {b['title']} ({b['total']} pts): {'; '.join(b['reasons'])}")
    if report.get("notes") or report.get("errors"):
        L += ["", "## Processing notes and errors", ""]
        L += [f"- {n}" for n in report.get("notes", [])]
        L += [f"- ERROR: {e}" for e in report.get("errors", [])]
    L += ["", f"_{report['disclaimer']}_", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Reviewer feedback (stored separately from run outputs)
# ---------------------------------------------------------------------------
Decision = Literal["accepted", "rejected", "needs_editing"]


class FeedbackEntry(BaseModel):
    run_id: str
    promo_id: str
    decision: Decision
    notes: str = ""
    reviewer: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    snapshot: dict[str, Any] = Field(default_factory=dict)


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def add(self, entry: FeedbackEntry) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json() + "\n")

    def load(self, run_id: Optional[str] = None) -> list[FeedbackEntry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = FeedbackEntry.model_validate_json(line)
            except Exception:
                continue
            if run_id is None or e.run_id == run_id:
                out.append(e)
        return out

    def latest(self, run_id: str) -> dict[str, FeedbackEntry]:
        latest: dict[str, FeedbackEntry] = {}
        for e in self.load(run_id):
            latest[e.promo_id] = e
        return latest

    def agreement_summary(self, run_id: str, cands: list[PromoCandidate]) -> dict[str, Any]:
        """How reviewer decisions compare with the system's recommendations."""
        latest = self.latest(run_id)
        by_id = {c.promo_id: c for c in cands}
        rows = Counter()
        for pid, e in latest.items():
            c = by_id.get(pid)
            if c is None:
                continue
            rows[(c.status == "recommended", e.decision)] += 1
        return {
            "reviewed": len(latest),
            "recommended_accepted": rows[(True, "accepted")],
            "recommended_rejected": rows[(True, "rejected")],
            "recommended_needs_editing": rows[(True, "needs_editing")],
            "not_recommended_accepted": rows[(False, "accepted")],
        }


def snapshot(c: PromoCandidate) -> dict[str, Any]:
    return {
        "title": c.title, "segments": [s.model_dump(include={"episode_id", "start_seconds", "end_seconds"}) for s in c.segments],
        "scores": c.scores.model_dump() if c.scores else None, "status": c.status, "rank": c.rank,
    }


def feedback_to_json(entries: list[FeedbackEntry]) -> str:
    return json.dumps([e.model_dump() for e in entries], indent=2)
