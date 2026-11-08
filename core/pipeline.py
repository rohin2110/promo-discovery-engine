"""End-to-end orchestration. Used identically by the Streamlit UI, the CLI and tests.

Stages: file processing -> episode extraction -> event normalisation ->
candidate generation -> candidate validation -> promo evaluation ->
cross-episode analysis -> final ranking -> export.

Every run gets a fresh run ID and its own output directories; nothing from a
previous dataset is reused.
"""
from __future__ import annotations

import json
import logging
import platform
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import AppConfig, LLMSettings, ScoringConfig, load_app_config
from core.candidate_generator import CandidateGenerator
from core.cross_episode_analyser import CrossEpisodeAnalyser, select_windows
from core.normalizer import load_dataset
from core.promo_evaluator import PromoEvaluator
from core.quality_assurance import build_qa_report, qa_report_markdown, tag_failure_cases
from core.ranking_engine import rank_candidates
from core.scoring_engine import apply_eligibility, build_criterion, compute_scores
from core.validator import EpisodeIndex
from llm.prompts import PROMPT_VERSION
from llm.provider import BaseLLMProvider, build_provider
from llm.structured_output import RawResponseRecorder, StructuredLLMClient
from models.schemas import CriterionAssessment, Episode, ParseReport, PromoCandidate
from utils.export import export_run
from utils.logging_config import add_run_file_handler, remove_handler, setup_logging

log = logging.getLogger(__name__)

STAGES = [
    "File processing", "Episode extraction", "Event normalisation", "Candidate generation",
    "Candidate validation", "Promo evaluation", "Cross-episode analysis", "Final ranking", "Export",
]
ProgressCallback = Callable[[str, float, str], None]


def new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


@dataclass
class RunResult:
    run_id: str
    output_dir: Path
    raw_dir: Path
    episodes: list[Episode]
    parse_report: ParseReport
    candidates: list[PromoCandidate]
    qa_report: dict[str, Any]
    backend: str
    files: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def ranked(self) -> list[PromoCandidate]:
        return sorted([c for c in self.candidates if c.rank is not None], key=lambda c: c.rank)  # type: ignore[arg-type,return-value]


class PromoPipeline:
    def __init__(self, scoring: ScoringConfig, llm: LLMSettings, app: Optional[AppConfig] = None, *,
                 provider: Optional[BaseLLMProvider] = None, progress: Optional[ProgressCallback] = None) -> None:
        self.cfg = scoring
        self.llm = llm
        self.app = app or load_app_config()
        self.provider = provider
        self.progress_cb = progress
        setup_logging()

    def _progress(self, stage: str, frac: float, msg: str = "") -> None:
        log.info("[%s] %.0f%% %s", stage, frac * 100, msg)
        if self.progress_cb:
            try:
                self.progress_cb(stage, max(0.0, min(1.0, frac)), msg)
            except Exception:  # UI callbacks must never break the run
                log.debug("progress callback failed", exc_info=True)

    def _map(self, fn: Callable, items: list, stage: str, label: Callable[[Any], str]) -> list:
        results: list = [None] * len(items)
        if not items:
            self._progress(stage, 1.0, "nothing to do")
            return results
        workers = max(1, self.llm.concurrency)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # isolate failures per item
                    log.exception("%s failed for %s", stage, label(items[i]))
                    results[i] = exc
                done += 1
                self._progress(stage, done / len(items), f"{done}/{len(items)} - {label(items[i])}")
        return results

    # ------------------------------------------------------------------
    def run(self, uploads: list[tuple[str, bytes]], *, max_episodes: Optional[int] = None) -> RunResult:
        t0 = time.monotonic()
        run_id = new_run_id()
        out_dir = self.app.resolve(self.app.output_dir) / run_id
        raw_dir = self.app.resolve(self.app.raw_output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        raw_dir.mkdir(parents=True, exist_ok=True)
        handler = add_run_file_handler(out_dir / "run.log")
        errors: list[str] = []
        try:
            log.info("run %s started with %d upload(s)", run_id, len(uploads))
            self._progress("File processing", 0.0, f"{len(uploads)} upload(s)")
            episodes, report, ingestion = load_dataset(uploads, self.app)
            n_docs = len(ingestion.documents)
            self._progress("File processing", 1.0, f"{n_docs} screenplay file(s), {len(report.errors)} error(s)")
            self._progress("Episode extraction", 1.0, f"{len(episodes)} episode(s) detected")
            self._progress("Event normalisation", 1.0, f"{report.events_total} events ({report.events_untimed} without valid timestamps)")
            usable = [e for e in episodes if e.usable_for_promos]
            if max_episodes:
                usable = usable[:max_episodes]
            if not usable:
                errors.append("no episodes with usable timestamps were found; no promos can be generated")
            notes = [f"{e.episode_id} skipped: no usable timestamps" for e in episodes if not e.usable_for_promos]
            notes += [f"{e.episode_id} skipped: timeline shorter than {self.cfg.min_duration_seconds:g}s"
                      for e in usable if (e.timeline_end or 0) - (e.timeline_start or 0) < self.cfg.min_duration_seconds]

            provider = self.provider or build_provider(self.llm)
            recorder = RawResponseRecorder(raw_dir / "raw_responses.jsonl", self.llm.save_raw_responses)
            client = StructuredLLMClient(provider, self.llm, recorder)
            backend = provider.describe()
            index = EpisodeIndex(episodes)

            # Step 1: candidate generation (+ validation-driven repair)
            gen = CandidateGenerator(client, index, self.cfg)
            gen_results = self._map(gen.generate_for_episode, usable, "Candidate generation", lambda e: e.episode_id)
            candidates: list[PromoCandidate] = []
            for ep, res in zip(usable, gen_results):
                if isinstance(res, Exception):
                    errors.append(f"{ep.episode_id}: candidate generation crashed: {res}")
                    continue
                candidates.extend(res)
            self._assign_ids(candidates, 1)
            valid = [c for c in candidates if not c.rejection_reasons]
            self._progress("Candidate validation", 1.0, f"{len(valid)} valid, {len(candidates) - len(valid)} rejected")

            # Step 2: evaluation of every valid candidate as a standalone clip
            evaluator = PromoEvaluator(client, index, self.cfg)
            self._evaluate_all(evaluator, valid, errors)

            # Step 3: cross-episode analysis on the global pool
            cross = CrossEpisodeAnalyser(client, index, self.cfg, episodes)
            if cross.enabled and len(usable) > 1 and not client.quota_exhausted:
                windows = select_windows(usable, candidates, self.cfg)
                cross_res = self._map(cross.propose, windows, "Cross-episode analysis",
                                      lambda w: "+".join(e.episode_id for e in w))
                cross_cands = []
                for w, res in zip(windows, cross_res):
                    if isinstance(res, Exception):
                        errors.append(f"cross-episode analysis crashed for {[e.episode_id for e in w]}: {res}")
                        continue
                    cross_cands.extend(res)
                self._assign_ids(cross_cands, len(candidates) + 1)
                self._evaluate_all(evaluator, [c for c in cross_cands if not c.rejection_reasons], errors, stage="Promo evaluation")
                candidates.extend(cross_cands)
            else:
                self._progress("Cross-episode analysis", 1.0, "skipped (disabled, quota exhausted, or fewer than two usable episodes)")

            if client.quota_exhausted:
                errors.append(
                    f"provider quota exhausted; {client.skipped_calls} later model call(s) were skipped, so some "
                    "episodes/candidates were not processed. Raise the quota or switch model and run again. "
                    f"Provider message: {client.quota_exhausted[:300]}")

            # Step 4: ranking, dedupe, QA
            self._progress("Final ranking", 0.2, "ranking and de-duplicating")
            rank_candidates(candidates, self.cfg)
            for c in candidates:
                tag_failure_cases(c, self.cfg, index)
            qa = build_qa_report(candidates, self.cfg, backend=backend, episodes=len(usable))
            qa["errors"] = errors
            qa["notes"] = notes
            qa["llm_calls"] = len(recorder.records)
            self._progress("Final ranking", 1.0, f"{sum(1 for c in candidates if c.rank)} ranked")

            # Step 5: export
            run_meta = self._run_meta(run_id, backend, uploads, t0)
            inventory = [r.model_dump() for r in report.files]
            files = export_run(out_dir, raw_dir, run_meta=run_meta, episodes=episodes, parse_report=report,
                               inventory=inventory, candidates=candidates, qa_report=qa,
                               qa_markdown=qa_report_markdown(qa))
            self._progress("Export", 1.0, f"artifacts written to {out_dir}")
            elapsed = round(time.monotonic() - t0, 2)
            log.info("run %s finished in %.1fs", run_id, elapsed)
            return RunResult(run_id=run_id, output_dir=out_dir, raw_dir=raw_dir, episodes=episodes, parse_report=report,
                             candidates=candidates, qa_report=qa, backend=backend, files=files, errors=errors,
                             elapsed_seconds=elapsed)
        finally:
            remove_handler(handler)

    def _evaluate_all(self, evaluator: PromoEvaluator, cands: list[PromoCandidate], errors: list[str],
                      stage: str = "Promo evaluation") -> None:
        res = self._map(evaluator.evaluate, cands, stage, lambda c: c.promo_id)
        for c, r in zip(cands, res):
            if isinstance(r, Exception):
                errors.append(f"{c.promo_id}: evaluation crashed: {r}")
                c.status = "unevaluated"
                c.review_reasons.append(f"evaluation crashed: {r}")
                c.human_review_required = True

    @staticmethod
    def _assign_ids(cands: list[PromoCandidate], start: int) -> None:
        for i, c in enumerate(cands, start=start):
            c.promo_id = f"PROMO_{i:03d}"

    def _run_meta(self, run_id: str, backend: str, uploads: list[tuple[str, bytes]], t0: float) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "prompt_version": PROMPT_VERSION,
            "uploads": [{"name": n, "bytes": len(b)} for n, b in uploads],
            "scoring_config": self.cfg.model_dump(),
            "llm_settings": self.llm.public_dict(),
            "app_config": self.app.model_dump(),
            "python": platform.python_version(),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
        }


# ---------------------------------------------------------------------------
# Re-scoring with new weights/thresholds (no new LLM calls)
# ---------------------------------------------------------------------------
def rescore_candidates(cands: list[PromoCandidate], cfg: ScoringConfig, episodes: list[Episode]) -> None:
    """Re-apply weights, thresholds and ranking to stored model ratings."""
    index = EpisodeIndex(episodes)
    w = cfg.weights
    for c in cands:
        if c.scores is None or not c.segments:
            continue
        for name in ("hook", "cliffhanger", "context", "pacing"):
            crit = getattr(c, name)
            if crit is None:
                continue
            assess = CriterionAssessment(rating=crit.rating, description=crit.description, summary=crit.summary,
                                         reasoning=crit.reasoning)
            setattr(c, name, build_criterion(assess, getattr(w, name), crit.evidence))
        c.scores = compute_scores(c)
        c.rejection_reasons = list(c.validation.errors)
        c.validation.duration_valid = cfg.min_duration_seconds - 0.05 <= c.total_duration_seconds <= cfg.max_duration_seconds + 0.05
        cross = len(set(c.episode_ids)) > 1
        if cross and not cfg.allow_cross_episode_segments:
            c.rejection_reasons.append("cross-episode promos are disabled in the configuration")
        if len(c.segments) > cfg.effective_max_segments:
            c.rejection_reasons.append(f"{len(c.segments)} segments exceed the maximum of {cfg.effective_max_segments}")
        c.review_reasons = [r for r in c.review_reasons if r.startswith("evaluation")]
        if c.status in ("rejected", "duplicate", "recommended", "ranked"):
            c.status = "ranked"
        apply_eligibility(c, cfg)
    rank_candidates(cands, cfg)
    for c in cands:
        tag_failure_cases(c, cfg, index)


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load a previous run's artifacts for display."""
    def read(name: str) -> Any:
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    cands_doc = read("candidates_all.json") or {"candidates": []}
    return {
        "run_id": run_dir.name,
        "run_config": read("run_config.json") or {},
        "parse_report": ParseReport.model_validate(read("parse_report.json") or {}),
        "episodes": [Episode.model_validate(e) for e in (read("normalized_episodes.json") or [])],
        "candidates": [PromoCandidate.model_validate(c) for c in cands_doc.get("candidates", [])],
        "qa_report": read("qa_report.json") or {},
        "backend": cands_doc.get("backend", ""),
    }


def list_runs(app: Optional[AppConfig] = None) -> list[Path]:
    app = app or load_app_config()
    root = app.resolve(app.output_dir)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "candidates_all.json").exists()], reverse=True)
