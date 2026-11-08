"""Final ranking: Python-only ordering of evaluated, eligible candidates.

Sort key: (passes all thresholds, ranking score, total, confidence).
``ranking_score`` = total minus small configurable penalties for extra
segments and cross-episode joins, so a simple continuous promo wins when the
scores are comparable. Duration is never rewarded directly.
"""
from __future__ import annotations

from config.settings import ScoringConfig
from core.deduplicator import mark_near_duplicates
from models.schemas import PromoCandidate


def ranking_score(c: PromoCandidate, cfg: ScoringConfig) -> float:
    total = c.scores.total if c.scores else 0.0
    if cfg.prefer_continuous_segments:
        total -= cfg.multi_segment_penalty * max(0, len(c.segments) - 1)
        if len(set(c.episode_ids)) > 1:
            total -= cfg.cross_episode_penalty
    return round(total, 2)


def _sort_key(c: PromoCandidate) -> tuple:
    return (
        0 if c.passes_thresholds else 1,
        -(c.ranking_score or 0.0),
        -(c.scores.total if c.scores else 0.0),
        -c.confidence,
        len(c.segments),
        c.promo_id,
    )


def rank_candidates(cands: list[PromoCandidate], cfg: ScoringConfig) -> list[PromoCandidate]:
    """Assign rank/status in place; returns the ranked (eligible) list."""
    pool = []
    for c in cands:
        c.rank = None
        if c.status == "duplicate":
            c.status = "ranked"
            c.duplicate_of = None
            c.failure_cases = [f for f in c.failure_cases if f != "duplicate_moment"]
            c.warnings = [w for w in c.warnings if not w.startswith("near-duplicate of")]
        if c.scores is not None and c.validation.eligible and not c.rejection_reasons:
            c.ranking_score = ranking_score(c, cfg)
            pool.append(c)
        elif c.rejection_reasons:
            c.status = "rejected"
            c.ranking_score = None
    pool.sort(key=_sort_key)
    if cfg.remove_near_duplicates:
        mark_near_duplicates(pool, cfg.near_duplicate_overlap_threshold)
    ranked = [c for c in pool if c.status != "duplicate"]
    for i, c in enumerate(ranked, start=1):
        c.rank = i
        c.status = "recommended" if i <= cfg.top_k_recommendations else "ranked"
    return ranked
