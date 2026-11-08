"""Deterministic weighted scoring and eligibility rules.

The model returns a 0-10 rating per criterion. Python converts ratings into
points (rating / 10 * weight), sums the total, applies the configured
thresholds and records every rejection / review reason.
"""
from __future__ import annotations

from config.settings import ScoringConfig
from models.schemas import CriterionAssessment, CriterionResult, PromoCandidate, Scores, VerifiedEvidence

CRITERIA = ("hook", "cliffhanger", "context", "pacing")


def scale_rating(rating: float, weight: float) -> float:
    rating = max(0.0, min(10.0, float(rating)))
    return round(rating / 10.0 * weight, 2)


def build_criterion(assessment: CriterionAssessment, weight: float, evidence: list[VerifiedEvidence]) -> CriterionResult:
    return CriterionResult(
        rating=assessment.rating, score=scale_rating(assessment.rating, weight), max_score=weight,
        description=assessment.description, summary=assessment.summary, evidence=evidence,
        reasoning=assessment.reasoning,
    )


def compute_scores(c: PromoCandidate) -> Scores:
    parts = {k: getattr(c, k).score if getattr(c, k) else 0.0 for k in CRITERIA}
    return Scores(**parts, total=round(sum(parts.values()), 2))


def apply_eligibility(c: PromoCandidate, cfg: ScoringConfig) -> None:
    """Hard rejection rules + human-review flags (never silently discards)."""
    v = c.validation
    # Hard constraints (re-checked here so every path goes through one rule set)
    if c.segments and not (cfg.min_duration_seconds - 0.05 <= c.total_duration_seconds <= cfg.max_duration_seconds + 0.05):
        msg = f"total duration {c.total_duration_seconds:.2f}s outside {cfg.min_duration_seconds:g}-{cfg.max_duration_seconds:g}s"
        if msg not in c.rejection_reasons:
            c.rejection_reasons.append(msg)
        v.duration_valid = False
    if not v.timestamps_valid and "timestamps are invalid" not in c.rejection_reasons and not c.rejection_reasons:
        c.rejection_reasons.append("timestamps are invalid")
    if not v.source_verified and not any("do not exist" in r for r in c.rejection_reasons):
        c.rejection_reasons.append("clip references events that cannot be found in the source")
    v.eligible = not c.rejection_reasons

    if c.scores is None:
        c.passes_thresholds = False
        c.human_review_required = True
        return

    m, s = cfg.minimum_scores, c.scores
    review: list[str] = []
    for name in CRITERIA:
        minimum = getattr(m, name)
        if minimum and getattr(s, name) < minimum:
            review.append(f"{name} score {getattr(s, name):.1f} below minimum {minimum:g}")
    if s.total < m.total:
        review.append(f"total score {s.total:.1f} below minimum {m.total:g}")
    c.passes_thresholds = not review

    unverified = [e for k in CRITERIA for e in (getattr(c, k).evidence if getattr(c, k) else []) if not e.verified]
    if unverified:
        review.append(f"{len(unverified)} evidence quote(s) could not be verified against the source")
    for k in ("hook", "cliffhanger"):
        crit = getattr(c, k)
        if crit is not None and not any(e.verified for e in crit.evidence):
            review.append(f"no verified source evidence for the {k}")
    if c.cross_episode_coherent is False:
        review.append("evaluator judged the cross-episode combination incoherent")
    if v.warnings:
        review.append("validation warnings present")

    # Confidence: model self-report, reduced for objective risk signals.
    conf = c.model_confidence
    conf -= min(0.3, 0.1 * len(unverified))
    conf -= min(0.2, 0.05 * sum(1 for n in c.deterministic_notes if "mid-line" in n or "without dialogue" in n))
    conf -= 0.05 * len(v.warnings)
    c.confidence = round(max(0.0, min(1.0, conf)), 2)
    if c.confidence < cfg.min_confidence_for_auto_accept:
        review.append(f"confidence {c.confidence:.2f} below {cfg.min_confidence_for_auto_accept:g}")

    c.review_reasons = review
    c.human_review_required = bool(review) or not v.eligible
