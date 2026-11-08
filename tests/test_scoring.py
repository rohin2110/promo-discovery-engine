import pytest
from pydantic import ValidationError

from config.settings import ScoringConfig, Weights, load_scoring_config
from core.deduplicator import mark_near_duplicates, overlap_ratio
from core.quality_assurance import tag_failure_cases
from core.ranking_engine import rank_candidates
from core.scoring_engine import apply_eligibility, build_criterion, compute_scores, scale_rating
from models.schemas import CriterionAssessment, PromoCandidate, SegmentResult, ValidationResult, VerifiedEvidence


def seg(ep="EP001", a=10.0, b=40.0):
    return SegmentResult(episode_id=ep, start_event_id=f"{ep}-E001", end_event_id=f"{ep}-E002",
                         start_timestamp="", end_timestamp="", start_seconds=a, end_seconds=b, duration_seconds=b - a)


def cand(pid, ratings=(8, 8, 8, 8), segments=None, cfg=None, conf=0.9):
    cfg = cfg or ScoringConfig()
    segments = segments or [seg()]
    c = PromoCandidate(promo_id=pid, segments=segments, episode_ids=list(dict.fromkeys(s.episode_id for s in segments)),
                       total_duration_seconds=sum(s.duration_seconds for s in segments), model_confidence=conf,
                       validation=ValidationResult(timestamps_valid=True, duration_valid=True, source_verified=True,
                                                   segments_ordered=True, eligible=True), status="ranked")
    ev = [VerifiedEvidence(event_id="x", quote="q", verified=True)]
    for name, r in zip(("hook", "cliffhanger", "context", "pacing"), ratings):
        setattr(c, name, build_criterion(CriterionAssessment(rating=r), getattr(cfg.weights, name), ev))
    c.scores = compute_scores(c)
    apply_eligibility(c, cfg)
    return c


def test_default_config_file_is_valid():
    cfg = load_scoring_config()
    assert cfg.weights.hook + cfg.weights.cliffhanger + cfg.weights.context + cfg.weights.pacing == 100
    assert cfg.min_duration_seconds == 20 and cfg.max_duration_seconds == 60


def test_weights_must_sum_to_100():
    with pytest.raises(ValidationError, match="sum to 100"):
        Weights(hook=40, cliffhanger=35, context=20, pacing=10)
    Weights(hook=25, cliffhanger=25, context=25, pacing=25)


def test_config_cross_field_validation():
    with pytest.raises(ValidationError, match="greater than min"):
        ScoringConfig(min_duration_seconds=60, max_duration_seconds=20, target_duration_seconds=30)
    with pytest.raises(ValidationError, match="exceeds its weight"):
        ScoringConfig(minimum_scores={"hook": 40, "cliffhanger": 18, "context": 10, "total": 60})
    with pytest.raises(ValidationError):
        ScoringConfig(unknown_parameter=1)


def test_scale_and_total_arithmetic():
    assert scale_rating(8, 35) == 28.0
    assert scale_rating(10, 20) == 20.0
    assert scale_rating(15, 10) == 10.0  # clamped
    c = cand("P1", ratings=(8, 9, 7, 6))
    assert (c.scores.hook, c.scores.cliffhanger, c.scores.context, c.scores.pacing) == (28.0, 31.5, 14.0, 6.0)
    assert c.scores.total == pytest.approx(79.5)


def test_custom_weights_change_points():
    cfg = ScoringConfig(weights={"hook": 25, "cliffhanger": 25, "context": 25, "pacing": 25},
                        minimum_scores={"hook": 10, "cliffhanger": 10, "context": 10, "total": 50})
    c = cand("P1", ratings=(8, 8, 8, 8), cfg=cfg)
    assert c.scores.total == pytest.approx(80.0) and c.scores.hook == 20.0


def test_eligibility_thresholds_flag_review_not_discard():
    c = cand("P1", ratings=(4, 9, 9, 9))  # hook 14 < 18
    assert c.validation.eligible and not c.passes_thresholds and c.human_review_required
    assert any("hook score" in r for r in c.review_reasons)
    low = cand("P2", ratings=(6, 6, 6, 6))  # total 60 -> passes exactly
    assert low.passes_thresholds
    lower = cand("P3", ratings=(5.5, 6, 6, 6))
    assert any("total score" in r for r in lower.review_reasons)


def test_duration_hard_constraint_rejects():
    c = cand("P1", segments=[seg(a=0, b=15)])
    assert not c.validation.eligible and any("outside" in r for r in c.rejection_reasons)
    c2 = cand("P2", segments=[seg(a=0, b=75)])
    assert not c2.validation.eligible


def test_unverified_evidence_and_low_confidence_need_review():
    c = cand("P1", conf=0.3)
    assert c.human_review_required and any("confidence" in r for r in c.review_reasons)
    c2 = cand("P2")
    c2.hook.evidence = [VerifiedEvidence(event_id="x", quote="made up", verified=False, note="not found")]
    apply_eligibility(c2, ScoringConfig())
    assert c2.human_review_required and c2.confidence < 0.9
    assert any("no verified source evidence for the hook" in r for r in c2.review_reasons)


def test_duration_not_rewarded_directly():
    cfg = ScoringConfig()
    short = cand("P1", segments=[seg(a=0, b=30)])
    long = cand("P2", segments=[seg(ep="EP002", a=0, b=50)])
    assert short.scores.total == long.scores.total
    ranked = rank_candidates([long, short], cfg)
    assert ranked[0].promo_id == "P1"  # tie broken only by id, not by duration
    assert short.ranking_score == long.ranking_score


def test_continuous_preferred_when_scores_comparable():
    cfg = ScoringConfig()
    multi = cand("P1", ratings=(8.2, 8.2, 8, 8), segments=[seg(a=0, b=15), seg(a=30, b=45)])
    single = cand("P2", ratings=(8, 8, 8, 8), segments=[seg(ep="EP002")])
    assert multi.scores.total > single.scores.total
    ranked = rank_candidates([multi, single], cfg)
    assert [c.promo_id for c in ranked] == ["P2", "P1"]
    much_better = cand("P3", ratings=(10, 10, 9, 9), segments=[seg(ep="EP003", a=0, b=15), seg(ep="EP003", a=30, b=45)])
    ranked = rank_candidates([much_better, single], cfg)
    assert ranked[0].promo_id == "P3"


def test_threshold_passers_rank_first_and_top_k():
    cfg = ScoringConfig(top_k_recommendations=1)
    weak_hook = cand("P1", ratings=(4, 10, 10, 10), segments=[seg(ep="EP001")])
    ok = cand("P2", ratings=(7, 7, 7, 7), segments=[seg(ep="EP002")])
    ranked = rank_candidates([weak_hook, ok], cfg)
    assert ranked[0].promo_id == "P2" and ranked[0].status == "recommended" and ranked[1].status == "ranked"


def test_near_duplicate_detection():
    a = cand("P1", segments=[seg(a=10, b=40)])
    b = cand("P2", ratings=(7, 7, 7, 7), segments=[seg(a=12, b=42)])
    c = cand("P3", segments=[seg(a=60, b=90)])
    assert overlap_ratio(a, b) > 0.9 and overlap_ratio(a, c) == 0
    dups = mark_near_duplicates([a, b, c], 0.5)
    assert [d[0].promo_id for d in dups] == ["P2"] and b.duplicate_of == "P1" and b.status == "duplicate"
    ranked = rank_candidates([a, b, c], ScoringConfig())
    assert [x.promo_id for x in ranked] == ["P1", "P3"]
    assert "duplicate_moment" in b.failure_cases


def test_failure_case_tags():
    cfg = ScoringConfig()
    hook_only = cand("P1", ratings=(9, 2, 7, 7))
    assert "strong_hook_weak_ending" in tag_failure_cases(hook_only, cfg)
    cliff_no_ctx = cand("P2", ratings=(7, 9, 2, 7))
    assert "strong_cliffhanger_weak_context" in tag_failure_cases(cliff_no_ctx, cfg)
    too_short = cand("P3", segments=[seg(a=0, b=10)])
    assert "duration_out_of_range" in tag_failure_cases(too_short, cfg)


def test_duration_rejection_is_not_tagged_as_timestamp_issue():
    cfg = ScoringConfig()
    c = cand("P1", segments=[seg(a=0, b=10)])
    c.rejection_reasons.append("total duration 10.00s is outside the allowed 20-60s")
    tags = tag_failure_cases(c, cfg)
    assert "duration_out_of_range" in tags and "timestamp_issues" not in tags
