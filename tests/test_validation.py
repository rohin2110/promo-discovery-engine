import pytest

from config.settings import ScoringConfig
from core.cross_episode_analyser import CharacterLexicon, coherence_check
from core.normalizer import load_dataset
from core.validator import EpisodeIndex, quote_in_text, resolve_candidate, verify_evidence
from models.schemas import EvidenceQuote, EventType, ProposedCandidate, ProposedSegment
from tests.conftest import make_screenplay


@pytest.fixture
def data(app_cfg):
    files = [("episode-001.txt", make_screenplay(1).encode()), ("episode-002.txt", make_screenplay(2).encode()),
             ("episode-003.txt", make_screenplay(3, names=("CASEY", "DREW")).encode())]
    eps, _, _ = load_dataset(files, app_cfg)
    return eps, EpisodeIndex(eps)


def story(ep):
    return [e for e in ep.timed_events if e.event_type != EventType.camera]


def prop(*pairs, **kw):
    return ProposedCandidate(title="t", segments=[ProposedSegment(start_event_id=a, end_event_id=b) for a, b in pairs], **kw)


def pick_end(ep, start, seconds):
    """Story event whose end is closest to start + seconds."""
    return min(story(ep), key=lambda e: abs(e.end_seconds - start.start_seconds - seconds))


def test_single_segment_duration_from_source(data, scoring):
    eps, idx = data
    s = story(eps[0])[1]
    e = pick_end(eps[0], s, 30)
    cand, errors, _ = resolve_candidate(prop((s.event_id, e.event_id)), idx, scoring)
    assert errors == []
    seg = cand.segments[0]
    assert seg.start_seconds == s.start_seconds and seg.end_seconds == e.end_seconds
    assert seg.start_timestamp == s.start_timestamp and seg.end_timestamp == e.end_timestamp
    assert cand.total_duration_seconds == pytest.approx(e.end_seconds - s.start_seconds)
    assert cand.validation.eligible and cand.validation.source_verified and cand.origin == "single_segment"
    assert s.event_id in seg.source_event_ids and e.event_id in seg.source_event_ids


def test_multi_segment_duration_is_sum_of_segments(data, scoring):
    eps, idx = data
    st = story(eps[0])
    a1, a2 = st[1], pick_end(eps[0], st[1], 12)
    b1 = next(e for e in st if e.start_seconds > a2.end_seconds + 5)
    b2 = pick_end(eps[0], b1, 12)
    cand, errors, _ = resolve_candidate(prop((a1.event_id, a2.event_id), (b1.event_id, b2.event_id)), idx, scoring)
    assert errors == [], errors
    expected = (a2.end_seconds - a1.start_seconds) + (b2.end_seconds - b1.start_seconds)
    assert cand.total_duration_seconds == pytest.approx(expected)
    assert cand.total_duration_seconds < b2.end_seconds - a1.start_seconds
    assert cand.origin == "multi_segment_same_episode"


def test_hallucinated_event_ids_rejected(data, scoring):
    eps, idx = data
    cand, errors, _ = resolve_candidate(prop(("EP001-E999", "EP001-E010")), idx, scoring)
    assert errors and not cand.validation.source_verified and not cand.validation.eligible
    assert "do not exist" in errors[0]


def test_short_event_ids_resolved_within_episode(data, scoring):
    eps, idx = data
    s = story(eps[0])[1]
    e = pick_end(eps[0], s, 30)
    short = (s.event_id.split("-")[-1], e.event_id.split("-")[-1])
    cand, errors, _ = resolve_candidate(prop(short), idx, scoring, default_episode="EP001")
    assert errors == [] and cand.segments[0].start_event_id == s.event_id


def test_end_before_start_and_bad_duration(data, scoring):
    eps, idx = data
    st = story(eps[0])
    _, errors, _ = resolve_candidate(prop((st[10].event_id, st[2].event_id)), idx, scoring)
    assert any("not after start" in e for e in errors)
    short_end = pick_end(eps[0], st[1], 8)
    cand, errors, hints = resolve_candidate(prop((st[1].event_id, short_end.event_id)), idx, scoring)
    assert any("outside the allowed" in e for e in errors) and not cand.validation.duration_valid
    assert hints and "valid end events" in hints[0]
    long_cfg = ScoringConfig(min_duration_seconds=5, max_duration_seconds=10, target_duration_seconds=8)
    _, errors, _ = resolve_candidate(prop((st[1].event_id, pick_end(eps[0], st[1], 30).event_id)), idx, long_cfg)
    assert any("outside the allowed" in e for e in errors)


def test_overlapping_and_unordered_segments(data, scoring):
    eps, idx = data
    st = story(eps[0])
    a = (st[1].event_id, pick_end(eps[0], st[1], 15).event_id)
    b = (st[4].event_id, pick_end(eps[0], st[4], 15).event_id)
    _, errors, _ = resolve_candidate(prop(a, b), idx, scoring)
    assert any("overlap" in e for e in errors)
    late = next(e for e in st if e.start_seconds > 30)
    c = (late.event_id, pick_end(eps[0], late, 10).event_id)
    first = (st[1].event_id, pick_end(eps[0], st[1], 12).event_id)
    _, errors, _ = resolve_candidate(prop(c, first), idx, scoring)
    assert any("chronological" in e for e in errors)
    cand, errors, _ = resolve_candidate(prop(c, first, order_justification="flash-forward opening"), idx, scoring)
    assert not any("chronological" in e for e in errors) and cand.validation.warnings


def test_too_many_segments_and_disabled_multi(data):
    eps, idx = data
    st = story(eps[0])
    pairs = [(st[i].event_id, st[i + 1].event_id) for i in (1, 5, 9, 13)]
    cfg = ScoringConfig(max_segments_per_promo=3, min_segment_seconds=1)
    _, errors, _ = resolve_candidate(prop(*pairs), idx, cfg)
    assert any("exceed the maximum" in e for e in errors)
    single = ScoringConfig(allow_multiple_segments=False)
    _, errors, _ = resolve_candidate(prop(*pairs[:2]), idx, single)
    assert any("exceed the maximum of 1" in e for e in errors)


def test_cross_episode_segments(data, scoring):
    eps, idx = data
    s1 = story(eps[0])[1]
    e1 = pick_end(eps[0], s1, 12)
    s2 = story(eps[1])[1]
    e2 = pick_end(eps[1], s2, 15)
    cand, errors, _ = resolve_candidate(prop((s1.event_id, e1.event_id), (s2.event_id, e2.event_id)), idx, scoring)
    assert errors == [] and cand.origin == "cross_episode" and cand.episode_ids == ["EP001", "EP002"]
    ok, notes = coherence_check(cand, idx, CharacterLexicon(eps), scoring)
    assert ok and any("morgan" in n for n in notes)
    # unrelated characters -> not supported
    s3 = story(eps[2])[1]
    e3 = pick_end(eps[2], s3, 15)
    cand2, errors2, _ = resolve_candidate(prop((s2.event_id, e2.event_id), (s3.event_id, e3.event_id)), idx, scoring)
    assert errors2 == []
    ok2, notes2 = coherence_check(cand2, idx, CharacterLexicon(eps), scoring)
    assert not ok2 and any("share no character" in n for n in notes2)
    # disabled in config
    no_cross = ScoringConfig(allow_cross_episode_segments=False)
    _, errors3, _ = resolve_candidate(prop((s1.event_id, e1.event_id), (s2.event_id, e2.event_id)), idx, no_cross)
    assert any("cross-episode" in e for e in errors3)
    # one segment may not straddle two episodes
    _, errors4, _ = resolve_candidate(prop((s1.event_id, e2.event_id)), idx, scoring)
    assert any("different episodes" in e for e in errors4)


def test_untimed_boundary_rejected(app_cfg, scoring):
    text = make_screenplay(1).replace("Start: (00:02.20) | End: (00:05.20)", "Start: (??) | End: (00:05.20)")
    eps, _, _ = load_dataset([("e1.txt", text.encode())], app_cfg)
    idx = EpisodeIndex(eps)
    bad = next(e for e in eps[0].timeline_events if not e.is_timed)
    end = pick_end(eps[0], eps[0].timed_events[3], 30)
    _, errors, _ = resolve_candidate(prop((bad.event_id, end.event_id)), idx, scoring)
    assert any("missing/invalid timestamps" in e for e in errors)


def test_evidence_verification(data):
    eps, idx = data
    ev = next(e for e in eps[0].timeline_events if "lied to me" in e.text)
    clip = [e.event_id for e in eps[0].timeline_events[:12]]
    assert ev.event_id in clip
    out = verify_evidence([
        EvidenceQuote(event_id=ev.event_id, quote="You lied to me about everything!"),
        EvidenceQuote(event_id=ev.event_id, quote="You betrayed our whole family!"),  # paraphrase/hallucination
        EvidenceQuote(event_id="EP001-E001", quote="lied to me"),  # wrong id, correct text
        EvidenceQuote(event_id="EP001-E999", quote="nonexistent"),
        EvidenceQuote(event_id=ev.event_id, quote="You lied ... everything"),  # ellipsis fragments
    ], clip, idx)
    assert [o.verified for o in out] == [True, False, True, False, True]
    assert out[2].event_id == ev.event_id and "found in" in out[2].note
    outside = eps[0].timeline_events[-1]
    res = verify_evidence([EvidenceQuote(event_id=outside.event_id, quote=outside.text[:20])], clip, idx)
    assert not res[0].verified and "outside the clip" in res[0].note


def test_quote_normalisation():
    assert quote_in_text("“Then who really killed her…?”", '"Then who really killed her...?"')
    assert not quote_in_text("killed him", '"Then who really killed her...?"')
    assert not quote_in_text("", "anything")
