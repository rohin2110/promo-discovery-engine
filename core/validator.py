"""Deterministic validation and hallucination prevention.

The model proposes promos as *event ID ranges*. Python then:
1. verifies every referenced episode and event exists in the normalised data
2. derives timestamps from those events (the model never supplies them)
3. checks timestamps are valid and inside the episode timeline
4. computes duration as SUM(end - start) over segments
5. checks ordering, overlap, segment count and cross-episode permissions
6. verifies every evidence quote verbatim against the source event text
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from typing import Iterable, Optional

from config.settings import ScoringConfig
from core.timestamp_utils import format_timestamp, overlap_seconds, segments_total_duration
from models.schemas import (
    Episode, EventType, EvidenceQuote, PromoCandidate, ProposedCandidate, SegmentResult, TimelineEvent,
    ValidationResult, VerifiedEvidence,
)

TOL = 0.05  # seconds of floating-point tolerance


class EpisodeIndex:
    """Fast lookups over the normalised dataset."""

    def __init__(self, episodes: list[Episode]) -> None:
        self.episodes: dict[str, Episode] = {e.episode_id: e for e in episodes}
        self.events: dict[str, TimelineEvent] = {}
        self.event_episode: dict[str, str] = {}
        self.timed: dict[str, list[TimelineEvent]] = {}
        self._starts: dict[str, list[float]] = {}
        for ep in episodes:
            for ev in ep.timeline_events:
                self.events[ev.event_id] = ev
                self.event_episode[ev.event_id] = ep.episode_id
            timed = sorted(ep.timed_events, key=lambda e: (e.start_seconds, e.order))
            self.timed[ep.episode_id] = timed
            self._starts[ep.episode_id] = [e.start_seconds for e in timed]  # type: ignore[misc]

    def resolve_event_id(self, raw: str, default_episode: Optional[str] = None) -> Optional[str]:
        rid = (raw or "").strip().strip("\"'`[]() ")
        if rid in self.events:
            return rid
        up = rid.upper()
        if up in self.events:
            return up
        if default_episode:
            m = re.fullmatch(r"E?0*(\d+)", up)
            if m:
                cand = f"{default_episode}-E{int(m.group(1)):03d}"
                if cand in self.events:
                    return cand
        return None

    def events_overlapping(self, episode_id: str, start: float, end: float) -> list[TimelineEvent]:
        """Timed events that overlap [start, end) (zero-length events if inside)."""
        out = []
        timed = self.timed.get(episode_id, [])
        hi = bisect.bisect_left(self._starts.get(episode_id, []), end)
        for ev in timed[:hi]:
            if ev.end_seconds - ev.start_seconds <= TOL:  # type: ignore[operator]
                if start - TOL <= ev.start_seconds <= end:  # type: ignore[operator]
                    out.append(ev)
            elif overlap_seconds((ev.start_seconds, ev.end_seconds), (start, end)) > TOL:  # type: ignore[arg-type]
                out.append(ev)
        return out

    def end_options(self, episode_id: str, start: float, cfg: ScoringConfig, budget: Optional[float] = None) -> list[TimelineEvent]:
        """End events giving a segment within the duration budget (repair hints)."""
        lo = cfg.min_duration_seconds if budget is None else cfg.min_segment_seconds
        hi = cfg.max_duration_seconds if budget is None else budget
        return [e for e in self.timed.get(episode_id, [])
                if e.event_type != EventType.camera and lo <= e.end_seconds - start <= hi]  # type: ignore[operator]


def _evidence_line(ev: TimelineEvent) -> str:
    if ev.event_type == EventType.dialogue:
        kind = f"DIALOGUE {ev.speaker}"
    elif ev.event_type == EventType.other and ev.label:
        kind = ev.label.upper()
    else:
        kind = ev.event_type.value.upper()
    return f"[{ev.event_id} {format_timestamp(ev.start_seconds)}] {kind}: {ev.text}"


def build_segment(index: EpisodeIndex, start_ev: TimelineEvent, end_ev: TimelineEvent) -> SegmentResult:
    ep_id = index.event_episode[start_ev.event_id]
    ep = index.episodes[ep_id]
    start, end = start_ev.start_seconds, end_ev.end_seconds
    inside = index.events_overlapping(ep_id, start, end)  # type: ignore[arg-type]
    scene_ids = list(dict.fromkeys(e.scene_id for e in inside))
    return SegmentResult(
        episode_id=ep_id, episode_title=ep.episode_title, source_filename=ep.source_filename, scene_ids=scene_ids,
        start_event_id=start_ev.event_id, end_event_id=end_ev.event_id,
        start_timestamp=start_ev.start_timestamp or format_timestamp(start),
        end_timestamp=end_ev.end_timestamp or format_timestamp(end),
        start_seconds=round(start, 3), end_seconds=round(end, 3),  # type: ignore[arg-type]
        duration_seconds=round(end - start, 3),  # type: ignore[operator]
        source_event_ids=[e.event_id for e in inside],
        source_evidence=[_evidence_line(e) for e in inside if e.event_type != EventType.camera],
    )


def resolve_candidate(proposal: ProposedCandidate, index: EpisodeIndex, cfg: ScoringConfig,
                      default_episode: Optional[str] = None) -> tuple[PromoCandidate, list[str], list[str]]:
    """Validate a model proposal and turn it into a PromoCandidate.

    Returns (candidate, errors, repair_hints). Errors become rejection reasons.
    """
    errors: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []
    segments: list[SegmentResult] = []
    source_ok = True
    ts_ok = True

    if not proposal.segments:
        errors.append("candidate has no segments")
    if len(proposal.segments) > cfg.effective_max_segments:
        errors.append(f"{len(proposal.segments)} segments exceed the maximum of {cfg.effective_max_segments}")

    for i, seg in enumerate(proposal.segments, start=1):
        sid = index.resolve_event_id(seg.start_event_id, default_episode)
        eid = index.resolve_event_id(seg.end_event_id, default_episode)
        if sid is None or eid is None:
            missing = [x for x, r in ((seg.start_event_id, sid), (seg.end_event_id, eid)) if r is None]
            errors.append(f"segment {i}: event(s) {missing} do not exist in the source data")
            source_ok = False
            continue
        s_ev, e_ev = index.events[sid], index.events[eid]
        ep_s, ep_e = index.event_episode[sid], index.event_episode[eid]
        if ep_s != ep_e:
            errors.append(f"segment {i}: start ({sid}) and end ({eid}) are in different episodes; a segment must stay inside one episode")
            ts_ok = False
            continue
        bad = [e.event_id for e in (s_ev, e_ev) if not e.is_timed]
        if bad:
            errors.append(f"segment {i}: boundary event(s) {bad} have missing/invalid timestamps")
            ts_ok = False
            continue
        if e_ev.order < s_ev.order or e_ev.end_seconds <= s_ev.start_seconds:  # type: ignore[operator]
            errors.append(f"segment {i}: end event {eid} ({format_timestamp(e_ev.end_seconds)}) is not after start event {sid} ({format_timestamp(s_ev.start_seconds)})")
            ts_ok = False
            continue
        ep = index.episodes[ep_s]
        if (ep.timeline_start is not None and s_ev.start_seconds < ep.timeline_start - TOL) or (  # type: ignore[operator]
                ep.timeline_end is not None and e_ev.end_seconds > ep.timeline_end + TOL):  # type: ignore[operator]
            errors.append(f"segment {i}: falls outside the {ep_s} timeline")
            ts_ok = False
            continue
        built = build_segment(index, s_ev, e_ev)
        if built.duration_seconds < cfg.min_segment_seconds:
            errors.append(f"segment {i}: only {built.duration_seconds:.2f}s long (minimum segment length {cfg.min_segment_seconds:g}s)")
            continue
        untimed_inside = [e.event_id for e in ep.timeline_events if s_ev.order < e.order < e_ev.order and not e.is_timed]
        if untimed_inside:
            warnings.append(f"segment {i} contains {len(untimed_inside)} event(s) with missing/invalid timestamps: {untimed_inside[:5]}")
        flagged = [e.event_id for e in index.events_overlapping(ep_s, built.start_seconds, built.end_seconds)
                   if {"outside_scene_bounds", "out_of_order"} & set(e.flags)]
        if flagged:
            warnings.append(f"segment {i} contains events with inconsistent timestamps: {flagged[:5]}")
        segments.append(built)

    ordered = True
    if len(segments) > 1:
        key = [(index.episodes[s.episode_id].order_index, s.start_seconds) for s in segments]
        if key != sorted(key):
            ordered = False
            if proposal.order_justification.strip():
                warnings.append(f"segments are not chronological; model justification: {proposal.order_justification.strip()}")
            else:
                errors.append("segments are not in chronological order and no justification was given")
        for a in range(len(segments)):
            for b in range(a + 1, len(segments)):
                sa, sb = segments[a], segments[b]
                if sa.episode_id == sb.episode_id and overlap_seconds(
                        (sa.start_seconds, sa.end_seconds), (sb.start_seconds, sb.end_seconds)) > TOL:
                    errors.append(f"segments {a + 1} and {b + 1} overlap in {sa.episode_id} (would double count footage)")
    episode_ids = list(dict.fromkeys(s.episode_id for s in segments))
    if len(episode_ids) > 1 and not cfg.allow_cross_episode_segments:
        errors.append("cross-episode promos are disabled in the configuration")

    total = segments_total_duration((s.start_seconds, s.end_seconds) for s in segments)
    duration_ok = bool(segments) and cfg.min_duration_seconds - TOL <= total <= cfg.max_duration_seconds + TOL
    if segments and not duration_ok:
        errors.append(f"total duration {total:.2f}s is outside the allowed {cfg.min_duration_seconds:g}-{cfg.max_duration_seconds:g}s")
        if len(segments) == 1:
            s_ev = index.events[segments[0].start_event_id]
            opts = index.end_options(segments[0].episode_id, s_ev.start_seconds, cfg)  # type: ignore[arg-type]
            if opts:
                hints.append(
                    f"for start event {s_ev.event_id} ({s_ev.start_seconds:.2f}s) valid end events range from "
                    f"{opts[0].event_id} (ends {opts[0].end_seconds:.2f}s, total {opts[0].end_seconds - s_ev.start_seconds:.1f}s) to "
                    f"{opts[-1].event_id} (ends {opts[-1].end_seconds:.2f}s, total {opts[-1].end_seconds - s_ev.start_seconds:.1f}s)"
                )
            else:
                hints.append(f"no end event after {s_ev.event_id} yields a valid duration; choose an earlier start event")

    if len(segments) <= 1:
        origin = "single_segment"
    elif len(episode_ids) == 1:
        origin = "multi_segment_same_episode"
    else:
        origin = "cross_episode"
    validation = ValidationResult(
        timestamps_valid=ts_ok and bool(segments), duration_valid=duration_ok, source_verified=source_ok,
        segments_ordered=ordered, eligible=not errors, errors=list(errors), warnings=list(warnings),
    )
    cand = PromoCandidate(
        title=proposal.title.strip() or "Untitled promo", origin=origin, episode_ids=episode_ids, segments=segments,
        total_duration_seconds=total, opening_hook=proposal.opening_hook, central_conflict=proposal.central_conflict,
        ending_cliffhanger=proposal.ending_cliffhanger, promotional_reason=proposal.promotional_reason,
        narrative_connection=proposal.narrative_connection, proposed_segments=list(proposal.segments),
        validation=validation, warnings=list(warnings), rejection_reasons=list(errors),
    )
    return cand, errors, hints


# ---------------------------------------------------------------------------
# Evidence verification (quotes must exist verbatim in the clip's events)
# ---------------------------------------------------------------------------
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "..."})


def normalise_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").translate(_QUOTES).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _fragments(quote: str) -> list[str]:
    q = normalise_text(quote).strip(" \"'.,!?;:-")
    parts = [p.strip(" \"'.,!?;:-") for p in re.split(r"\.\.\.+|\[\.\.\.\]", q)]
    return [p for p in parts if len(p) >= 2]


def quote_in_text(quote: str, text: str) -> bool:
    frags = _fragments(quote)
    if not frags:
        return False
    hay = normalise_text(text)
    pos = 0
    for f in frags:
        i = hay.find(f, pos)
        if i < 0:
            return False
        pos = i + len(f)
    return True


def verify_evidence(quotes: Iterable[EvidenceQuote], clip_event_ids: list[str], index: EpisodeIndex) -> list[VerifiedEvidence]:
    clip_set = set(clip_event_ids)
    out: list[VerifiedEvidence] = []
    for q in quotes:
        if not (q.quote or "").strip():
            continue
        eid = index.resolve_event_id(q.event_id) if q.event_id else None
        if eid and eid in clip_set and quote_in_text(q.quote, index.events[eid].text):
            out.append(VerifiedEvidence(event_id=eid, quote=q.quote, verified=True))
            continue
        found = next((e for e in clip_event_ids if quote_in_text(q.quote, index.events[e].text)), None)
        if found:
            note = "" if found == eid else f"quote found in {found} (model cited {q.event_id or 'no event'})"
            out.append(VerifiedEvidence(event_id=found, quote=q.quote, verified=True, note=note))
            continue
        if eid and eid not in clip_set:
            note = f"event {eid} is outside the clip"
        elif q.event_id and not eid:
            note = f"event {q.event_id} does not exist"
        else:
            note = "quote not found verbatim in any event of the clip"
        out.append(VerifiedEvidence(event_id=eid or q.event_id, quote=q.quote, verified=False, note=note))
    return out
