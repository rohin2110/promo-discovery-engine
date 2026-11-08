"""Builds the exact content of a promo clip for evaluation.

The evaluator only ever sees what is *inside* the proposed edit (no logline,
no continuity notes, no other episodes), so the context score reflects what a
brand-new viewer would actually see. Python also pre-computes the hook and
cliffhanger windows and objective notes such as mid-line cuts and dead air.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import ScoringConfig
from core.validator import TOL, EpisodeIndex
from llm.prompts import render_event
from models.schemas import EventType, PromoCandidate, SegmentResult, TimelineEvent

SPOKEN = {EventType.dialogue}
STORY = {EventType.dialogue, EventType.action, EventType.description}


@dataclass
class ClipSegmentView:
    segment: SegmentResult
    events: list[TimelineEvent]
    edit_start: float  # where this segment starts on the promo's own timeline


@dataclass
class ClipView:
    title: str
    segments: list[ClipSegmentView]
    total_duration: float
    hook_events: list[TimelineEvent]
    cliff_events: list[TimelineEvent]
    notes: list[str] = field(default_factory=list)
    cross_episode: bool = False

    @property
    def event_ids(self) -> list[str]:
        return list(dict.fromkeys(e.event_id for s in self.segments for e in s.events))

    @property
    def all_events(self) -> list[TimelineEvent]:
        seen, out = set(), []
        for s in self.segments:
            for e in s.events:
                if e.event_id not in seen:
                    seen.add(e.event_id)
                    out.append(e)
        return out

    def to_prompt_text(self, cfg: ScoringConfig) -> str:
        lines = [f"PROMO CLIP: {len(self.segments)} segment(s), total runtime {self.total_duration:.2f}s"]
        lines.append(f"\nHOOK WINDOW (first {cfg.hook_window_seconds:g}s of the promo):")
        lines += [f"  {render_event(e)}" for e in self.hook_events] or ["  (no events)"]
        lines.append("\nFULL CLIP IN EDIT ORDER:")
        for i, sv in enumerate(self.segments, start=1):
            s = sv.segment
            if i > 1:
                lines.append("  --- CUT ---")
            lines.append(f"  Segment {i}: {s.episode_id} {s.start_seconds:.2f}-{s.end_seconds:.2f}s ({s.duration_seconds:.2f}s)")
            for e in sv.events:
                partial = ""
                if e.start_seconds < s.start_seconds - TOL:  # type: ignore[operator]
                    partial = "begins before the cut"
                elif e.end_seconds > s.end_seconds + TOL:  # type: ignore[operator]
                    partial = "cut off before it ends"
                lines.append(f"    {render_event(e, partial=partial)}")
        lines.append(f"\nCLIFFHANGER WINDOW (final {cfg.cliffhanger_window_seconds:g}s of the promo):")
        lines += [f"  {render_event(e)}" for e in self.cliff_events] or ["  (no events)"]
        if self.notes:
            lines.append("\nDETERMINISTIC NOTES:")
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def _window(views: list[ClipSegmentView], index: EpisodeIndex, start_edit: float, end_edit: float) -> list[TimelineEvent]:
    """Events overlapping an interval of the promo's edit timeline."""
    out: list[TimelineEvent] = []
    for v in views:
        seg = v.segment
        a = max(start_edit, v.edit_start)
        b = min(end_edit, v.edit_start + seg.duration_seconds)
        if b <= a:
            continue
        src_a = seg.start_seconds + (a - v.edit_start)
        src_b = seg.start_seconds + (b - v.edit_start)
        for e in index.events_overlapping(seg.episode_id, src_a, src_b):
            if e not in out:
                out.append(e)
    return out


def build_clip_view(cand: PromoCandidate, index: EpisodeIndex, cfg: ScoringConfig) -> ClipView:
    views: list[ClipSegmentView] = []
    t = 0.0
    for seg in cand.segments:
        evs = index.events_overlapping(seg.episode_id, seg.start_seconds, seg.end_seconds)
        views.append(ClipSegmentView(segment=seg, events=evs, edit_start=t))
        t += seg.duration_seconds
    total = round(t, 3)
    hook = _window(views, index, 0.0, min(cfg.hook_window_seconds, total))
    cliff = _window(views, index, max(0.0, total - cfg.cliffhanger_window_seconds), total)
    notes = clip_notes(views, hook, cliff)
    return ClipView(title=cand.title, segments=views, total_duration=total, hook_events=hook, cliff_events=cliff,
                    notes=notes, cross_episode=len({s.episode_id for s in cand.segments}) > 1)


def clip_notes(views: list[ClipSegmentView], hook: list[TimelineEvent], cliff: list[TimelineEvent]) -> list[str]:
    notes: list[str] = []
    if not any(e.event_type in STORY for e in hook):
        notes.append("hook window contains no dialogue or action (camera/sound only)")
    if not any(e.event_type in STORY for e in cliff):
        notes.append("cliffhanger window contains no dialogue or action (camera/sound only)")
    for i, v in enumerate(views, start=1):
        s = v.segment
        for e in v.events:
            if e.event_type not in SPOKEN:
                continue
            if e.start_seconds < s.start_seconds - 0.2 and e.end_seconds > s.start_seconds + 0.2:  # type: ignore[operator]
                notes.append(f"segment {i} starts mid-line: {e.event_id} began at {e.start_seconds:.2f}s")
            if e.start_seconds < s.end_seconds - 0.2 and e.end_seconds > s.end_seconds + 0.2:  # type: ignore[operator]
                notes.append(f"segment {i} cuts dialogue {e.event_id} mid-line at {s.end_seconds:.2f}s")
        story = sorted((e.start_seconds, e.end_seconds) for e in v.events if e.event_type in STORY)  # type: ignore[misc]
        cursor = s.start_seconds
        for a, b in story:
            if a - cursor > 6.0:
                notes.append(f"segment {i} has {a - cursor:.1f}s without dialogue or action ({cursor:.2f}-{a:.2f}s)")
            cursor = max(cursor, b)
        if s.end_seconds - cursor > 6.0:
            notes.append(f"segment {i} ends with {s.end_seconds - cursor:.1f}s without dialogue or action")
    for a, b in zip(views, views[1:]):
        sa, sb = a.segment, b.segment
        if sa.episode_id == sb.episode_id:
            notes.append(f"cut skips {sb.start_seconds - sa.end_seconds:.1f}s of {sa.episode_id} between segments")
        else:
            notes.append(f"cut jumps from {sa.episode_id} to {sb.episode_id}")
    return notes
