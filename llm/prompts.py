"""Centralised prompt templates (versioned).

The model is only asked for *semantic* judgements: which events form a good
promo and how strong its hook/cliffhanger/context/pacing are on a 0-10 rubric.
It never computes durations, timestamps or weighted scores - Python does.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

from models.schemas import Episode, EventType, TimelineEvent

PROMPT_VERSION = "2026-11-08.1"

SYSTEM_PROMPT = """You are a senior promo editor for short vertical microdrama series.
You read screenplay timelines and pick promotional clips that make new viewers want to watch.

Hard rules:
1. Use ONLY the events provided. Reference events by their exact event_id.
2. Never invent dialogue, actions, characters, scenes or timestamps.
3. Every quote must be copied verbatim from the text of the event you cite.
4. Do not compute durations or weighted scores; the system does that deterministically.
5. Do not claim a promo will achieve views, clicks or conversions.
6. Reply with ONE JSON object only (no markdown, no commentary)."""


def _fmt(sec: Optional[float]) -> str:
    return "?" if sec is None else f"{sec:.2f}"


def render_event(ev: TimelineEvent, *, partial: str = "") -> str:
    if ev.is_timed:
        span = f"[{_fmt(ev.start_seconds)}-{_fmt(ev.end_seconds)}]"
    else:
        span = f"[no valid timestamp: {ev.timestamp_status}; cannot be a cut point]"
    if ev.event_type == EventType.dialogue:
        who = ev.speaker + (f" ({ev.parenthetical})" if ev.parenthetical else "")
        kind = f"DIALOGUE {who}"
    elif ev.event_type == EventType.other:
        kind = (ev.label or "OTHER").upper()
    else:
        kind = ev.event_type.value.upper()
    suffix = f" <{partial}>" if partial else ""
    return f"{ev.event_id} {span} {kind}: {ev.text}{suffix}"


def render_timeline(ep: Episode) -> str:
    lines: list[str] = []
    events_by_scene: dict[str, list[TimelineEvent]] = {}
    for ev in ep.timeline_events:
        events_by_scene.setdefault(ev.scene_id, []).append(ev)
    for sc in ep.scenes:
        evs = events_by_scene.get(sc.scene_id, [])
        if not evs:
            continue
        bounds = f"{_fmt(sc.start_seconds)}-{_fmt(sc.end_seconds)}" if sc.start_seconds is not None else "unknown"
        lines.append(f"## SCENE {sc.scene_id} | {sc.heading or sc.label or 'untitled'} | {bounds}s")
        lines.extend(render_event(e) for e in evs)
    return "\n".join(lines)


def _episode_header(ep: Episode) -> str:
    parts = [f"EPISODE {ep.episode_id} - {ep.episode_title}"]
    if ep.logline:
        parts.append(f"Logline: {ep.logline}")
    if ep.continuity_notes:
        parts.append(f"Continuity notes: {ep.continuity_notes}")
    end = ep.timeline_end if ep.timeline_end is not None else 0
    parts.append(f"Timeline length: {end:.2f}s")
    return "\n".join(parts)


_CANDIDATE_FORMAT = """{"candidates": [{
  "title": "short evocative promo title (no spoilers needed)",
  "segments": [{"start_event_id": "<event_id>", "end_event_id": "<event_id>"}],
  "opening_hook": "what grabs attention in the first seconds",
  "central_conflict": "who is in conflict and why",
  "ending_cliffhanger": "the open question the ending leaves",
  "promotional_reason": "why this could attract a new viewer"%s
}]}"""


def _requirements(cfg, *, max_segments: int, cross: bool) -> str:
    seg_rule = (
        "- Prefer ONE continuous segment. Use multiple segments only when cutting out a slow stretch genuinely "
        f"improves the promo (max {max_segments}). Segments must be chronological and must not overlap."
        if max_segments > 1 else "- Use exactly ONE continuous segment."
    )
    if cross:
        seg_rule = (
            f"- Use 2 to {max_segments} segments taken from at least two DIFFERENT episodes below, in chronological "
            "order (earlier episode first) unless you justify another order in order_justification."
        )
    return f"""PROMO REQUIREMENTS
- A segment runs from the START time of its start event to the END time of its end event and includes every event in between.
- Total runtime of all segments combined: {cfg.min_duration_seconds:g}-{cfg.max_duration_seconds:g} seconds (around {cfg.target_duration_seconds:g}s is typical, but story quality matters more than length). Each segment must be at least {cfg.min_segment_seconds:g}s.
- HOOK: the first {cfg.hook_window_seconds:g}s must grab attention: a revelation, betrayal, confrontation, physical action, accusation, emotional breakdown, surprising line, unexpected arrival, power shift or a question that sparks curiosity. Do not open on a calm establishing shot.
- CLIFFHANGER: the final {cfg.cliffhanger_window_seconds:g}s must leave an open loop: an unanswered question, unresolved confrontation, revelation, imminent danger, discovered secret, interrupted action or a pending decision. Never end on a resolution or an arbitrary/mid-sentence cut.
- CONTEXT: a viewer who has never seen the show must understand who is in conflict and what is at stake from the clip alone.
- PACING: no dead air, redundant lines or unnecessary setup.
{seg_rule}
- Do not choose a moment only because it contains dramatic dialogue; judge the whole progression from hook to cliffhanger.
- Times are in seconds from the start of each episode, e.g. [41.70-43.10]. Segment length = end time of end event - start time of start event."""


def candidate_generation_messages(ep: Episode, cfg, k: int) -> list[dict[str, str]]:
    user = f"""TASK: Propose up to {k} DISTINCT promo candidates from this episode (different moments, not overlapping variations).

{_requirements(cfg, max_segments=cfg.effective_max_segments, cross=False)}

{_episode_header(ep)}

TIMELINE (event_id [start-end seconds] TYPE: text)
{render_timeline(ep)}

Return JSON exactly in this format:
{_CANDIDATE_FORMAT % ''}
If no moment in this episode satisfies the requirements, return {{"candidates": []}}."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cross_episode_messages(episodes: list[Episode], cfg) -> list[dict[str, str]]:
    blocks = "\n\n".join(f"{_episode_header(ep)}\nTIMELINE\n{render_timeline(ep)}" for ep in episodes)
    fmt = _CANDIDATE_FORMAT % (
        ',\n  "narrative_connection": "cite the specific events/characters that link the segments",\n'
        '  "order_justification": "empty unless segments are not chronological"'
    )
    user = f"""TASK: Decide whether moments from these episodes can be combined into {cfg.cross_episode_candidates_per_window} coherent cross-episode promo(s).
Only combine segments when the scripts show a direct narrative link (same characters in the same unresolved situation, a setup paid off later, continuity notes that connect them). Do not create artificial continuity between unrelated scenes. The assembled promo must work as ONE standalone story.

{_requirements(cfg, max_segments=cfg.effective_max_segments, cross=True)}

{blocks}

Return JSON exactly in this format:
{fmt}
If no combination is genuinely coherent and better than a single continuous clip, return {{"candidates": []}}."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def candidate_repair_message(problems: list[dict], cfg) -> str:
    lines = []
    for p in problems:
        lines.append(f'- Candidate "{p["title"]}" segments {json.dumps(p["segments"])}:')
        lines.extend(f"    * {e}" for e in p["errors"])
        lines.extend(f"    hint: {h}" for h in p.get("hints", []))
    return f"""Some candidates failed deterministic validation against the source timeline:
{chr(10).join(lines)}

Fix ONLY these candidates using real event_ids from the timeline above (total runtime {cfg.min_duration_seconds:g}-{cfg.max_duration_seconds:g}s, max {cfg.effective_max_segments} segment(s)). Keep their dramatic intent if possible, otherwise drop them.
Return JSON in the same {{"candidates": [...]}} format containing only the corrected candidates."""


EVALUATION_FORMAT = """{
  "title": "short promo title",
  "hook": {"rating": 0-10, "description": "what happens in the hook window", "evidence": [{"event_id": "...", "quote": "verbatim text"}], "reasoning": "..."},
  "cliffhanger": {"rating": 0-10, "description": "how the clip ends", "evidence": [{"event_id": "...", "quote": "verbatim text"}], "reasoning": "..."},
  "context": {"rating": 0-10, "summary": "what a brand-new viewer understands", "evidence": [{"event_id": "...", "quote": "verbatim text"}], "reasoning": "..."},
  "pacing": {"rating": 0-10, "reasoning": "..."},
  "confidence": 0.0-1.0,
  "weaknesses": ["concrete weakness", "..."],
  "why_it_works": "editorial rationale (no performance predictions)",
  "cross_episode_coherent": true | false | null
}"""


def evaluation_messages(clip_text: str, cfg, *, cross_episode: bool) -> list[dict[str, str]]:
    cross_rule = (
        "\ncross_episode_coherent: this clip joins segments from different episodes. Set true only if the segments "
        "read as one coherent story for a new viewer; set false if the join feels unrelated or confusing (and lower "
        "context/pacing accordingly)." if cross_episode else "\ncross_episode_coherent: null (single-episode clip)."
    )
    user = f"""TASK: Evaluate this promo clip as ONE complete standalone video for a viewer who has NEVER seen the show.
Use ONLY the content inside the clip below. Do not use knowledge of other episodes, loglines or anything outside the clip.

RUBRIC - rate each criterion from 0 to 10 (the system converts ratings into weighted points):
hook (first {cfg.hook_window_seconds:g}s only): 9-10 instantly gripping, clear and surprising; 6-8 strong but slightly slow or unclear; 3-5 mild interest; 0-2 calm, confusing or establishing-only. A weak or confusing opening must be rated low even if the rest is strong.
cliffhanger (final {cfg.cliffhanger_window_seconds:g}s especially): 9-10 powerful open loop, revelation or danger; 6-8 clear unresolved tension; 3-5 mild curiosity; 0-2 resolved, flat, or an arbitrary/mid-sentence cut.
context: 9-10 a newcomer clearly understands who is involved, the conflict and the stakes; 6-8 mostly clear; 3-5 partially confusing; 0-2 incomprehensible without earlier episodes.
pacing: 9-10 every beat escalates, nothing redundant; 6-8 minor slack; 3-5 noticeable dead air or repetition; 0-2 incoherent or dragging.{cross_rule}
confidence: how certain you are of these ratings given only a script (0-1).

Evidence quotes must be copied verbatim from events inside this clip, with their event_id.
Consider the deterministic notes; they were computed from the timestamps.

{clip_text}

Return JSON exactly in this format:
{EVALUATION_FORMAT}"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def evidence_repair_message(bad: Iterable[dict]) -> str:
    lines = "\n".join(f'- {b["criterion"]}: event_id={b["event_id"]!r} quote={b["quote"]!r} ({b["note"]})' for b in bad)
    return f"""These evidence quotes could not be found verbatim in the clip's events:
{lines}

Re-issue the full evaluation JSON. Quote text exactly as it appears in the event you cite, and only cite events inside the clip. Adjust ratings if your reasoning relied on content that is not in the clip."""


def json_repair_message(error: str) -> str:
    return (
        "Your previous reply could not be parsed as the required JSON object. "
        f"Validation error: {error[:800]}\n"
        "Reply again with ONLY the JSON object in the requested format, no markdown fences or commentary."
    )
