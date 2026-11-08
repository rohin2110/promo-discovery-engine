"""Universal normalisation layer: RawEpisode -> Episode (common schema).

Responsibilities
* detect episode identifiers dynamically (metadata first, then filename)
* convert every timestamp to seconds and flag missing / invalid / ambiguous /
  inconsistent values instead of guessing
* assign stable, dataset-unique scene and event IDs
* compute timelines with interval unions so overlapping camera, action and
  dialogue tracks are never double counted
* produce a parse report that lists every file, error and warning
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

from config.settings import AppConfig
from core.ingestion import IngestionResult, ingest_uploads
from core.parsers import ParserContext, ParserError, RawEpisode, parse_document
from core.timestamp_utils import format_timestamp, merge_intervals, parse_timestamp, union_duration
from models.schemas import Episode, ParseIssue, ParseReport, Scene, Severity, TimelineEvent

log = logging.getLogger(__name__)

_FILENAME_EP = re.compile(r"(?:^|[^a-z0-9])(?:episode|ep|e)[\s_\-.#]*0*(\d{1,4})(?!\d)", re.I)
_LEADING_NUM = re.compile(r"^0*(\d{1,4})(?!\d)")
_LABEL_NUM = re.compile(r"(\d{1,5})")
_MAX_EVENT_ISSUES_PER_FILE = 40


def _stem(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def _natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def detect_episode_number(meta_label: str, filename: str) -> tuple[Optional[int], list[str]]:
    """Episode number from metadata label, falling back to the filename."""
    warnings: list[str] = []
    from_meta: Optional[int] = None
    if meta_label:
        m = _LABEL_NUM.search(meta_label)
        if m:
            from_meta = int(m.group(1))
    stem = _stem(filename)
    m = _FILENAME_EP.search(stem) or _LEADING_NUM.match(stem)
    from_file = int(m.group(1)) if m else None
    if from_meta is not None and from_file is not None and from_meta != from_file:
        warnings.append(f"episode number in metadata ({from_meta}) differs from filename ({from_file}); using metadata")
    number = from_meta if from_meta is not None else from_file
    if number is None:
        warnings.append("no episode number found in metadata or filename; ordering by filename")
    return number, warnings


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40] or "untitled"


def normalize_episodes(raw_eps: list[RawEpisode], app_cfg: AppConfig) -> tuple[list[Episode], list[ParseIssue]]:
    issues: list[ParseIssue] = []
    detected = []
    for raw in raw_eps:
        number, warns = detect_episode_number(raw.meta.get("episode_label", ""), raw.source_filename)
        for w in warns:
            issues.append(ParseIssue(severity=Severity.warning, source_filename=raw.source_filename, message=w, code="episode_id"))
        detected.append((number, raw))
    detected.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0, _natural_key(t[1].source_filename)))

    numbers = [n for n, _ in detected if n is not None]
    width = max(3, len(str(max(numbers)))) if numbers else 3
    used: Counter[str] = Counter()
    episodes: list[Episode] = []
    for order, (number, raw) in enumerate(detected):
        base_id = f"EP{number:0{width}d}" if number is not None else f"EP-{_slug(_stem(raw.source_filename))}"
        used[base_id] += 1
        episode_id = base_id if used[base_id] == 1 else f"{base_id}-{chr(64 + used[base_id])}"
        if used[base_id] > 1:
            issues.append(ParseIssue(
                severity=Severity.warning, source_filename=raw.source_filename, code="duplicate_episode_id",
                message=f"episode identifier {base_id} already used by another file; assigned {episode_id}",
            ))
        ep, ep_issues = _normalize_one(raw, episode_id, number, order, app_cfg)
        issues.extend(ep_issues)
        episodes.append(ep)
    return episodes, issues


def _normalize_one(raw: RawEpisode, episode_id: str, number: Optional[int], order: int,
                   app_cfg: AppConfig) -> tuple[Episode, list[ParseIssue]]:
    issues: list[ParseIssue] = []
    src = raw.source_filename
    fps = app_cfg.frames_per_second
    tol = app_cfg.scene_bounds_tolerance_seconds
    event_issue_count = 0

    def add(sev: Severity, msg: str, code: str, line: Optional[int] = None) -> None:
        nonlocal event_issue_count
        if code.startswith("event_"):
            event_issue_count += 1
            if event_issue_count > _MAX_EVENT_ISSUES_PER_FILE:
                return
        issues.append(ParseIssue(severity=sev, source_filename=src, message=f"[{episode_id}] {msg}", code=code, line_number=line))

    # Implicit scene for events that appear before any scene marker.
    raw_scenes = list(raw.scenes)
    shift = 0
    if any(e.scene_index < 0 for e in raw.events) or not raw_scenes:
        from core.parsers import RawScene
        raw_scenes.insert(0, RawScene(label="", heading="(no scene heading)"))
        shift = 1
        if raw.scenes:
            add(Severity.warning, "events found before the first scene heading; grouped into an implicit scene", "implicit_scene")
        else:
            add(Severity.info, "no scene markers found; all events grouped into one implicit scene", "implicit_scene")

    scenes: list[Scene] = []
    for k, rs in enumerate(raw_scenes, start=1):
        scenes.append(Scene(
            scene_id=f"{episode_id}-S{k:02d}", label=rs.label, heading=rs.heading,
            start_timestamp=rs.start_raw or "", end_timestamp=rs.end_raw or "",
            source_text="\n".join(rs.source_lines).strip(),
        ))

    events: list[TimelineEvent] = []
    for n, re_ in enumerate(raw.events, start=1):
        sidx = re_.scene_index + shift if re_.scene_index >= 0 else 0
        scene = scenes[min(sidx, len(scenes) - 1)]
        ps = parse_timestamp(re_.start_raw, fps)
        pe = parse_timestamp(re_.end_raw, fps)
        flags: list[str] = []
        statuses = {ps.status, pe.status}
        if "invalid" in statuses:
            status = "invalid"
        elif "ambiguous" in statuses:
            status = "ambiguous"
        elif "missing" in statuses:
            status = "missing"
        elif pe.seconds < ps.seconds:  # type: ignore[operator]
            status = "inconsistent"
            flags.append("end_before_start")
        else:
            status = "valid"
        if status != "valid":
            detail = "; ".join(m for m in (ps.message, pe.message) if m) or "end time is before start time"
            sev = Severity.info if status == "missing" else Severity.warning
            add(sev, f"event {n} ({re_.label or re_.event_type.value}) timestamp {status}: {detail}", f"event_timestamp_{status}", re_.line_number)
        text = re_.text
        if not text:
            flags.append("empty_text")
        ev = TimelineEvent(
            event_id=f"{episode_id}-E{n:03d}", scene_id=scene.scene_id, event_type=re_.event_type,
            label=re_.label, speaker=re_.speaker, parenthetical=re_.parenthetical, text=text,
            start_timestamp=(re_.start_raw or "").strip(), end_timestamp=(re_.end_raw or "").strip(),
            start_seconds=ps.seconds, end_seconds=pe.seconds, source_filename=src,
            line_number=re_.line_number, order=n, timestamp_status=status, flags=flags,
        )
        events.append(ev)
        scene.event_ids.append(ev.event_id)

    # Scene bounds: from source when valid, otherwise derived from event times.
    by_scene: dict[str, list[TimelineEvent]] = {}
    for ev in events:
        by_scene.setdefault(ev.scene_id, []).append(ev)
    for sc in scenes:
        ps = parse_timestamp(sc.start_timestamp or None, fps)
        pe = parse_timestamp(sc.end_timestamp or None, fps)
        timed = [e for e in by_scene.get(sc.scene_id, []) if e.is_timed]
        if ps.ok and pe.ok and pe.seconds >= ps.seconds:  # type: ignore[operator]
            sc.start_seconds, sc.end_seconds, sc.bounds_source = ps.seconds, pe.seconds, "source"
        else:
            if sc.start_timestamp or sc.end_timestamp:
                sc.flags.append("invalid_scene_bounds")
                add(Severity.warning, f"scene {sc.label or sc.scene_id} has invalid start/end ({sc.start_timestamp!r}, {sc.end_timestamp!r})", "scene_timestamp_invalid")
            if timed:
                sc.start_seconds = min(e.start_seconds for e in timed)  # type: ignore[type-var]
                sc.end_seconds = max(e.end_seconds for e in timed)  # type: ignore[type-var]
                sc.bounds_source = "derived_from_events"
                sc.flags.append("bounds_derived_from_events")
        if sc.bounds_source == "source":
            for ev in by_scene.get(sc.scene_id, []):
                if ev.is_timed and (ev.start_seconds < sc.start_seconds - tol or ev.end_seconds > sc.end_seconds + tol):  # type: ignore[operator]
                    ev.flags.append("outside_scene_bounds")
                    add(Severity.warning, f"event {ev.event_id} ({format_timestamp(ev.start_seconds)}-{format_timestamp(ev.end_seconds)}) "
                        f"falls outside scene {sc.label or sc.scene_id} bounds", "event_outside_scene", ev.line_number)

    # Out-of-order checks (same track only; parallel tracks may overlap).
    last_start: dict[str, float] = {}
    for ev in events:
        if not ev.is_timed:
            continue
        track = ev.event_type.value
        if track in last_start and ev.start_seconds < last_start[track] - tol:  # type: ignore[operator]
            ev.flags.append("out_of_order")
            add(Severity.warning, f"event {ev.event_id} starts before the previous {track} event (non-chronological)", "event_out_of_order", ev.line_number)
        last_start[track] = max(last_start.get(track, 0.0), ev.start_seconds)  # type: ignore[arg-type]
    timed_scenes = [s for s in scenes if s.start_seconds is not None]
    for prev, cur in zip(timed_scenes, timed_scenes[1:]):
        if cur.start_seconds < prev.end_seconds - tol:  # type: ignore[operator]
            cur.flags.append("overlaps_previous_scene")
            add(Severity.warning, f"scene {cur.label or cur.scene_id} starts before scene {prev.label or prev.scene_id} ends", "scene_overlap")

    timed_events = [e for e in events if e.is_timed]
    bounds = [(e.start_seconds, e.end_seconds) for e in timed_events] + [
        (s.start_seconds, s.end_seconds) for s in scenes if s.bounds_source == "source"
    ]
    warnings: list[str] = []
    if event_issue_count > _MAX_EVENT_ISSUES_PER_FILE:
        add(Severity.warning, f"{event_issue_count - _MAX_EVENT_ISSUES_PER_FILE} further event timestamp issues not listed", "event_issues_truncated")
    untimed = len(events) - len(timed_events)
    if untimed:
        warnings.append(f"{untimed} of {len(events)} events have missing/invalid timestamps and cannot be used as cut points")
    if len(timed_events) < 2:
        warnings.append("episode has no usable timestamps; excluded from promo detection")
        add(Severity.warning, "no usable timestamps; this episode cannot be used for promo timing", "no_timestamps")

    meta = raw.meta
    episode = Episode(
        episode_id=episode_id, episode_number=number,
        episode_title=meta.get("title") or meta.get("episode_label") or _stem(src),
        source_filename=src, logline=meta.get("logline", ""), continuity_notes=meta.get("continuity", ""),
        scenes=scenes, timeline_events=events,
        timeline_start=min((a for a, _ in bounds), default=None),
        timeline_end=max((b for _, b in bounds), default=None),
        content_duration_seconds=union_duration((e.start_seconds, e.end_seconds) for e in timed_events),
        order_index=order, warnings=warnings,
    )
    return episode, issues


# ---------------------------------------------------------------------------
# One-call dataset loader used by the pipeline and the UI
# ---------------------------------------------------------------------------
def load_dataset(uploads: list[tuple[str, bytes]], app_cfg: AppConfig) -> tuple[list[Episode], ParseReport, IngestionResult]:
    ingestion = ingest_uploads(uploads, app_cfg)
    ctx = ParserContext(app_config=app_cfg)
    raw_eps: list[RawEpisode] = []
    issues = list(ingestion.issues)
    records = {r.display_name: r for r in ingestion.records}
    for doc in ingestion.documents:
        rec = records.get(doc.display_name)
        try:
            eps = parse_document(doc, ctx)
        except ParserError as exc:
            issues.append(ParseIssue(severity=Severity.error, source_filename=doc.display_name, message=str(exc), code=exc.code))
            if rec:
                rec.status, rec.message = "failed", str(exc)
            continue
        except Exception as exc:  # never let one bad file kill the run
            log.exception("unexpected parser failure for %s", doc.display_name)
            issues.append(ParseIssue(severity=Severity.error, source_filename=doc.display_name,
                                     message=f"unexpected parser error: {exc}", code="parser_crash"))
            if rec:
                rec.status, rec.message = "failed", f"unexpected parser error: {exc}"
            continue
        for ep in eps:
            issues.extend(ep.issues)
        raw_eps.extend(eps)
        if rec:
            rec.parser = eps[0].parser if eps else ""
            rec.episodes_found = len(eps)
    episodes, norm_issues = normalize_episodes(raw_eps, app_cfg)
    issues.extend(norm_issues)
    report = ParseReport(
        files=list(records.values()), issues=issues, episodes_detected=len(episodes),
        events_total=sum(len(e.timeline_events) for e in episodes),
        events_untimed=sum(len(e.timeline_events) - len(e.timed_events) for e in episodes),
        episode_summaries=[{
            "episode_id": e.episode_id, "episode_title": e.episode_title, "source_filename": e.source_filename,
            "scenes": len(e.scenes), "events": len(e.timeline_events), "timed_events": len(e.timed_events),
            "timeline": f"{format_timestamp(e.timeline_start)}-{format_timestamp(e.timeline_end)}",
            "content_duration_seconds": e.content_duration_seconds, "usable_for_promos": e.usable_for_promos,
            "warnings": e.warnings,
        } for e in episodes],
    )
    return episodes, report, ingestion
