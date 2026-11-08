"""Format parsers. Each one turns an input document into ``RawEpisode`` objects.

``RawEpisode`` holds raw (string) timestamps exactly as found in the source.
The normaliser (``core/normalizer.py``) converts them to seconds, assigns IDs
and validates them, so every parser stays simple and format-specific.

Add a new format by subclassing ``BaseParser`` and calling ``register_parser``.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from config.settings import AppConfig
from core.ingestion import InputDocument
from core.timestamp_utils import parse_time_range
from models.schemas import EventType, ParseIssue, Severity

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw intermediate representation
# ---------------------------------------------------------------------------
@dataclass
class RawEvent:
    label: str
    event_type: EventType
    speaker: str = ""
    parenthetical: str = ""
    text_lines: list[str] = field(default_factory=list)
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    scene_index: int = -1
    line_number: Optional[int] = None

    @property
    def text(self) -> str:
        return " ".join(t.strip() for t in self.text_lines if t.strip()).strip()


@dataclass
class RawScene:
    label: str = ""
    heading: str = ""
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    line_number: Optional[int] = None
    source_lines: list[str] = field(default_factory=list)


@dataclass
class RawEpisode:
    source_filename: str
    parser: str
    meta: dict[str, str] = field(default_factory=dict)
    scenes: list[RawScene] = field(default_factory=list)
    events: list[RawEvent] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)
    text: str = ""

    def issue(self, severity: Severity, msg: str, code: str, line: Optional[int] = None) -> None:
        self.issues.append(ParseIssue(severity=severity, source_filename=self.source_filename, message=msg, code=code, line_number=line))


class ParserError(Exception):
    """Raised when a document cannot be confidently interpreted."""

    def __init__(self, message: str, code: str = "parse_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ParserContext:
    app_config: AppConfig


# ---------------------------------------------------------------------------
# Label classification (generic screenplay vocabulary, no story knowledge)
# ---------------------------------------------------------------------------
CAMERA_KEYS = {"camera", "cam", "shot", "angle", "camera direction", "camera directions", "visual", "visuals",
               "cut", "cuts", "transition", "framing", "insert", "cinematography"}
ACTION_KEYS = {"action", "actions", "stage direction", "stage directions", "direction", "movement", "blocking"}
DESCRIPTION_KEYS = {"description", "desc", "setting", "scene description", "establishing", "location", "narrative"}
OTHER_KEYS = {"sound", "sound fx", "sfx", "fx", "vfx", "music", "score", "audio", "title card", "text on screen",
              "on-screen text", "on screen text", "super", "caption", "graphic", "graphics", "note", "notes", "ambience"}
_PAREN = re.compile(r"\(([^)]*)\)")
_SPEAKERISH = re.compile(r"^[A-Za-z][A-Za-z0-9 .'’\-&/#]*$")


def classify_label(label: str) -> tuple[EventType, str, str]:
    """Return (event_type, speaker, parenthetical) for a block label."""
    parens = [p.strip() for p in _PAREN.findall(label) if p.strip()]
    base = _PAREN.sub("", label).strip(" :*-\t").strip()
    key = re.sub(r"\s+", " ", base.lower())
    if key in CAMERA_KEYS or key.startswith("camera"):
        return EventType.camera, "", "; ".join(parens)
    if key in ACTION_KEYS:
        return EventType.action, "", "; ".join(parens)
    if key in DESCRIPTION_KEYS:
        return EventType.description, "", "; ".join(parens)
    if key in OTHER_KEYS or key.startswith("sound"):
        return EventType.other, "", "; ".join(parens)
    if base and len(base.split()) <= 6 and _SPEAKERISH.match(base):
        return EventType.dialogue, base, "; ".join(parens)
    return EventType.other, "", "; ".join(parens)


# ---------------------------------------------------------------------------
# Text decoding
# ---------------------------------------------------------------------------
def decode_text(data: bytes, name: str) -> tuple[str, list[ParseIssue]]:
    issues: list[ParseIssue] = []
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = data.decode("cp1252")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
            issues.append(ParseIssue(severity=Severity.warning, source_filename=name,
                                     message="file is not valid UTF-8; decoded with a legacy encoding", code="encoding"))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise ParserError("file looks binary (contains NUL bytes), not a text screenplay", "binary_content")
    return text, issues


# ---------------------------------------------------------------------------
# Parser interface + registry
# ---------------------------------------------------------------------------
class BaseParser(ABC):
    name: str = "base"
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
        ...


PARSER_REGISTRY: dict[str, BaseParser] = {}


def register_parser(parser: BaseParser) -> BaseParser:
    for ext in parser.extensions:
        PARSER_REGISTRY[ext] = parser
    return parser


def get_parser(extension: str) -> Optional[BaseParser]:
    return PARSER_REGISTRY.get(extension.lower())


# ---------------------------------------------------------------------------
# Plain-text / Markdown screenplay parser (also used for PDF and DOCX text)
# ---------------------------------------------------------------------------
_MD_STRIP = re.compile(r"^[#>\s]*|[\s]*$")
_META_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("logline", re.compile(r"^(?:episode\s+)?(?:logline|log line|synopsis|summary)\s*[:\-–]\s*(.+)$", re.I)),
    ("continuity", re.compile(r"^continuity(?:\s*notes?)?\s*[:\-–]\s*(.+)$", re.I)),
    ("title", re.compile(r"^(?:episode\s+)?title\s*[:\-–]\s*(.+)$", re.I)),
    ("show_title", re.compile(r"^(?:show|series)(?:\s*title)?\s*[:\-–]\s*(.+)$", re.I)),
    ("episode_label", re.compile(r"^episode\s*(?:number|no\.?|num|#|id)?\s*[:\-–]\s*(.+)$", re.I)),
]
_EPISODE_HEADING = re.compile(r"^episode\s+(\d+)\s*(?:[:\-–—]\s*(.*))?$", re.I)
_SCENE_MARK = re.compile(r"^scene\s*(?:#\s*)?(?P<num>\d+[a-z]?)\b\s*(?:[:.\-–—]\s*(?P<rest>.*))?$", re.I)
_SCENE_HEADING = re.compile(r"^scene\s*(?:heading|slugline|location)\s*[:\-–]\s*(.+)$", re.I)
_SCENE_START = re.compile(r"^scene\s*start(?:\s*time)?\s*[:\-–]\s*(.+)$", re.I)
_SCENE_END = re.compile(r"^scene\s*end(?:\s*time)?\s*[:\-–]\s*(.+)$", re.I)
_SCENE_RANGE = re.compile(r"^scene\s*(?:time|timecode|timing|range)\s*[:\-–]\s*(.+)$", re.I)
_SLUGLINE = re.compile(r"^(?:\d+[a-z]?\s*[.)]?\s+)?(?:INT|EXT|INT\./EXT|INT/EXT|EXT/INT|I/E|EST)\.?\s+\S")
_IGNORE = re.compile(
    r"^(?:-{2,}\s*(?:script|screenplay)\s*(?:start|end|begins|ends)\s*-{2,}|actions?\s*(?:and|&)\s*dialogue\s*:?|"
    r"[-=*_]{3,}|(?:end of episode|the end|fade (?:in|out)\.?)\s*)$",
    re.I,
)
_BOLD_LABEL = re.compile(r"^\s*(?:[-*]\s+)?\*\*\s*(?P<label>[^*]+?)\s*(?::\s*\*\*|\*\*\s*:)\s*(?P<rest>.*)$")
_PLAIN_LABEL = re.compile(r"^\s*(?P<label>[A-Z][A-Z0-9 .'’\-&/#]{0,60}?(?:\s*\([^)]*\))*)\s*:\s*(?P<rest>.*)$")
_TITLE_LABEL = re.compile(r"^\s*(?P<label>[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}(?:\s*\([^)]*\))*)\s*:\s*(?P<rest>[\"“].*)$")


def _clean(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^\*\*(.*?)\*\*\s*", r"\1 ", s).strip()
    return s.strip("*_ ").strip()


class TextScreenplayParser(BaseParser):
    name = "text_screenplay"
    extensions = (".txt", ".md", ".markdown", ".fountain")

    def parse(self, doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
        text, issues = decode_text(doc.data, doc.display_name)
        ep = self.parse_text(text, doc.display_name, self.name)
        ep.issues[:0] = issues
        return [ep]

    def parse_text(self, text: str, source: str, parser_name: str) -> RawEpisode:
        ep = RawEpisode(source_filename=source, parser=parser_name, text=text)
        if not text.strip():
            raise ParserError("file is empty", "empty_file")

        scene: Optional[RawScene] = None
        current: Optional[RawEvent] = None
        current_open = False  # whether body lines may still be appended
        pending_times: Optional[tuple[str, str]] = None
        labelled_events = 0

        def new_scene(label: str = "", heading: str = "", line_no: Optional[int] = None) -> RawScene:
            sc = RawScene(label=label, heading=heading, line_number=line_no)
            ep.scenes.append(sc)
            return sc

        def new_event(label: str, etype: EventType, speaker: str, paren: str, line_no: int) -> RawEvent:
            nonlocal pending_times, current, current_open
            ev = RawEvent(label=label, event_type=etype, speaker=speaker, parenthetical=paren,
                          scene_index=len(ep.scenes) - 1, line_number=line_no)
            if pending_times:
                ev.start_raw, ev.end_raw = pending_times
                pending_times = None
            ep.events.append(ev)
            current, current_open = ev, True
            return ev

        for idx, raw_line in enumerate(text.split("\n"), start=1):
            line = raw_line.rstrip()
            if scene is not None:
                scene.source_lines.append(raw_line)
            stripped = line.strip()
            if not stripped:
                if current is not None and current.text_lines:
                    current_open = False
                continue
            cleaned = _clean(stripped)
            if _IGNORE.match(cleaned):
                continue

            # --- metadata (only before the first scene/event) ---------------
            if not ep.events and not ep.scenes:
                matched = False
                for key, pat in _META_PATTERNS:
                    m = pat.match(cleaned)
                    if m:
                        ep.meta.setdefault(key, m.group(1).strip())
                        matched = True
                        break
                if matched:
                    continue
                m = _EPISODE_HEADING.match(cleaned)
                if m:
                    ep.meta.setdefault("episode_label", f"Episode {m.group(1)}")
                    if m.group(2):
                        ep.meta.setdefault("title", m.group(2).strip())
                    continue

            # --- scene structure ------------------------------------------
            m = _SCENE_MARK.match(cleaned)
            if m:
                scene = new_scene(label=f"SCENE {m.group('num')}", heading=(m.group("rest") or "").strip(), line_no=idx)
                scene.source_lines.append(raw_line)
                current, current_open = None, False
                continue
            m = _SCENE_HEADING.match(cleaned)
            if m:
                if scene is None or scene.heading:
                    scene = new_scene(line_no=idx)
                    scene.source_lines.append(raw_line)
                scene.heading = m.group(1).strip()
                continue
            m = _SCENE_START.match(cleaned)
            if m and scene is not None:
                scene.start_raw = m.group(1).strip()
                continue
            m = _SCENE_END.match(cleaned)
            if m and scene is not None:
                scene.end_raw = m.group(1).strip()
                continue
            m = _SCENE_RANGE.match(cleaned)
            if m and scene is not None:
                rng = parse_time_range(m.group(1))
                if rng:
                    scene.start_raw, scene.end_raw = rng[0], rng[1]
                    continue
            if _SLUGLINE.match(cleaned) and not (current is not None and current_open and not current.text_lines):
                if scene is not None and not scene.heading and not any(e.scene_index == len(ep.scenes) - 1 for e in ep.events):
                    scene.heading = cleaned
                else:
                    scene = new_scene(heading=cleaned, line_no=idx)
                    scene.source_lines.append(raw_line)
                current, current_open = None, False
                continue

            # --- time ranges ----------------------------------------------
            rng = parse_time_range(stripped)
            if rng:
                start_raw, end_raw, rest = rng
                if rest:
                    lab = _match_label(rest)
                    if lab:
                        label, body = lab
                        et, spk, par = classify_label(label)
                        ev = new_event(label, et, spk, par, idx)
                        labelled_events += 1
                    else:
                        ev = new_event("", EventType.description, "", "", idx)
                        body = rest
                    ev.start_raw, ev.end_raw = start_raw, end_raw
                    if body:
                        ev.text_lines.append(body)
                elif current is not None and current.start_raw is None and current.end_raw is None and not current.text_lines:
                    current.start_raw, current.end_raw = start_raw, end_raw
                else:
                    pending_times = (start_raw, end_raw)
                continue

            # --- labelled blocks ----------------------------------------------
            lab = _match_label(stripped)
            if lab:
                label, body = lab
                et, spk, par = classify_label(label)
                ev = new_event(label, et, spk, par, idx)
                labelled_events += 1
                if body:
                    ev.text_lines.append(body)
                continue

            # --- body text ----------------------------------------------------
            if current is not None and current_open:
                current.text_lines.append(stripped)
                continue
            if scene is None and not ep.events:
                ep.meta.setdefault("preamble", "")
                ep.meta["preamble"] = (ep.meta["preamble"] + " " + stripped).strip()
                continue
            ev = new_event("", EventType.description, "", "", idx)
            ev.text_lines.append(stripped)

        if pending_times:
            ep.issue(Severity.warning, f"timestamp range {pending_times} at end of file has no event", "orphan_timestamp")
        timed = [e for e in ep.events if e.start_raw or e.end_raw]
        if not ep.scenes and labelled_events == 0 and not timed:
            raise ParserError(
                "could not recognise any screenplay structure (no scenes, labelled blocks or timestamps). "
                "Provide a supported layout or a JSON/CSV file with mapped columns.",
                "unrecognised_structure",
            )
        if not ep.events:
            raise ParserError("no actions, dialogue or camera events found", "no_events")
        return ep


def _match_label(line: str) -> Optional[tuple[str, str]]:
    for pat in (_BOLD_LABEL, _PLAIN_LABEL, _TITLE_LABEL):
        m = pat.match(line)
        if m:
            label = m.group("label").strip().rstrip(":").strip()
            if not label or label.lower().startswith(("http", "www")):
                continue
            if pat is _PLAIN_LABEL and len(re.sub(r"[^A-Za-z]", "", _PAREN.sub("", label))) < 2:
                continue
            return label, (m.group("rest") or "").strip()
    return None


# ---------------------------------------------------------------------------
# PDF / DOCX: extract text, then reuse the screenplay text parser
# ---------------------------------------------------------------------------
class PDFParser(BaseParser):
    name = "pdf"
    extensions = (".pdf",)

    def parse(self, doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ParserError("PDF support requires the 'pypdf' package", "missing_dependency") from exc
        try:
            reader = PdfReader(io.BytesIO(doc.data))
            if reader.is_encrypted:
                raise ParserError("PDF is encrypted", "encrypted_pdf")
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except ParserError:
            raise
        except Exception as exc:  # pypdf raises many exception types
            raise ParserError(f"cannot read PDF: {exc}", "bad_pdf") from exc
        if not text.strip():
            raise ParserError("PDF contains no extractable text (scanned image PDFs need OCR, which is not supported)", "pdf_no_text")
        return [TextScreenplayParser().parse_text(text, doc.display_name, self.name)]


class DOCXParser(BaseParser):
    name = "docx"
    extensions = (".docx",)

    def parse(self, doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
        try:
            import docx  # python-docx
        except ImportError as exc:  # pragma: no cover
            raise ParserError("DOCX support requires the 'python-docx' package", "missing_dependency") from exc
        try:
            d = docx.Document(io.BytesIO(doc.data))
        except Exception as exc:
            raise ParserError(f"cannot read DOCX: {exc}", "bad_docx") from exc
        lines = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells))
        return [TextScreenplayParser().parse_text("\n".join(lines), doc.display_name, self.name)]


# ---------------------------------------------------------------------------
# SRT subtitles
# ---------------------------------------------------------------------------
_SRT_SPEAKER = re.compile(r"^\s*(?:\[(?P<b>[^\]]{1,40})\]|(?P<n>[A-Z][A-Za-z .'\-]{0,40}))\s*:\s*(?P<t>.+)$")


class SRTParser(BaseParser):
    name = "srt"
    extensions = (".srt",)

    def parse(self, doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
        text, issues = decode_text(doc.data, doc.display_name)
        ep = RawEpisode(source_filename=doc.display_name, parser=self.name, text=text, issues=issues)
        if not text.strip():
            raise ParserError("file is empty", "empty_file")
        ep.scenes.append(RawScene(label="SUBTITLES", heading="(subtitle track)", line_number=1))
        blocks = re.split(r"\n\s*\n", text.strip())
        line_no = 1
        for block in blocks:
            lines = [l for l in block.split("\n") if l.strip()]
            block_line = line_no
            line_no += block.count("\n") + 2
            if not lines:
                continue
            if lines[0].strip().isdigit():
                lines = lines[1:]
            if not lines:
                continue
            rng = parse_time_range(lines[0])
            if not rng:
                ep.issue(Severity.warning, f"subtitle block without a valid time line: {lines[0][:60]!r}", "srt_block", block_line)
                continue
            body = " ".join(l.strip() for l in lines[1:]).strip()
            if rng[2]:
                body = (rng[2] + " " + body).strip()
            speaker, etype = "", EventType.dialogue
            m = _SRT_SPEAKER.match(body)
            if m:
                speaker = (m.group("b") or m.group("n") or "").strip()
            elif re.fullmatch(r"[\[(].*[\])]", body):
                etype = EventType.other
            ev = RawEvent(label="SUBTITLE", event_type=etype, speaker=speaker, text_lines=[body],
                          start_raw=rng[0], end_raw=rng[1], scene_index=0, line_number=block_line)
            ep.events.append(ev)
        if not ep.events:
            raise ParserError("no valid subtitle cues found", "no_events")
        return [ep]


# ---------------------------------------------------------------------------
# JSON / CSV (structured data with a configurable field mapping)
# ---------------------------------------------------------------------------
def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _resolve_columns(columns: list[str], mapping: dict[str, list[str]]) -> dict[str, Optional[str]]:
    norm = {_norm_key(c): c for c in columns}
    resolved: dict[str, Optional[str]] = {}
    for target, synonyms in mapping.items():
        resolved[target] = next((norm[_norm_key(s)] for s in synonyms if _norm_key(s) in norm), None)
    return resolved


def _event_type_from_value(value: Any, has_speaker: bool) -> tuple[EventType, str]:
    if value is None or str(value).strip() == "":
        return (EventType.dialogue if has_speaker else EventType.description), ""
    v = str(value).strip()
    try:
        return EventType(v.lower()), v
    except ValueError:
        et, _, _ = classify_label(v)
        if et == EventType.dialogue:  # an unknown type word is not a speaker
            et = EventType.other
        return et, v


def rows_to_episodes(rows: list[dict[str, Any]], source: str, parser_name: str, mapping: dict[str, list[str]],
                     text: str = "") -> list[RawEpisode]:
    if not rows:
        raise ParserError("no rows/records found", "no_events")
    columns = list({k for r in rows for k in r.keys()})
    cols = _resolve_columns(columns, mapping)
    missing = [k for k in ("text", "start", "end") if not cols.get(k)]
    if missing:
        raise ParserError(
            f"cannot map required fields {missing}. Columns found: {sorted(columns)}. "
            "Add the matching column names to 'field_mapping' in config/app_config.json.",
            "field_mapping_required",
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = str(r.get(cols["episode"]) if cols.get("episode") else "") or ""
        groups.setdefault(key, []).append(r)
    episodes = []
    for ep_key, ep_rows in groups.items():
        ep = RawEpisode(source_filename=source, parser=parser_name, text=text)
        if ep_key:
            ep.meta["episode_label"] = ep_key
        scene_idx: dict[str, int] = {}
        for i, r in enumerate(ep_rows, start=1):
            scene_key = str(r.get(cols["scene"]) or "") if cols.get("scene") else ""
            if scene_key not in scene_idx:
                scene_idx[scene_key] = len(ep.scenes)
                ep.scenes.append(RawScene(label=scene_key, heading=scene_key if not scene_key.isdigit() else ""))
            speaker = str(r.get(cols["speaker"]) or "").strip() if cols.get("speaker") else ""
            et, label = _event_type_from_value(r.get(cols["event_type"]) if cols.get("event_type") else None, bool(speaker))
            start, end = r.get(cols["start"]), r.get(cols["end"])
            ev = RawEvent(
                label=label or (speaker if speaker else et.value.upper()),
                event_type=et, speaker=speaker if et == EventType.dialogue else "",
                text_lines=[str(r.get(cols["text"]) or "")],
                start_raw=None if start is None or str(start).strip() in ("", "nan") else str(start),
                end_raw=None if end is None or str(end).strip() in ("", "nan") else str(end),
                scene_index=scene_idx[scene_key], line_number=i,
            )
            ep.events.append(ev)
        episodes.append(ep)
    return episodes


def _normalised_episode_to_raw(d: dict[str, Any], source: str) -> RawEpisode:
    """Accept the engine's own normalised Episode JSON as an input format."""
    ep = RawEpisode(source_filename=d.get("source_filename") or source, parser="json_normalised")
    ep.meta.update({
        "episode_label": str(d.get("episode_title") or d.get("episode_id") or ""),
        "title": str(d.get("episode_title") or ""),
        "logline": str(d.get("logline") or ""),
        "continuity": str(d.get("continuity_notes") or ""),
    })
    scene_ids: dict[str, int] = {}
    for s in d.get("scenes") or []:
        scene_ids[str(s.get("scene_id", len(scene_ids)))] = len(ep.scenes)
        ep.scenes.append(RawScene(
            label=str(s.get("label") or s.get("scene_id") or ""), heading=str(s.get("heading") or ""),
            start_raw=_ts_field(s, "start"), end_raw=_ts_field(s, "end"), source_lines=[str(s.get("source_text") or "")],
        ))
    for e in d.get("timeline_events") or []:
        sid = str(e.get("scene_id", ""))
        if sid not in scene_ids:
            scene_ids[sid] = len(ep.scenes)
            ep.scenes.append(RawScene(label=sid))
        try:
            et = EventType(str(e.get("event_type", "other")).lower())
        except ValueError:
            et = EventType.other
        ep.events.append(RawEvent(
            label=str(e.get("label") or e.get("speaker") or et.value.upper()), event_type=et,
            speaker=str(e.get("speaker") or ""), parenthetical=str(e.get("parenthetical") or ""),
            text_lines=[str(e.get("text") or "")], start_raw=_ts_field(e, "start"), end_raw=_ts_field(e, "end"),
            scene_index=scene_ids[sid],
        ))
    if not ep.events:
        raise ParserError("normalised episode JSON has no timeline_events", "no_events")
    return ep


def _ts_field(d: dict[str, Any], which: str) -> Optional[str]:
    raw = d.get(f"{which}_timestamp")
    if raw not in (None, ""):
        return str(raw)
    sec = d.get(f"{which}_seconds")
    return None if sec in (None, "") else str(sec)


class JSONParser(BaseParser):
    name = "json"
    extensions = (".json",)

    def parse(self, doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
        text, _ = decode_text(doc.data, doc.display_name)
        if not text.strip():
            raise ParserError("file is empty", "empty_file")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParserError(f"malformed JSON: {exc}", "malformed_json") from exc
        if isinstance(data, dict) and isinstance(data.get("episodes"), list):
            data = data["episodes"]
        if isinstance(data, dict) and "timeline_events" in data:
            return [_normalised_episode_to_raw(data, doc.display_name)]
        if isinstance(data, list) and data and all(isinstance(x, dict) and "timeline_events" in x for x in data):
            return [_normalised_episode_to_raw(x, doc.display_name) for x in data]
        if isinstance(data, dict):
            for key in ("events", "rows", "lines", "data", "records", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
            return rows_to_episodes(data, doc.display_name, self.name, ctx.app_config.field_mapping, text)
        raise ParserError(
            "unrecognised JSON structure. Expected normalised episodes (with 'timeline_events') or a list of "
            "event records; configure 'field_mapping' in config/app_config.json for custom column names.",
            "unrecognised_json",
        )


class CSVParser(BaseParser):
    name = "csv"
    extensions = (".csv",)

    def parse(self, doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
        text, _ = decode_text(doc.data, doc.display_name)
        if not text.strip():
            raise ParserError("file is empty", "empty_file")
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        try:
            rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
        except csv.Error as exc:
            raise ParserError(f"malformed CSV: {exc}", "malformed_csv") from exc
        rows = [r for r in rows if any((v or "").strip() for v in r.values() if isinstance(v, str))]
        return rows_to_episodes(rows, doc.display_name, self.name, ctx.app_config.field_mapping, text)


for _p in (TextScreenplayParser(), PDFParser(), DOCXParser(), SRTParser(), JSONParser(), CSVParser()):
    register_parser(_p)


def parse_document(doc: InputDocument, ctx: ParserContext) -> list[RawEpisode]:
    parser = get_parser(doc.extension)
    if parser is None:
        raise ParserError(f"no parser registered for '{doc.extension}'", "unsupported_format")
    if not doc.data:
        raise ParserError("file is empty", "empty_file")
    return parser.parse(doc, ctx)
