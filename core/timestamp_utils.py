"""Deterministic timestamp parsing, validation, formatting and interval maths.

Supported formats (all converted to float seconds):

* ``MM:SS``, ``MM:SS.ss``, ``M:SS.s``             (two-part, minutes:seconds)
* ``HH:MM:SS``, ``HH:MM:SS.mmm``, ``HH:MM:SS,mmm`` (three-part, SRT style)
* ``HH:MM:SS:FF``                                  (frames; ambiguous unless fps is known)
* ``12.5``, ``12.5s``, ``12 sec``, ``1m 05.5s``, ``1h2m3s``

Rules:
* Never invent a timestamp. Missing input yields ``status="missing"`` and ``None``.
* Out-of-range components (e.g. ``00:75.00``) are ``invalid``.
* Two-part values are always read as minutes:seconds (screenplay convention).
* Frame-based values are ``ambiguous`` unless a frame rate is configured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

_WS = re.compile(r"\s+")

_HMS = re.compile(r"^(\d{1,3}):(\d{1,2}):(\d{1,2})(?:[.,](\d{1,6}))?$")
_HMSF = re.compile(r"^(\d{1,3}):(\d{2}):(\d{2})[:;](\d{1,3})$")
_MS = re.compile(r"^(\d{1,4}):(\d{1,2})(?:[.,](\d{1,6}))?$")
_SECONDS = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(?:s|sec|secs|second|seconds)?$", re.I)
_COMPOUND = re.compile(
    r"^(?:(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in(?:utes?|s)?)?)?\s*"
    r"(?:(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?|s)?)?)?$",
    re.I,
)

# Token used to find timestamps inside free text (permissive on purpose so that
# malformed values like 00:75.00 are found and then flagged as invalid).
TIMESTAMP_TOKEN = r"\d{1,3}(?::\d{1,2}){1,3}(?:[.,]\d{1,6})?"
_TOKEN_RE = re.compile(rf"(?<![\d:.,]){TIMESTAMP_TOKEN}(?![\d:])")

_RANGE_LABELLED = re.compile(
    r"^\W*start\s*(?:time)?\s*[:=]\s*[\[(]?\s*(?P<start>[^)\]|]*?)\s*[\])]?\s*[|,;/]?\s*"
    r"end\s*(?:time)?\s*[:=]\s*[\[(]?\s*(?P<end>[^)\]|]*?)\s*[\])]?\s*\W*$",
    re.I,
)
_RANGE_ARROW = re.compile(
    rf"^\s*[\[(]?\s*(?P<start>{TIMESTAMP_TOKEN}|\d+(?:\.\d+)?s?)\s*"
    rf"(?:-->|->|–|—|-|to)\s*(?P<end>{TIMESTAMP_TOKEN}|\d+(?:\.\d+)?s?)\s*[\])]?\s*(?P<rest>.*)$",
    re.I,
)


@dataclass(frozen=True)
class ParsedTimestamp:
    raw: str
    seconds: Optional[float]
    status: str  # valid | missing | invalid | ambiguous
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "valid" and self.seconds is not None


def _frac(digits: Optional[str]) -> float:
    return float(f"0.{digits}") if digits else 0.0


def parse_timestamp(raw: Optional[str], fps: Optional[float] = None) -> ParsedTimestamp:
    """Convert one timestamp string to seconds without guessing."""
    if raw is None:
        return ParsedTimestamp("", None, "missing", "no timestamp provided")
    original = str(raw)
    s = original.strip().strip("()[]{}<>").strip()
    s = _WS.sub(" ", s)
    if not s or s.lower() in {"none", "null", "n/a", "na", "-", "--", "?", "tbd", "unknown"}:
        return ParsedTimestamp(original, None, "missing", "no timestamp provided")

    m = _HMSF.match(s)
    if m:
        h, mi, se, ff = (int(x) for x in m.groups())
        if mi >= 60 or se >= 60:
            return ParsedTimestamp(original, None, "invalid", f"minutes/seconds out of range in '{s}'")
        if not fps:
            return ParsedTimestamp(original, None, "ambiguous", f"'{s}' looks like a frame timecode but no frame rate is configured")
        if ff >= fps:
            return ParsedTimestamp(original, None, "invalid", f"frame {ff} exceeds frame rate {fps}")
        return ParsedTimestamp(original, round(h * 3600 + mi * 60 + se + ff / fps, 3), "valid")

    m = _HMS.match(s)
    if m:
        h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mi >= 60 or se >= 60:
            return ParsedTimestamp(original, None, "invalid", f"minutes/seconds out of range in '{s}'")
        return ParsedTimestamp(original, round(h * 3600 + mi * 60 + se + _frac(m.group(4)), 3), "valid")

    m = _MS.match(s)
    if m:
        mi, se = int(m.group(1)), int(m.group(2))
        if se >= 60:
            return ParsedTimestamp(original, None, "invalid", f"seconds component {se} >= 60 in '{s}'")
        return ParsedTimestamp(original, round(mi * 60 + se + _frac(m.group(3)), 3), "valid")

    m = _SECONDS.match(s)
    if m:
        return ParsedTimestamp(original, round(float(m.group(1).replace(",", ".")), 3), "valid")

    m = _COMPOUND.match(s)
    if m and any(m.groups()):
        h, mi, se = (float(x) if x else 0.0 for x in m.groups())
        return ParsedTimestamp(original, round(h * 3600 + mi * 60 + se, 3), "valid")

    if re.search(r"\d", s):
        return ParsedTimestamp(original, None, "invalid", f"unrecognised timestamp format '{s}'")
    return ParsedTimestamp(original, None, "missing", f"no timestamp found in '{s}'")


def extract_timestamps(text: str) -> list[str]:
    """Return every timestamp-like token found in free text, in order."""
    return _TOKEN_RE.findall(text or "")


def parse_time_range(line: str) -> Optional[tuple[str, str, str]]:
    """Detect a start/end time range in a line.

    Returns ``(start_raw, end_raw, remainder)`` or ``None``. Recognises
    ``Start: (00:01.00) | End: (00:04.80)``, ``[00:01 - 00:04]``,
    ``00:00:01,000 --> 00:00:04,800`` and similar.
    """
    if not line or not re.search(r"\d", line):
        return None
    m = _RANGE_LABELLED.match(line.strip())
    if m:
        return m.group("start").strip(), m.group("end").strip(), ""
    m = _RANGE_ARROW.match(line.strip())
    if m:
        return m.group("start").strip(), m.group("end").strip(), (m.group("rest") or "").strip()
    return None


def format_timestamp(seconds: Optional[float]) -> str:
    """Format seconds as ``MM:SS.ss`` (or ``H:MM:SS.ss`` past one hour)."""
    if seconds is None:
        return ""
    total_cs = int(round(seconds * 100))
    h, rem = divmod(total_cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    if h:
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    return f"{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Interval maths (used to avoid double counting overlapping timelines)
# ---------------------------------------------------------------------------
def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ivs = sorted((a, b) for a, b in intervals if a is not None and b is not None and b >= a)
    merged: list[list[float]] = []
    for a, b in ivs:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return round(sum(b - a for a, b in merge_intervals(intervals)), 3)


def overlap_seconds(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def segments_total_duration(segments: Iterable[tuple[float, float]]) -> float:
    """Promo duration = SUM(end - start) for every selected segment.

    Deliberately *not* last_end - first_start: gaps between segments are cut
    out of the edit and must not count towards the runtime.
    """
    return round(sum(max(0.0, e - s) for s, e in segments), 3)
