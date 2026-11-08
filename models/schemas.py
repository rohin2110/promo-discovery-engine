"""Pydantic data models shared by every layer of the engine.

Three families of models live here:

1. Normalised screenplay data (Episode / Scene / TimelineEvent) produced by the
   universal normalisation layer.
2. LLM I/O contracts (GenerationResponse / EvaluationResponse) that every model
   response is validated against before the pipeline trusts it.
3. Promo output records (PromoCandidate and friends) written to the exports.

Nothing in this module knows about a particular show, episode or character.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Parsing / normalisation
# ---------------------------------------------------------------------------
class EventType(str, Enum):
    dialogue = "dialogue"
    action = "action"
    camera = "camera"
    description = "description"
    other = "other"


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


TimestampStatus = Literal["valid", "missing", "invalid", "ambiguous", "inconsistent"]


class ParseIssue(BaseModel):
    severity: Severity
    source_filename: str
    message: str
    code: str = "generic"
    line_number: Optional[int] = None


class TimelineEvent(BaseModel):
    event_id: str
    scene_id: str
    event_type: EventType
    label: str = ""  # original block label, e.g. "CAMERA" or "NAME (Angry)"
    speaker: str = ""
    parenthetical: str = ""
    text: str  # original source text, preserved verbatim
    start_timestamp: str = ""  # original timestamp string as written in source
    end_timestamp: str = ""
    start_seconds: Optional[float] = None  # None when missing/invalid: never invented
    end_seconds: Optional[float] = None
    source_filename: str
    line_number: Optional[int] = None
    order: int = 0
    timestamp_status: TimestampStatus = "valid"
    flags: list[str] = Field(default_factory=list)

    @property
    def is_timed(self) -> bool:
        return (
            self.timestamp_status == "valid"
            and self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds >= self.start_seconds
        )

    @property
    def duration(self) -> float:
        return (self.end_seconds - self.start_seconds) if self.is_timed else 0.0


class Scene(BaseModel):
    scene_id: str
    label: str = ""  # e.g. "SCENE 01" as written in source
    heading: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    bounds_source: Literal["source", "derived_from_events", "unknown"] = "unknown"
    source_text: str = ""
    event_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    episode_id: str
    episode_number: Optional[int] = None
    episode_title: str = ""
    source_filename: str
    logline: str = ""
    continuity_notes: str = ""
    scenes: list[Scene] = Field(default_factory=list)
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    timeline_start: Optional[float] = None
    timeline_end: Optional[float] = None
    content_duration_seconds: float = 0.0  # union of event intervals (no double counting)
    order_index: int = 0
    warnings: list[str] = Field(default_factory=list)

    @property
    def timed_events(self) -> list[TimelineEvent]:
        return [e for e in self.timeline_events if e.is_timed]

    @property
    def usable_for_promos(self) -> bool:
        return len(self.timed_events) >= 2


class FileRecord(BaseModel):
    display_name: str
    size_bytes: int = 0
    sha256: str = ""
    extension: str = ""
    status: Literal["parsed", "failed", "skipped", "unsupported", "duplicate", "rejected"] = "parsed"
    message: str = ""
    parser: str = ""
    episodes_found: int = 0


class ParseReport(BaseModel):
    files: list[FileRecord] = Field(default_factory=list)
    issues: list[ParseIssue] = Field(default_factory=list)
    episodes_detected: int = 0
    events_total: int = 0
    events_untimed: int = 0
    episode_summaries: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def errors(self) -> list[ParseIssue]:
        return [i for i in self.issues if i.severity == Severity.error]

    @property
    def warnings(self) -> list[ParseIssue]:
        return [i for i in self.issues if i.severity == Severity.warning]


# ---------------------------------------------------------------------------
# LLM response contracts. Lenient on input shape, strict on meaning.
# ---------------------------------------------------------------------------
class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ProposedSegment(_Lenient):
    start_event_id: str
    end_event_id: str


class ProposedCandidate(_Lenient):
    title: str = ""
    segments: list[ProposedSegment] = Field(default_factory=list)
    opening_hook: str = ""
    central_conflict: str = ""
    ending_cliffhanger: str = ""
    promotional_reason: str = ""
    narrative_connection: str = ""
    order_justification: str = ""


class GenerationResponse(_Lenient):
    candidates: list[ProposedCandidate] = Field(default_factory=list)


class EvidenceQuote(_Lenient):
    event_id: str = ""
    quote: str = ""


def _coerce_rating(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    if isinstance(v, str):
        v = v.strip().split("/")[0]
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"rating must be a number 0-10, got {v!r}") from exc
    return max(0.0, min(10.0, f))


class CriterionAssessment(_Lenient):
    rating: float = Field(0.0, description="0-10 rubric rating")
    description: str = ""
    summary: str = ""
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    reasoning: str = ""

    @field_validator("rating", mode="before")
    @classmethod
    def _rating(cls, v: Any) -> float:
        return _coerce_rating(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, (dict, str)):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, str):
                out.append({"event_id": "", "quote": item})
            else:
                out.append(item)
        return out


class EvaluationResponse(_Lenient):
    title: str = ""
    hook: CriterionAssessment
    cliffhanger: CriterionAssessment
    context: CriterionAssessment
    pacing: CriterionAssessment
    confidence: float = 0.5
    weaknesses: list[str] = Field(default_factory=list)
    why_it_works: str = ""
    cross_episode_coherent: Optional[bool] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        if f > 1.0:  # model answered on a 0-100 scale
            f = f / 100.0
        return max(0.0, min(1.0, f))

    @field_validator("weaknesses", mode="before")
    @classmethod
    def _weak(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


# ---------------------------------------------------------------------------
# Promo outputs
# ---------------------------------------------------------------------------
class SegmentResult(BaseModel):
    episode_id: str
    episode_title: str = ""
    source_filename: str = ""
    scene_ids: list[str] = Field(default_factory=list)
    start_event_id: str
    end_event_id: str
    start_timestamp: str
    end_timestamp: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    source_event_ids: list[str] = Field(default_factory=list)
    source_evidence: list[str] = Field(default_factory=list)


class VerifiedEvidence(BaseModel):
    event_id: str = ""
    quote: str = ""
    verified: bool = False
    note: str = ""


class CriterionResult(BaseModel):
    rating: float = 0.0  # model's 0-10 rubric rating
    score: float = 0.0  # Python-scaled: rating / 10 * weight
    max_score: float = 0.0
    description: str = ""
    summary: str = ""
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    reasoning: str = ""


class Scores(BaseModel):
    hook: float = 0.0
    cliffhanger: float = 0.0
    context: float = 0.0
    pacing: float = 0.0
    total: float = 0.0


class ValidationResult(BaseModel):
    timestamps_valid: bool = False
    duration_valid: bool = False
    source_verified: bool = False
    evidence_verified: bool = False
    segments_ordered: bool = False
    eligible: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


CandidateStatus = Literal["recommended", "ranked", "duplicate", "rejected", "unevaluated"]
CandidateOrigin = Literal["single_segment", "multi_segment_same_episode", "cross_episode"]


class PromoCandidate(BaseModel):
    promo_id: str = ""
    rank: Optional[int] = None
    title: str = ""
    origin: CandidateOrigin = "single_segment"
    episode_ids: list[str] = Field(default_factory=list)
    segments: list[SegmentResult] = Field(default_factory=list)
    total_duration_seconds: float = 0.0

    # Proposal (from generation step)
    opening_hook: str = ""
    central_conflict: str = ""
    ending_cliffhanger: str = ""
    promotional_reason: str = ""
    narrative_connection: str = ""
    proposed_segments: list[ProposedSegment] = Field(default_factory=list)

    # Evaluation
    scores: Optional[Scores] = None
    hook: Optional[CriterionResult] = None
    cliffhanger: Optional[CriterionResult] = None
    context: Optional[CriterionResult] = None
    pacing: Optional[CriterionResult] = None
    confidence: float = 0.0
    model_confidence: float = 0.0
    weaknesses: list[str] = Field(default_factory=list)
    why_it_works: str = ""
    cross_episode_coherent: Optional[bool] = None

    # Validation, QA and ranking
    validation: ValidationResult = Field(default_factory=ValidationResult)
    warnings: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    failure_cases: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    passes_thresholds: bool = False
    ranking_score: Optional[float] = None
    duplicate_of: Optional[str] = None
    status: CandidateStatus = "unevaluated"
    deterministic_notes: list[str] = Field(default_factory=list)

    # Traceability
    generation_call_ids: list[str] = Field(default_factory=list)
    evaluation_call_ids: list[str] = Field(default_factory=list)
    backend: str = ""
