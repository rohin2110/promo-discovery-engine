"""Configuration loading and validation.

* ``ScoringConfig``  - every weight, threshold and selection parameter
  (``config/scoring_config.json``; editable in the UI).
* ``AppConfig``      - ingestion limits, paths and field mappings
  (``config/app_config.json``).
* ``LLMSettings``    - provider/model settings from environment variables
  (``.env``). Completely independent from business logic.

No API key, model, provider, show or episode identifier is hardcoded here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

load_dotenv(PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hook: float = 35
    cliffhanger: float = 35
    context: float = 20
    pacing: float = 10

    @model_validator(mode="after")
    def _sum_to_100(self) -> "Weights":
        for name, v in self.model_dump().items():
            if v < 0:
                raise ValueError(f"weight '{name}' must be >= 0")
        total = self.hook + self.cliffhanger + self.context + self.pacing
        if abs(total - 100) > 1e-6:
            raise ValueError(f"scoring weights must sum to 100 (got {total:g})")
        return self


class MinimumScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hook: float = 18
    cliffhanger: float = 18
    context: float = 10
    pacing: float = 0
    total: float = 60


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_duration_seconds: float = 20
    max_duration_seconds: float = 60
    target_duration_seconds: float = 35
    hook_window_seconds: float = 3
    cliffhanger_window_seconds: float = 5
    weights: Weights = Field(default_factory=Weights)
    minimum_scores: MinimumScores = Field(default_factory=MinimumScores)
    allow_multiple_segments: bool = True
    allow_cross_episode_segments: bool = True
    max_segments_per_promo: int = 3
    prefer_continuous_segments: bool = True
    remove_near_duplicates: bool = True
    top_k_recommendations: int = 10

    # Candidate generation
    candidates_per_episode: int = 3
    min_segment_seconds: float = 3.0
    generation_repair_attempts: int = 2
    evaluation_retry_attempts: int = 1

    # Cross-episode analysis
    cross_episode_max_windows: int = 8
    cross_episode_max_episode_gap: int = 1
    cross_episode_candidates_per_window: int = 1
    cross_episode_require_shared_character: bool = True

    # Ranking / dedupe / QA
    multi_segment_penalty: float = 2.0
    cross_episode_penalty: float = 2.0
    near_duplicate_overlap_threshold: float = 0.5
    min_confidence_for_auto_accept: float = 0.5
    failure_case_strong_ratio: float = 0.7
    failure_case_weak_ratio: float = 0.5

    @model_validator(mode="after")
    def _check(self) -> "ScoringConfig":
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be > 0")
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must be greater than min_duration_seconds")
        if not (self.min_duration_seconds <= self.target_duration_seconds <= self.max_duration_seconds):
            raise ValueError("target_duration_seconds must lie between the min and max duration")
        if self.hook_window_seconds <= 0 or self.cliffhanger_window_seconds <= 0:
            raise ValueError("hook/cliffhanger windows must be > 0")
        if self.max_segments_per_promo < 1:
            raise ValueError("max_segments_per_promo must be >= 1")
        if self.top_k_recommendations < 1:
            raise ValueError("top_k_recommendations must be >= 1")
        if self.candidates_per_episode < 1:
            raise ValueError("candidates_per_episode must be >= 1")
        if not 0 < self.near_duplicate_overlap_threshold <= 1:
            raise ValueError("near_duplicate_overlap_threshold must be in (0, 1]")
        w, m = self.weights, self.minimum_scores
        for name in ("hook", "cliffhanger", "context", "pacing"):
            if getattr(m, name) > getattr(w, name):
                raise ValueError(f"minimum score for '{name}' ({getattr(m, name)}) exceeds its weight ({getattr(w, name)})")
        if m.total > 100:
            raise ValueError("minimum total score cannot exceed 100")
        return self

    @property
    def effective_max_segments(self) -> int:
        return self.max_segments_per_promo if self.allow_multiple_segments else 1


def load_scoring_config(path: Optional[str | Path] = None) -> ScoringConfig:
    path = Path(path or os.getenv("PROMO_SCORING_CONFIG") or CONFIG_DIR / "scoring_config.json")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return ScoringConfig(**data)


# ---------------------------------------------------------------------------
# Application (ingestion limits, paths)
# ---------------------------------------------------------------------------
class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_upload_mb: float = 200
    max_file_mb: float = 25
    max_files: int = 5000
    max_total_uncompressed_mb: float = 500
    max_zip_depth: int = 3
    max_compression_ratio: float = 200
    frames_per_second: Optional[float] = None
    scene_bounds_tolerance_seconds: float = 0.5
    output_dir: str = "outputs"
    raw_output_dir: str = "raw_outputs"
    feedback_path: str = "feedback/reviewer_feedback.jsonl"
    field_mapping: dict[str, list[str]] = Field(default_factory=dict)

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_app_config(path: Optional[str | Path] = None) -> AppConfig:
    path = Path(path or os.getenv("PROMO_APP_CONFIG") or CONFIG_DIR / "app_config.json")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return AppConfig(**data)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
ProviderName = Literal["openai_compatible", "ollama", "heuristic", "mock"]

DEFAULT_API_BASES = {
    "openai_compatible": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434",
}


class LLMSettings(BaseModel):
    provider: ProviderName = "heuristic"
    model: str = ""
    api_key: SecretStr = SecretStr("")
    api_base: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 90
    max_retries: int = 2
    seed: Optional[int] = 42
    response_format: Literal["json_object", "json_schema", "none"] = "json_object"
    structured_output: bool = True
    save_raw_responses: bool = True
    concurrency: int = 1
    context_window: int = 16384  # used by the native Ollama provider (num_ctx)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _norm_provider(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().lower().replace("-", "_")
            aliases = {"openai": "openai_compatible", "openai_compat": "openai_compatible", "offline": "heuristic"}
            return aliases.get(v, v)
        return v

    @model_validator(mode="after")
    def _check(self) -> "LLMSettings":
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens < 256:
            raise ValueError("max_tokens must be >= 256")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.concurrency < 1:
            self.concurrency = 1
        if not self.api_base and self.provider in DEFAULT_API_BASES:
            self.api_base = DEFAULT_API_BASES[self.provider]
        return self

    @property
    def is_llm(self) -> bool:
        return self.provider in ("openai_compatible", "ollama")

    def missing_requirements(self) -> list[str]:
        problems = []
        if self.provider in ("openai_compatible", "ollama"):
            if not self.model:
                problems.append("LLM_MODEL is not set")
            if not self.api_base:
                problems.append("LLM_API_BASE is not set")
        if self.provider == "openai_compatible" and not self.api_key.get_secret_value():
            problems.append("LLM_API_KEY is not set (required by most hosted OpenAI-compatible APIs)")
        return problems

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to persist in run artifacts (API key removed)."""
        d = self.model_dump(exclude={"api_key"})
        d["api_key_set"] = bool(self.api_key.get_secret_value())
        return d


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def load_llm_settings(**overrides: Any) -> LLMSettings:
    extra_raw = _env("LLM_EXTRA_BODY")
    extra: dict[str, Any] = {}
    if extra_raw:
        try:
            extra = json.loads(extra_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_EXTRA_BODY is not valid JSON: {exc}") from exc
    seed = _env("LLM_SEED", "42")
    data: dict[str, Any] = {
        "provider": _env("LLM_PROVIDER", "heuristic"),
        "model": _env("LLM_MODEL", ""),
        "api_key": _env("LLM_API_KEY", ""),
        "api_base": _env("LLM_API_BASE", ""),
        "temperature": float(_env("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(_env("LLM_MAX_TOKENS", "4096")),
        "timeout": float(_env("LLM_TIMEOUT", "90")),
        "max_retries": int(_env("LLM_MAX_RETRIES", "2")),
        "seed": int(seed) if seed not in (None, "", "none") else None,
        "response_format": _env("LLM_RESPONSE_FORMAT", "json_object"),
        "save_raw_responses": (_env("LLM_SAVE_RAW_RESPONSES", "true") or "true").lower() != "false",
        "concurrency": int(_env("LLM_CONCURRENCY", "1")),
        "context_window": int(_env("LLM_CONTEXT_WINDOW", "16384")),
        "extra_body": extra,
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    return LLMSettings(**data)
