"""Structured JSON output: extraction, schema validation, repair and raw logging.

Every request/response pair (including failures) is appended to
``raw_outputs/<run_id>/raw_responses.jsonl`` with a call ID that candidates
reference, so any score can be traced back to the exact model output.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config.settings import LLMSettings
from llm.prompts import json_repair_message
from llm.provider import BaseLLMProvider, LLMProviderError, LLMQuotaExhausted, LLMRequest

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply."""
    if text is None:
        raise ValueError("empty response")
    s = _THINK.sub("", text).strip()
    if not s:
        raise ValueError("empty response")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for block in _FENCE.findall(s):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in response")


class RawResponseRecorder:
    """Thread-safe JSONL log of every LLM call."""

    def __init__(self, path: Optional[Path], enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled and path is not None
        self._lock = threading.Lock()
        self._n = 0
        self.records: list[dict[str, Any]] = []
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

    def record(self, entry: dict[str, Any]) -> str:
        with self._lock:
            self._n += 1
            call_id = f"CALL_{self._n:05d}"
            entry = {"call_id": call_id, "timestamp": datetime.now(timezone.utc).isoformat(), **entry}
            summary = {k: entry.get(k) for k in ("call_id", "task", "status", "provider", "model", "latency_seconds", "subject")}
            self.records.append(summary)
            if self.enabled:
                with open(self.path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
                    fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            return call_id


@dataclass
class StructuredResult(Generic[T]):
    parsed: Optional[T] = None
    call_ids: list[str] = field(default_factory=list)
    error: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)  # final conversation (for follow-ups)
    raw_text: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None


class StructuredLLMClient:
    def __init__(self, provider: BaseLLMProvider, settings: LLMSettings, recorder: RawResponseRecorder) -> None:
        self.provider = provider
        self.settings = settings
        self.recorder = recorder
        self.quota_exhausted: str = ""  # set once the provider reports a long-term quota limit
        self.skipped_calls = 0

    @property
    def backend(self) -> str:
        return self.provider.describe()

    def call(self, task: str, messages: list[dict[str, str]], response_model: Type[T], *,
             context: Any = None, subject: str = "") -> StructuredResult[T]:
        result: StructuredResult[T] = StructuredResult()
        if self.quota_exhausted:
            self.skipped_calls += 1
            result.error = f"skipped: provider quota exhausted ({self.quota_exhausted[:200]})"
            return result
        convo = list(messages)
        schema = response_model.model_json_schema()
        max_attempts = 1 + max(0, self.settings.max_retries)
        for attempt in range(1, max_attempts + 1):
            req = LLMRequest(task=task, messages=convo, response_schema=schema, context=context,
                             metadata={"subject": subject, "attempt": attempt})
            t0 = time.monotonic()
            try:
                resp = self.provider.complete(req)
            except LLMProviderError as exc:
                if isinstance(exc, LLMQuotaExhausted):
                    self.quota_exhausted = str(exc)
                cid = self.recorder.record({
                    "task": task, "subject": subject, "attempt": attempt, "status": "provider_error",
                    "provider": self.provider.name, "model": self.provider.model, "request_messages": convo,
                    "error": str(exc), "http_status": exc.status, "latency_seconds": round(time.monotonic() - t0, 3),
                })
                result.call_ids.append(cid)
                result.error = str(exc)
                log.error("LLM call for %s (%s) failed: %s", task, subject, exc)
                return result
            entry: dict[str, Any] = {
                "task": task, "subject": subject, "attempt": attempt, "provider": resp.provider, "model": resp.model,
                "latency_seconds": resp.latency_seconds, "usage": resp.usage, "finish_reason": resp.finish_reason,
                "transport_attempts": resp.attempts, "request_messages": convo, "response_text": resp.text,
            }
            try:
                parsed = response_model.model_validate(extract_json(resp.text))
            except (ValueError, ValidationError) as exc:
                err = str(exc)[:1500]
                entry.update(status="invalid_json", error=err)
                result.call_ids.append(self.recorder.record(entry))
                result.error = f"response did not match the expected JSON schema: {err}"
                log.warning("structured output invalid for %s (%s), attempt %d: %s", task, subject, attempt, err[:200])
                convo = convo + [
                    {"role": "assistant", "content": (resp.text or "")[:6000]},
                    {"role": "user", "content": json_repair_message(err)},
                ]
                continue
            entry.update(status="ok")
            result.call_ids.append(self.recorder.record(entry))
            result.parsed = parsed
            result.raw_text = resp.text
            result.error = ""
            result.messages = convo + [{"role": "assistant", "content": resp.text}]
            return result
        return result
