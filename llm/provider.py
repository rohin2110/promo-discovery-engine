"""LLM provider abstraction.

Business logic only ever sees ``BaseLLMProvider.complete(LLMRequest)``. Which
model answers - a hosted OpenAI-compatible API, a local Ollama server, the
deterministic offline baseline or a scripted test double - is decided purely
by ``LLMSettings`` (environment / UI), never by pipeline code.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import requests

from config.settings import LLMSettings

log = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    task: str
    messages: list[dict[str, str]]
    response_schema: Optional[dict[str, Any]] = None
    context: Any = None  # structured payload for non-LLM providers (heuristic/mock)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_seconds: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    attempts: int = 1
    raw: Optional[dict[str, Any]] = None


class LLMProviderError(Exception):
    def __init__(self, message: str, *, status: Optional[int] = None, body: str = "", attempts: int = 1) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.attempts = attempts


class LLMQuotaExhausted(LLMProviderError):
    """The provider's quota is used up for a long period (e.g. a daily cap).

    Retrying cannot succeed, so the pipeline stops calling the provider."""


_LONG_RETRY = re.compile(r'retry(?:delay)?["\s:]*(?:in\s*)?"?(\d+)h|retrydelay"?\s*:\s*"(\d+)s', re.I)


def is_hard_quota(body_text: str, cap_seconds: float = 300.0) -> bool:
    low = body_text.lower()
    if re.search(r"per ?day|perday|daily", low):
        return True
    m = _LONG_RETRY.search(body_text)
    if m:
        if m.group(1):
            return True
        return float(m.group(2)) > cap_seconds
    return False


class BaseLLMProvider(ABC):
    name = "base"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.model or self.name

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


class OpenAICompatibleProvider(BaseLLMProvider):
    """Any endpoint implementing POST {api_base}/chat/completions."""

    name = "openai_compatible"
    retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    backoff_base = 2.0
    backoff_cap = 30.0

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self._session = requests.Session()
        self._response_format_supported = settings.response_format != "none"
        self._seed_supported = settings.seed is not None
        self._lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.settings.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _body(self, req: LLMRequest) -> dict[str, Any]:
        s = self.settings
        body: dict[str, Any] = {
            "model": s.model,
            "messages": req.messages,
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
        }
        if self._seed_supported and s.seed is not None:
            body["seed"] = s.seed
        if self._response_format_supported and s.structured_output:
            if s.response_format == "json_schema" and req.response_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": req.task, "schema": req.response_schema},
                }
            elif s.response_format in ("json_object", "json_schema"):
                body["response_format"] = {"type": "json_object"}
        body.update(s.extra_body or {})
        return body

    def complete(self, request: LLMRequest) -> LLMResponse:
        url = self.settings.api_base.rstrip("/") + "/chat/completions"
        attempts = 0
        last_error = ""
        max_attempts = 1 + self.settings.max_retries
        while attempts < max_attempts:
            attempts += 1
            body = self._body(request)
            t0 = time.monotonic()
            try:
                resp = self._session.post(url, json=body, headers=self._headers(), timeout=self.settings.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("LLM request failed (attempt %d/%d): %s", attempts, max_attempts, last_error)
                self._sleep(attempts, None)
                continue
            latency = time.monotonic() - t0
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    choice = payload["choices"][0]
                    text = choice["message"].get("content") or ""
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    last_error = f"malformed provider response: {exc}: {resp.text[:300]}"
                    log.warning(last_error)
                    self._sleep(attempts, None)
                    continue
                return LLMResponse(
                    text=text, provider=self.name, model=payload.get("model") or self.settings.model,
                    latency_seconds=round(latency, 3), usage=payload.get("usage") or {},
                    finish_reason=choice.get("finish_reason") or "", attempts=attempts, raw=payload,
                )
            body_text = resp.text[:1000]
            if resp.status_code == 400 and self._downgrade(body_text, body):
                attempts -= 1  # a capability downgrade is not a real retry
                continue
            last_error = f"HTTP {resp.status_code}: {body_text}"
            if resp.status_code == 429 and is_hard_quota(body_text):
                raise LLMQuotaExhausted(f"provider quota exhausted: {body_text[:600]}", status=429, body=body_text, attempts=attempts)
            if resp.status_code in self.retry_statuses:
                log.warning("LLM HTTP %s (attempt %d/%d)", resp.status_code, attempts, max_attempts)
                self._sleep(attempts, resp.headers.get("Retry-After"))
                continue
            raise LLMProviderError(f"LLM request rejected: {last_error}", status=resp.status_code, body=body_text, attempts=attempts)
        raise LLMProviderError(f"LLM request failed after {attempts} attempt(s): {last_error}", attempts=attempts)

    def _downgrade(self, body_text: str, sent: dict[str, Any]) -> bool:
        """Drop optional parameters the provider says it does not support.

        Returns True when the failed request carried a rejected optional field,
        so it can be retried without it. This also covers parallel requests
        that were sent before another thread already downgraded the setting.
        """
        low = body_text.lower()
        with self._lock:
            if "response_format" in sent and ("response_format" in low or "json_object" in low or "json_schema" in low):
                if self._response_format_supported:
                    log.warning("provider rejected response_format; retrying without it")
                self._response_format_supported = False
                return True
            if "seed" in sent and "seed" in low:
                if self._seed_supported:
                    log.warning("provider rejected seed; retrying without it")
                self._seed_supported = False
                return True
        return False

    def _sleep(self, attempt: int, retry_after: Optional[str]) -> None:
        if self.backoff_base <= 0:
            return
        delay = min(self.backoff_cap, self.backoff_base ** attempt)
        if retry_after:
            try:
                delay = min(self.backoff_cap, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)


class OllamaProvider(OpenAICompatibleProvider):
    """Local Ollama server through its native /api/chat endpoint (no key).

    The native API is used because Ollama's OpenAI-compatible endpoint cannot
    set the context window, and the small default (2-4k tokens) silently
    truncates a full episode timeline.
    """

    name = "ollama"

    def _native_base(self) -> str:
        base = self.settings.api_base.rstrip("/")
        return base[:-3] if base.endswith("/v1") else base

    def complete(self, request: LLMRequest) -> LLMResponse:
        s = self.settings
        url = self._native_base() + "/api/chat"
        options: dict[str, Any] = {"temperature": s.temperature, "num_predict": s.max_tokens, "num_ctx": s.context_window}
        if s.seed is not None:
            options["seed"] = s.seed
        body: dict[str, Any] = {"model": s.model, "messages": request.messages, "stream": False, "options": options}
        if s.structured_output and s.response_format != "none":
            body["format"] = request.response_schema if (s.response_format == "json_schema" and request.response_schema) else "json"
        extra = dict(s.extra_body or {})
        options.update(extra.pop("options", {}) or {})
        body.update(extra)
        attempts, last_error = 0, ""
        while attempts < 1 + s.max_retries:
            attempts += 1
            t0 = time.monotonic()
            try:
                resp = self._session.post(url, json=body, timeout=s.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Ollama request failed (attempt %d): %s", attempts, last_error)
                self._sleep(attempts, None)
                continue
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    text = (payload.get("message") or {}).get("content") or ""
                except ValueError as exc:
                    last_error = f"malformed Ollama response: {exc}"
                    continue
                prompt_tokens = payload.get("prompt_eval_count") or 0
                if prompt_tokens and prompt_tokens >= s.context_window - 8:
                    log.warning("prompt used %d of %d context tokens and may have been truncated; raise LLM_CONTEXT_WINDOW",
                                prompt_tokens, s.context_window)
                return LLMResponse(
                    text=text, provider=self.name, model=payload.get("model") or s.model,
                    latency_seconds=round(time.monotonic() - t0, 3),
                    usage={"prompt_tokens": prompt_tokens, "completion_tokens": payload.get("eval_count"),
                           "context_window": s.context_window},
                    finish_reason=payload.get("done_reason") or "", attempts=attempts,
                    raw={k: v for k, v in payload.items() if k != "message"},
                )
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code in self.retry_statuses:
                self._sleep(attempts, None)
                continue
            raise LLMProviderError(f"Ollama request rejected: {last_error}", status=resp.status_code, body=resp.text[:500], attempts=attempts)
        raise LLMProviderError(f"Ollama request failed after {attempts} attempt(s): {last_error}", attempts=attempts)


Responder = Union[list[str], Callable[[LLMRequest], str]]


class MockProvider(BaseLLMProvider):
    """Scripted provider for tests: a list of replies or a callable."""

    name = "mock"

    def __init__(self, settings: LLMSettings, responder: Responder) -> None:
        super().__init__(settings)
        self._responder = responder
        self.requests: list[LLMRequest] = []
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.requests.append(request)
            if callable(self._responder):
                text = self._responder(request)
            else:
                if not self._responder:
                    raise LLMProviderError("mock provider has no scripted responses left")
                text = self._responder.pop(0)
        return LLMResponse(text=text, provider=self.name, model=self.settings.model or "mock")


def build_provider(settings: LLMSettings) -> BaseLLMProvider:
    if settings.provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    if settings.provider == "ollama":
        return OllamaProvider(settings)
    if settings.provider == "heuristic":
        from llm.heuristic_provider import HeuristicProvider

        return HeuristicProvider(settings)
    if settings.provider == "mock":
        raise ValueError("the mock provider must be constructed directly with scripted responses")
    raise ValueError(f"unknown LLM provider '{settings.provider}'")
