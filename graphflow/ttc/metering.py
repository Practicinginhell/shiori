"""Per-agent output-token accounting (the paper compares methods against total output tokens)."""

from __future__ import annotations

import threading
import time
from typing import Any

from graphflow.models.base import BaseModelAdapter, ModelResponse

BUDGET_EXHAUSTED = "[harness] Output-token budget exhausted. Stop now."


def output_tokens(usage: dict[str, Any]) -> int:
    """Output tokens from LangChain (`output_tokens`) or OpenAI-style (`completion_tokens`) usage."""
    return int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)


class TokenMeter:
    def __init__(self, budget: int | None = None, started_at: float | None = None):
        self.budget = budget
        self.started_at = started_at or time.time()
        self.output_tokens = 0
        self.timeline: list[tuple[float, int]] = []  # (elapsed seconds, cumulative output tokens)
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        return self.budget is not None and self.output_tokens >= self.budget

    def add(self, tokens: int) -> None:
        with self._lock:
            self.output_tokens += tokens
            self.timeline.append((time.time() - self.started_at, self.output_tokens))


class MeteredModel(BaseModelAdapter):
    """Wraps a model adapter, counting output tokens and refusing calls once the budget is spent."""

    def __init__(self, inner: BaseModelAdapter, meter: TokenMeter):
        self.inner = inner
        self.meter = meter

    def generate(self, messages, tools=None, structured_output=None, **kwargs) -> ModelResponse:
        if self.meter.exhausted:
            return ModelResponse(content=BUDGET_EXHAUSTED)
        resp = self.inner.generate(messages, tools=tools, structured_output=structured_output, **kwargs)
        self.meter.add(output_tokens(resp.usage))
        return resp

    async def agenerate(self, messages, tools=None, structured_output=None, **kwargs) -> ModelResponse:
        if self.meter.exhausted:
            return ModelResponse(content=BUDGET_EXHAUSTED)
        resp = await self.inner.agenerate(messages, tools=tools, structured_output=structured_output, **kwargs)
        self.meter.add(output_tokens(resp.usage))
        return resp
