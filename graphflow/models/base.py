"""Graphflow Model abstraction.

Provider-agnostic interface for chat models. Messages are plain dicts in the
common chat format::

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [{"id": str, "name": str, "args": dict}],   # assistant only
     "tool_call_id": str, "name": str}                          # tool only
"""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphflow.tools.base import Tool


@dataclass
class ToolCall:
    """Represents a tool invocation requested by the model."""
    id: str
    name: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "args": self.args}


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None
    usage: dict[str, int] = field(default_factory=dict)
    parsed: Any = None  # Structured output object, when requested and supported

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseModelAdapter(ABC):
    """Abstract interface for all model providers.

    Subclasses must implement ``generate``; everything else has a default.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        structured_output: type | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Synchronously generate a response."""

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        structured_output: type | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Asynchronously generate a response (default: run `generate` in a thread)."""
        return await asyncio.to_thread(
            self.generate, messages, tools=tools, structured_output=structured_output, **kwargs
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream tokens synchronously (default: a single chunk)."""
        yield self.generate(messages, tools=tools, **kwargs).content

    async def astream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens asynchronously (default: a single chunk)."""
        resp = await self.agenerate(messages, tools=tools, **kwargs)
        yield resp.content

    def to_langchain_model(self) -> Any:
        """Bridge to an underlying LangChain chat model, if there is one."""
        raise NotImplementedError(f"{type(self).__name__} has no LangChain equivalent")

    def clone(self) -> BaseModelAdapter:
        """Returns an independent copy of this adapter."""
        return copy.copy(self)
