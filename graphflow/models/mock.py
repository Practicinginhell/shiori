"""Deterministic MockModel for offline testing and reproduction."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any

from graphflow.models.base import BaseModelAdapter, ModelResponse, ToolCall

if TYPE_CHECKING:
    from graphflow.tools.base import Tool

MockResponse = str | ModelResponse | dict[str, Any]


class MockModel(BaseModelAdapter):
    """Deterministic, zero-cost mock model for unit and workflow testing.

    Args:
        responses: A single response, a list of responses returned sequentially
            (the last one repeats once exhausted), or a callable mapping
            messages -> response. A response is a string, a ModelResponse, or a
            dict ``{"content": ..., "tool_calls": [{"name": ..., "args": {...}}]}``.

    Structured output is simulated by returning JSON content; the Agent parses it.
    """

    def __init__(
        self,
        responses: list[MockResponse] | Callable[[list[Any]], MockResponse] | MockResponse = "Mock response",
    ):
        if isinstance(responses, (str, ModelResponse, dict)):
            self._responses: list[MockResponse] = [responses]
            self._callable: Callable[[list[Any]], MockResponse] | None = None
        elif callable(responses):
            self._responses = []
            self._callable = responses
        else:
            self._responses = list(responses)
            self._callable = None
        self._call_count = 0
        self.history: list[list[Any]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @staticmethod
    def _coerce(resp: MockResponse) -> ModelResponse:
        if isinstance(resp, ModelResponse):
            return resp
        if isinstance(resp, dict):
            tool_calls = [
                ToolCall(id=tc.get("id", f"call_{i}"), name=tc["name"], args=tc.get("args", {}))
                for i, tc in enumerate(resp.get("tool_calls", []))
            ]
            return ModelResponse(
                content=resp.get("content", ""),
                tool_calls=tool_calls,
                usage={"prompt_tokens": 10, "completion_tokens": 10},
            )
        return ModelResponse(content=str(resp), usage={"prompt_tokens": 10, "completion_tokens": 10})

    def _get_next_response(self, messages: list[Any]) -> ModelResponse:
        self.history.append(list(messages))
        self._call_count += 1

        if self._callable is not None:
            return self._coerce(self._callable(messages))

        if not self._responses:
            return ModelResponse(content=f"Default mock response {self._call_count}")

        idx = min(self._call_count - 1, len(self._responses) - 1)
        return self._coerce(self._responses[idx])

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        structured_output: type | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        return self._get_next_response(messages)

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        structured_output: type | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        await asyncio.sleep(0)
        return self._get_next_response(messages)

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        resp = self.generate(messages, tools=tools, **kwargs)
        words = resp.content.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    async def astream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        resp = await self.agenerate(messages, tools=tools, **kwargs)
        words = resp.content.split(" ")
        for i, word in enumerate(words):
            await asyncio.sleep(0)
            yield word + (" " if i < len(words) - 1 else "")

    def clone(self) -> MockModel:
        """Returns a copy with the same scripted responses and a fresh call history."""
        cloned = copy.copy(self)
        cloned._responses = list(self._responses)
        cloned._call_count = 0
        cloned.history = []
        return cloned

    def to_langchain_model(self) -> Any:
        """Create a LangChain-compatible chat model wrapping this mock model."""
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        outer = self

        def _result(resp: ModelResponse) -> ChatResult:
            msg = AIMessage(
                content=resp.content,
                tool_calls=[{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        class _LangChainMockChat(BaseChatModel):
            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                return _result(outer.generate(messages))

            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                return _result(await outer.agenerate(messages))

            @property
            def _llm_type(self) -> str:
                return "mock_chat_model"

            def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
                return self

        return _LangChainMockChat()
