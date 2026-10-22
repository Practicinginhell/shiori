"""Adapter for any LangChain chat model (OpenAI, Anthropic, Ollama, vLLM, ...)."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from graphflow.models.base import BaseModelAdapter, ModelResponse, ToolCall

if TYPE_CHECKING:
    from graphflow.tools.base import Tool


def to_langchain_messages(messages: list[Any]) -> list[Any]:
    """Converts Graphflow chat dicts (or LangChain messages) into LangChain messages."""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

    out: list[Any] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if not isinstance(m, dict):
            out.append(HumanMessage(content=str(m)))
            continue
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            calls = [
                {"id": tc.get("id", ""), "name": tc["name"], "args": tc.get("args", {}), "type": "tool_call"}
                for tc in m.get("tool_calls") or []
            ]
            out.append(AIMessage(content=content, tool_calls=calls))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", ""), name=m.get("name")))
        else:
            out.append(HumanMessage(content=content))
    return out


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _serialize_parsed(parsed: Any) -> str:
    if hasattr(parsed, "model_dump_json"):
        return parsed.model_dump_json()
    try:
        return json.dumps(parsed, default=str)
    except TypeError:
        return str(parsed)


class LangChainModel(BaseModelAdapter):
    """Wraps a LangChain ``BaseChatModel`` instance."""

    def __init__(self, chat_model: Any = None):
        self._chat_model = chat_model

    def to_langchain_model(self) -> Any:
        return self._chat_model

    def _bind(self, tools: list[Tool] | None) -> Any:
        llm = self.to_langchain_model()
        if tools:
            llm = llm.bind_tools([t.to_langchain_tool() for t in tools])
        return llm

    def _structured(self, structured_output: type) -> Any:
        return self.to_langchain_model().with_structured_output(structured_output, include_raw=True)

    @staticmethod
    def _to_response(resp: Any) -> ModelResponse:
        tool_calls = [
            ToolCall(id=tc.get("id") or "", name=tc.get("name", ""), args=tc.get("args") or {})
            for tc in getattr(resp, "tool_calls", None) or []
        ]
        usage = dict(getattr(resp, "usage_metadata", None) or {})
        return ModelResponse(
            content=_text_content(getattr(resp, "content", resp)),
            tool_calls=tool_calls,
            raw_response=resp,
            usage=usage,
        )

    @staticmethod
    def _to_structured_response(result: dict[str, Any]) -> ModelResponse:
        if result.get("parsing_error") is not None:
            raise ValueError(f"Structured output parsing failed: {result['parsing_error']}")
        parsed = result.get("parsed")
        return ModelResponse(content=_serialize_parsed(parsed), raw_response=result.get("raw"), parsed=parsed)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        structured_output: type | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        lc_messages = to_langchain_messages(messages)
        if structured_output and not tools:
            return self._to_structured_response(self._structured(structured_output).invoke(lc_messages, **kwargs))
        return self._to_response(self._bind(tools).invoke(lc_messages, **kwargs))

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        structured_output: type | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        lc_messages = to_langchain_messages(messages)
        if structured_output and not tools:
            result = await self._structured(structured_output).ainvoke(lc_messages, **kwargs)
            return self._to_structured_response(result)
        return self._to_response(await self._bind(tools).ainvoke(lc_messages, **kwargs))

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for chunk in self._bind(tools).stream(to_langchain_messages(messages), **kwargs):
            text = _text_content(getattr(chunk, "content", ""))
            if text:
                yield text

    async def astream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for chunk in self._bind(tools).astream(to_langchain_messages(messages), **kwargs):
            text = _text_content(getattr(chunk, "content", ""))
            if text:
                yield text

    def clone(self) -> LangChainModel:
        return copy.copy(self)
