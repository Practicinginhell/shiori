"""Agent middleware: hooks around an agent's model and tool calls.

The hooks match LangChain's agent middleware:

=====================  ==============================================  ==================
hook                   called with                                     returns
=====================  ==============================================  ==================
``before_agent``       ``(state, agent_name)``                         new state or None
``after_agent``        ``(state, update, agent_name)``                 new update or None
``before_model``       ``(request)``                                   new request or None
``after_model``        ``(request, response)``                         new response or None
``wrap_model_call``    ``(request, handler)``; ``handler(request)``    ModelResponse
``wrap_tool_call``     ``(request, handler)``; ``handler(request)``    tool result (str)
=====================  ==============================================  ==================

``before_*`` hooks run in list order and ``after_*`` hooks in reverse; for the
wrap hooks the first middleware is the outermost. Every hook may be ``async``.
The wrap hooks receive a synchronous handler in sync runs; for async runs,
define ``awrap_model_call`` / ``awrap_tool_call`` (whose handler is awaitable).
An async run whose middleware only has the sync wrap hooks runs that call
chain in a worker thread. ``wrap_tool_call`` sees tool failures as
``ToolExecutionError``; one that escapes the chain is reported to the model as
an ``Error: ...`` tool message, while any other exception fails the agent.

The legacy hook names ``pre_execute`` / ``post_execute`` are aliases of
``before_agent`` / ``after_agent``. Subclassing ``AgentMiddleware`` is
optional: any object with some of these methods works.

Hooks run inside the agent's checkpointed steps, so when an agent resumes
after an approval pause, hooks for steps that already finished do not run again.

Built-in middleware: ``HistoryLimit`` (bounds the prompt of long conversations).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from graphflow.errors import ConfigurationError
from graphflow.models.base import ModelResponse, ToolCall

if TYPE_CHECKING:
    from graphflow.tools.base import Tool

HOOKS = (
    "before_agent",
    "after_agent",
    "before_model",
    "after_model",
    "wrap_model_call",
    "awrap_model_call",
    "wrap_tool_call",
    "awrap_tool_call",
)
_ALIASES = {"before_agent": "pre_execute", "after_agent": "post_execute"}


@dataclass(frozen=True)
class ModelRequest:
    """One model call an agent is about to make."""
    messages: list[dict[str, Any]]
    tools: list[Tool] = field(default_factory=list)
    structured_output: type | None = None
    agent: str = ""
    state: dict[str, Any] = field(default_factory=dict)  # the state the agent was called with
    iteration: int = 0  # tool-calling rounds completed before this call
    context: Any = None  # the run context (see graphflow.context)

    def override(self, **changes: Any) -> ModelRequest:
        """A copy with `changes` applied."""
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class ToolRequest:
    """One tool call an agent is about to execute."""
    tool_call: ToolCall
    tool: Tool
    agent: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    context: Any = None  # the run context (see graphflow.context)

    def override(self, **changes: Any) -> ToolRequest:
        """A copy with `changes` applied."""
        return dataclasses.replace(self, **changes)


class AgentMiddleware:
    """Optional base class for agent middleware; override the hooks you need.

    Only overridden methods are called, so a subclass that defines
    ``wrap_model_call`` alone does not force async runs into a thread.
    """

    def before_agent(self, state: dict[str, Any], agent_name: str) -> dict[str, Any] | None:
        return None

    def after_agent(self, state: dict[str, Any], update: dict[str, Any], agent_name: str) -> dict[str, Any] | None:
        return None

    def before_model(self, request: ModelRequest) -> ModelRequest | None:
        return None

    def after_model(self, request: ModelRequest, response: ModelResponse) -> ModelResponse | None:
        return None

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> ModelResponse:
        return await handler(request)

    def wrap_tool_call(self, request: ToolRequest, handler: Callable[[ToolRequest], str]) -> str:
        return handler(request)

    async def awrap_tool_call(self, request: ToolRequest, handler: Callable[[ToolRequest], Any]) -> str:
        return await handler(request)


def estimate_tokens(message: dict[str, Any]) -> int:
    """A rough token count (about 4 characters per token), for when no tokenizer is given."""
    text = str(message.get("content") or "")
    if message.get("tool_calls"):
        text += json.dumps(message["tool_calls"], default=str)
    return len(text) // 4 + 4


class HistoryLimit(AgentMiddleware):
    """Sends the model only the most recent part of a long conversation.

    Keeps the leading system messages and the current turn (the latest user
    message and everything after it, such as this turn's tool calls and results),
    then as much of the earlier history as fits, dropping the oldest first. It
    cuts only where a user message starts, so a tool result never loses its call.
    Only the prompt is trimmed: the state's ``messages`` keep everything.

    Args:
        max_messages: Most messages to send besides the system messages.
        max_tokens: Most tokens to send in total, counted with `count_tokens`.
        count_tokens: ``(message) -> int``; defaults to ``estimate_tokens``. Pass
            your model's tokenizer for exact limits.

    The current turn is kept even when it alone exceeds a limit.
    """

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        count_tokens: Callable[[dict[str, Any]], int] | None = None,
    ):
        if max_messages is None and max_tokens is None:
            raise ConfigurationError("HistoryLimit needs max_messages or max_tokens")
        for name, value in (("max_messages", max_messages), ("max_tokens", max_tokens)):
            if value is not None and value < 1:
                raise ConfigurationError(f"HistoryLimit {name} must be at least 1, got {value}")
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens or estimate_tokens

    def _fits(self, kept: list[dict[str, Any]], system_tokens: int) -> bool:
        if self.max_messages is not None and len(kept) > self.max_messages:
            return False
        if self.max_tokens is not None and system_tokens + sum(map(self.count_tokens, kept)) > self.max_tokens:
            return False
        return True

    def before_model(self, request: ModelRequest) -> ModelRequest | None:
        messages = request.messages
        n_system = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(messages))
        system, rest = messages[:n_system], messages[n_system:]
        users = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        if not users:
            return None
        system_tokens = sum(map(self.count_tokens, system)) if self.max_tokens is not None else 0
        for start in sorted({0, *users}):  # the longest suffix that fits: all of it, or one starting at a user message
            if self._fits(rest[start:], system_tokens):
                break
        else:
            start = users[-1]  # the current turn, whatever its size
        return None if start == 0 else request.override(messages=system + rest[start:])


def hook(middleware: Any, name: str) -> Callable[..., Any] | None:
    """The bound hook `name` of `middleware`, or None when it does not define (override) one."""
    if isinstance(middleware, AgentMiddleware):
        if getattr(type(middleware), name) is getattr(AgentMiddleware, name):
            alias = _ALIASES.get(name)
            return getattr(middleware, alias, None) if alias else None
        return getattr(middleware, name)
    found = getattr(middleware, name, None)
    if found is None and name in _ALIASES:
        found = getattr(middleware, _ALIASES[name], None)
    return found if callable(found) else None


@dataclass
class MiddlewareStack:
    """The hooks of a list of middleware, grouped by kind."""
    before_agent: list[Callable[..., Any]]
    after_agent: list[Callable[..., Any]]      # already reversed
    before_model: list[Callable[..., Any]]
    after_model: list[Callable[..., Any]]      # already reversed
    wrap_model: list[tuple[Callable[..., Any] | None, Callable[..., Any] | None]]  # (sync, async), outermost first
    wrap_tool: list[tuple[Callable[..., Any] | None, Callable[..., Any] | None]]

    @classmethod
    def of(cls, middleware: list[Any]) -> MiddlewareStack:
        def collect(name: str) -> list[Callable[..., Any]]:
            return [h for h in (hook(mw, name) for mw in middleware) if h is not None]

        def wraps(sync_name: str, async_name: str) -> list[tuple[Any, Any]]:
            pairs = [(hook(mw, sync_name), hook(mw, async_name)) for mw in middleware]
            return [p for p in pairs if p != (None, None)]

        return cls(
            before_agent=collect("before_agent"),
            after_agent=list(reversed(collect("after_agent"))),
            before_model=collect("before_model"),
            after_model=list(reversed(collect("after_model"))),
            wrap_model=wraps("wrap_model_call", "awrap_model_call"),
            wrap_tool=wraps("wrap_tool_call", "awrap_tool_call"),
        )
