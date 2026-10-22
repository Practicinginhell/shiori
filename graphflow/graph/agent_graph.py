"""Compiles an AgentIR into a LangGraph subgraph.

    prepare -> model -+-> finalize -> END
                ^     +-> screen -+-> approve -> tools -+
                |     |           +-> tools ------------+
                |     +-> tools --------------------------+
                +-----------------------------------------+

Every step is its own checkpointed node, so when the agent pauses for tool
approval, resuming continues at ``approve``: the model is not called again and
tools that already ran do not run twice. Approval happens in its own node,
before any tool executes, for the same reason. ``screen`` first runs each
policy's ``check``: calls that fail it are refused without asking, and that
outcome is saved before the pause, so the reviewer answers exactly the calls
they were shown. All remaining calls of a round are put to the reviewer in one
request; an answer that doesn't fit (a disallowed decision, invalid edited
arguments) asks again, and approved calls are checked again before they run.

Model and tool calls go through the agent's middleware (see
``graphflow.middleware``).

The subgraph keeps the agent's working messages in its own state; the parent
workflow only receives the final update (``response``/``output``,
``structured_output``, and this turn's messages when it has a ``messages`` field).
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from graphflow._async import is_control_flow, run_coroutine_sync
from graphflow.approvals import (
    DECISION_TYPES,
    DecisionError,
    rejection_message,
    to_resume_value,
    tool_decisions,
)
from graphflow.context import get_context, resolve
from graphflow.errors import ConfigurationError, ExecutionError, GraphflowError, ModelError, ToolExecutionError
from graphflow.events import ToolCalled, ToolCompleted, emit_event
from graphflow.ir.agent import AgentIR
from graphflow.ir.registry import Registry
from graphflow.messages import normalize_message, strip_code_fences
from graphflow.middleware import MiddlewareStack, ModelRequest, ToolRequest
from graphflow.models.base import ModelResponse, ToolCall
from graphflow.reducers import append_reducer, merge_dict_reducer
from graphflow.retry import is_transient_error, status_code

logger = logging.getLogger("graphflow.agent")


class AgentState(TypedDict, total=False):
    parent_state: dict[str, Any]   # the workflow state the agent was called with (after before_agent hooks)
    prompt: list[Any]              # system + history + current input
    turn: Annotated[list[Any], append_reducer]  # messages produced during this call
    iterations: int                # tool-calling rounds so far
    decisions: dict[str, dict[str, Any]]  # tool_call_id -> decision (with its "source"), for the pending round
    edited: Annotated[dict[str, dict[str, Any]], merge_dict_reducer]  # "<round>:<tool_call_id>" -> args a reviewer set
    final: dict[str, Any] | None   # {"content": str} once the model answers
    update: dict[str, Any]         # what the agent returns to the workflow


def build_prompt(state: dict[str, Any], instructions: str, memory: bool) -> list[dict[str, Any]]:
    """System prompt + conversation history (if `memory`) + the current input/query."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    history_raw = state.get("messages")
    history = [normalize_message(m) for m in history_raw] if isinstance(history_raw, list) else []

    current = state.get("input") or state.get("query")
    if memory:
        messages.extend(history)
    elif history and not current:
        last_user = next((m for m in reversed(history) if m["role"] == "user"), None)
        if last_user:
            messages.append(last_user)

    if current:
        last = messages[-1]
        if not (last["role"] == "user" and last.get("content") == str(current)):
            messages.append({"role": "user", "content": str(current)})
    return messages


def _edit_key(tool_round: int, tool_call_id: str) -> str:
    # Keyed by round too: some models reuse tool-call ids ("call_0") in every message.
    return f"{tool_round}:{tool_call_id}"


def _with_edits(turn: list[Any], edited: dict[str, dict[str, Any]]) -> list[Any]:
    """This turn's messages, with the tool-call arguments reviewers replaced."""
    if not edited:
        return list(turn)
    out = []
    tool_round = 0
    for m in turn:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tool_round += 1
            m = {**m, "tool_calls": [
                {**tc, "args": edited.get(_edit_key(tool_round, tc.get("id", "")), tc.get("args"))}
                for tc in m["tool_calls"]
            ]}
        out.append(m)
    return out


def _pending_calls(state: AgentState) -> list[ToolCall]:
    turn = _with_edits(state.get("turn") or [], state.get("edited") or {})
    if not turn or turn[-1].get("role") != "assistant":
        return []
    return [ToolCall(id=tc.get("id", ""), name=tc["name"], args=tc.get("args") or {}) for tc in turn[-1].get("tool_calls") or []]


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _sync(value: Any) -> Any:
    return run_coroutine_sync(value) if inspect.isawaitable(value) else value


def _sync_chain(wraps: list[tuple[Any, Any]], base: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Composes wrap hooks around `base` for a sync run (first hook outermost)."""
    handler = base
    for sync_hook, async_hook in reversed(wraps):
        if sync_hook is not None:
            handler = (lambda h, nxt: lambda req: _sync(h(req, nxt)))(sync_hook, handler)
        else:
            async def as_async(req: Any, nxt: Callable[[Any], Any] = handler) -> Any:
                return nxt(req)

            handler = (lambda h, nxt: lambda req: run_coroutine_sync(h(req, nxt)))(async_hook, as_async)
    return handler


def _async_chain(
    wraps: list[tuple[Any, Any]], base: Callable[[Any], Any], sync_base: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Composes wrap hooks for an async run; sync-only hooks run the chain in a worker thread."""
    if any(async_hook is None for _, async_hook in wraps):
        chain = _sync_chain(wraps, sync_base)
        return lambda req: asyncio.to_thread(chain, req)
    handler = base
    for _, async_hook in reversed(wraps):
        handler = (lambda h, nxt: lambda req: h(req, nxt))(async_hook, handler)
    return handler


def _split_async_hooks(wraps: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """An ``async def wrap_*`` hook is the async hook, not a sync one."""
    out = []
    for sync_hook, async_hook in wraps:
        if sync_hook is not None and inspect.iscoroutinefunction(sync_hook):
            sync_hook, async_hook = None, async_hook or sync_hook
        out.append((sync_hook, async_hook))
    return out


class AgentRuntime:
    """An AgentIR with its references resolved, providing the subgraph's node functions."""

    def __init__(self, ir: AgentIR, registry: Registry, interrupts_enabled: bool):
        self.ir = ir
        self.model = registry.resolve(ir.model)
        self.tools = {}
        for t in ir.tools:
            if t.ref not in registry:
                raise ConfigurationError(f"Agent '{ir.name}' references unknown tool '{t.name}'")
            self.tools[t.name] = registry.resolve(t.ref)
        self.structured_output = registry.resolve(ir.structured_output) if ir.structured_output else None
        self.middleware = [registry.resolve(r) for r in ir.middleware]
        stack = MiddlewareStack.of(self.middleware)
        stack.wrap_model = _split_async_hooks(stack.wrap_model)
        stack.wrap_tool = _split_async_hooks(stack.wrap_tool)
        self.stack = stack
        self.approval_handler = registry.resolve(ir.approval_handler) if ir.approval_handler else None
        self.interrupts_enabled = interrupts_enabled
        self._registry = registry
        self.policies = self._approval_policies()

        self._model_sync = _sync_chain(stack.wrap_model, self._generate)
        self._model_async = _async_chain(stack.wrap_model, self._agenerate, self._generate)
        self._tool_sync = _sync_chain(stack.wrap_tool, self._execute_tool)
        self._tool_async = _async_chain(stack.wrap_tool, self._aexecute_tool, self._execute_tool)

    def _approval_policies(self) -> dict[str, dict[str, Any]]:
        """tool name -> {"allowed_decisions", "description", "check", "describe"} for tools that need approval."""
        unknown = sorted(set(self.ir.interrupt_on) - set(self.tools))
        if unknown:
            raise ConfigurationError(f"Agent '{self.ir.name}': interrupt_on names unknown tool(s) {unknown}")
        policies: dict[str, dict[str, Any]] = {}
        for name, t in self.tools.items():
            if name in self.ir.interrupt_on:
                policy = self.ir.interrupt_on[name]
                if policy is None:
                    continue
            elif getattr(t, "requires_approval", False):
                policy = {}
            else:
                continue
            policies[name] = {
                "allowed_decisions": list(policy.get("allowed_decisions") or DECISION_TYPES),
                "description": policy.get("description") or getattr(t, "description", ""),
                "check": self._registry.resolve(policy["check"]) if policy.get("check") else None,
                "describe": self._registry.resolve(policy["describe"]) if policy.get("describe") else None,
            }
        return policies

    # ---------------------------------------------------------------- #
    # prepare
    # ---------------------------------------------------------------- #

    def _prepared(self, parent_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "parent_state": parent_state,
            "prompt": build_prompt(parent_state, self.ir.instructions, self.ir.memory_enabled),
            "iterations": 0,
            "decisions": {},
            "final": None,
        }

    def prepare(self, s: AgentState) -> dict[str, Any]:
        state = dict(s.get("parent_state") or {})
        for h in self.stack.before_agent:
            out = _sync(h(state, self.ir.name))
            state = out if out is not None else state
        return self._prepared(state)

    async def aprepare(self, s: AgentState) -> dict[str, Any]:
        state = dict(s.get("parent_state") or {})
        for h in self.stack.before_agent:
            out = await _maybe_await(h(state, self.ir.name))
            state = out if out is not None else state
        return self._prepared(state)

    # ---------------------------------------------------------------- #
    # model
    # ---------------------------------------------------------------- #

    def _model_error(self, error: Exception) -> ModelError:
        return ModelError(
            f"Model call of agent '{self.ir.name}' failed: {type(error).__name__}: {error}",
            agent=self.ir.name, status_code=status_code(error), transient=is_transient_error(error),
        )

    def _generate(self, req: ModelRequest) -> ModelResponse:
        return self.model.generate(
            messages=req.messages, tools=req.tools or None, structured_output=req.structured_output
        )

    async def _agenerate(self, req: ModelRequest) -> ModelResponse:
        return await self.model.agenerate(
            messages=req.messages, tools=req.tools or None, structured_output=req.structured_output
        )

    # wrap_model_call middleware sees the provider's own exceptions (to retry or fall
    # back); what leaves the chain becomes a ModelError for the rest of the application.
    def _call_model_chain(self, req: ModelRequest) -> ModelResponse:
        try:
            return self._model_sync(req)
        except Exception as e:
            if is_control_flow(e) or isinstance(e, GraphflowError):
                raise
            raise self._model_error(e) from e

    async def _acall_model_chain(self, req: ModelRequest) -> ModelResponse:
        try:
            return await self._model_async(req)
        except Exception as e:
            if is_control_flow(e) or isinstance(e, GraphflowError):
                raise
            raise self._model_error(e) from e

    def _invoke_model(self, req: ModelRequest) -> ModelResponse:
        for h in self.stack.before_model:
            req = _sync(h(req)) or req
        resp = self._call_model_chain(req)
        for h in self.stack.after_model:
            resp = _sync(h(req, resp)) or resp
        return resp

    async def _ainvoke_model(self, req: ModelRequest) -> ModelResponse:
        for h in self.stack.before_model:
            req = await _maybe_await(h(req)) or req
        resp = await self._acall_model_chain(req)
        for h in self.stack.after_model:
            resp = await _maybe_await(h(req, resp)) or resp
        return resp

    def _messages(self, s: AgentState) -> list[Any]:
        return list(s.get("prompt") or []) + _with_edits(s.get("turn") or [], s.get("edited") or {})

    def _model_request(self, s: AgentState) -> tuple[ModelRequest, bool]:
        iterations = s.get("iterations") or 0
        forced = iterations >= self.ir.max_tool_iterations
        if forced and self.ir.max_tool_iterations > 0:
            logger.warning(
                "Agent '%s' reached max_tool_iterations=%d; requesting a final answer without tools.",
                self.ir.name,
                self.ir.max_tool_iterations,
            )
        use_tools = bool(self.tools) and not forced
        req = ModelRequest(
            messages=self._messages(s),
            tools=list(self.tools.values()) if use_tools else [],
            structured_output=None if use_tools else self.structured_output,
            agent=self.ir.name,
            state=dict(s.get("parent_state") or {}),
            iteration=iterations,
            context=get_context(),
        )
        return req, forced

    def _after_model(self, resp: ModelResponse, forced: bool, s: AgentState) -> dict[str, Any]:
        if resp.has_tool_calls and not forced:
            return {
                "turn": [{
                    "role": "assistant",
                    "content": resp.content,
                    "tool_calls": [tc.to_dict() for tc in resp.tool_calls],
                }],
                "iterations": (s.get("iterations") or 0) + 1,
                "decisions": {},
            }
        content = resp.content
        if not content and resp.parsed is not None:
            content = _serialize(resp.parsed)
        return {"final": {"content": content}}

    def call_model(self, s: AgentState) -> dict[str, Any]:
        req, forced = self._model_request(s)
        return self._after_model(self._invoke_model(req), forced, s)

    async def acall_model(self, s: AgentState) -> dict[str, Any]:
        req, forced = self._model_request(s)
        return self._after_model(await self._ainvoke_model(req), forced, s)

    def route_after_model(self, s: AgentState) -> str:
        if s.get("final") is not None:
            return "finalize"
        return "screen" if self._needing_approval(s) else "tools"

    # ---------------------------------------------------------------- #
    # approve
    # ---------------------------------------------------------------- #

    def _needing_approval(self, s: AgentState) -> list[ToolCall]:
        return [tc for tc in _pending_calls(s) if tc.name in self.policies]

    def _to_ask(self, s: AgentState) -> list[ToolCall]:
        """The calls needing approval that `screen` didn't refuse."""
        refused = s.get("decisions") or {}
        return [tc for tc in self._needing_approval(s) if tc.id not in refused]

    # Each hook returns an awaitable or a value; the sync path runs awaitables to completion.
    def _check_calls(self, calls: list[ToolCall], s: AgentState) -> list[tuple[ToolCall, Any]]:
        return [(tc, check(self._tool_request(tc, s))) for tc in calls
                if (check := self.policies[tc.name]["check"]) is not None]

    def _describe_calls(self, calls: list[ToolCall], s: AgentState) -> dict[str, Any]:
        return {tc.id: describe(self._tool_request(tc, s)) for tc in calls
                if (describe := self.policies[tc.name]["describe"]) is not None}

    @staticmethod
    def _refusals(results: list[tuple[ToolCall, Any]]) -> dict[str, dict[str, Any]]:
        return {tc.id: {"type": "reject", "feedback": str(reason), "source": "check"}
                for tc, reason in results if reason}

    def screen(self, s: AgentState) -> dict[str, Any]:
        checked = [(tc, _sync(r)) for tc, r in self._check_calls(self._needing_approval(s), s)]
        return {"decisions": self._refusals(checked)}

    async def ascreen(self, s: AgentState) -> dict[str, Any]:
        checked = [(tc, await _maybe_await(r)) for tc, r in self._check_calls(self._needing_approval(s), s)]
        return {"decisions": self._refusals(checked)}

    def route_after_screen(self, s: AgentState) -> str:
        return "approve" if self._to_ask(s) else "tools"

    def _action_request(self, tc: ToolCall, summary: Any = None) -> dict[str, Any]:
        policy = self.policies[tc.name]
        action = {
            "tool_call_id": tc.id,
            "tool": tc.name,
            "args": tc.args,
            "description": policy["description"],
            "allowed_decisions": list(policy["allowed_decisions"]),
        }
        if summary is not None:
            action["summary"] = str(summary)
        return action

    def _approval_payload(self, calls: list[ToolCall], summaries: dict[str, Any]) -> dict[str, Any]:
        actions = [self._action_request(tc, summaries.get(tc.id)) for tc in calls]
        if len(calls) == 1:
            tc = calls[0]
            payload = {
                "prompt": f"Approve call to tool '{tc.name}'?",
                "description": actions[0]["description"],
                "tool": tc.name,
                "args": tc.args,
                "tool_call_id": tc.id,
            }
        else:
            payload = {
                "prompt": f"Approve {len(calls)} tool calls ({', '.join(tc.name for tc in calls)})?",
                "description": "",
            }
        return {**payload, "agent": self.ir.name, "action_requests": actions}

    def _validate_decisions(
        self, decisions: dict[str, dict[str, Any]], calls: list[ToolCall]
    ) -> dict[str, dict[str, Any]]:
        """Enforces each tool's allowed decisions and validates edited arguments."""
        for tc in calls:
            d = decisions[tc.id]
            allowed = self.policies[tc.name]["allowed_decisions"]
            if d["type"] not in allowed:
                raise DecisionError(
                    f"'{d['type']}' is not allowed for tool '{tc.name}' (allowed: {', '.join(allowed)})"
                )
            if d["type"] == "edit":
                try:
                    self.tools[tc.name]._validate(d["args"])
                except ToolExecutionError as e:
                    raise DecisionError(f"edited arguments for '{tc.name}' are invalid: {e}") from None
        return decisions

    def _ask(self, calls: list[ToolCall], summaries: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Pauses for the reviewer until the answer fits the pending calls (LangGraph validation loop)."""
        if not self.interrupts_enabled:
            raise ToolExecutionError(
                f"Tool(s) {', '.join(tc.name for tc in calls)} require approval, but agent '{self.ir.name}' "
                "has no approval_handler and is not running in a workflow with a checkpointer "
                "(needed to pause for human approval)."
            )
        payload = self._approval_payload(calls, summaries)
        while True:
            answer = interrupt(payload)
            try:
                decisions = self._validate_decisions(tool_decisions(answer, [tc.id for tc in calls]), calls)
            except DecisionError as e:
                payload = {**payload, "error": str(e)}
                continue
            return {i: {**d, "source": "reviewer"} for i, d in decisions.items()}

    def _from_handler(self, tc: ToolCall, decision: Any) -> dict[str, dict[str, Any]]:
        try:
            decisions = self._validate_decisions(tool_decisions(to_resume_value(decision), [tc.id]), [tc])
        except DecisionError as e:
            raise ConfigurationError(f"Agent '{self.ir.name}': approval_handler returned {decision!r}: {e}") from None
        return {i: {**d, "source": "handler"} for i, d in decisions.items()}

    @staticmethod
    def _as_decided(tc: ToolCall, decision: dict[str, Any]) -> ToolCall:
        """The call as it would run after `decision` (with edited arguments)."""
        return ToolCall(id=tc.id, name=tc.name, args=decision["args"]) if decision["type"] == "edit" else tc

    def _accepted(self, calls: list[ToolCall], decisions: dict[str, dict[str, Any]]) -> list[ToolCall]:
        return [self._as_decided(tc, decisions[tc.id]) for tc in calls if decisions[tc.id]["type"] != "reject"]

    def _decided(self, s: AgentState, decisions: dict[str, dict[str, Any]], rechecked: list[tuple[ToolCall, Any]]) -> dict[str, Any]:
        tool_round = s.get("iterations") or 0  # the model node counted the pending round
        edited = {_edit_key(tool_round, i): d["args"] for i, d in decisions.items() if d["type"] == "edit"}
        # Approved calls are checked again: the world may have changed while the reviewer decided.
        return {"decisions": {**(s.get("decisions") or {}), **decisions, **self._refusals(rechecked)}, "edited": edited}

    def approve(self, s: AgentState) -> dict[str, Any]:
        calls = self._to_ask(s)
        if self.approval_handler is None:
            summaries = {i: _sync(v) for i, v in self._describe_calls(calls, s).items()}
            decisions = self._ask(calls, summaries)
        else:
            decisions = {}
            for tc in calls:
                decisions.update(self._from_handler(tc, _sync(self.approval_handler(tc))))
        rechecked = [(tc, _sync(r)) for tc, r in self._check_calls(self._accepted(calls, decisions), s)]
        return self._decided(s, decisions, rechecked)

    async def aapprove(self, s: AgentState) -> dict[str, Any]:
        calls = self._to_ask(s)
        if self.approval_handler is None:
            summaries = {i: await _maybe_await(v) for i, v in self._describe_calls(calls, s).items()}
            decisions = self._ask(calls, summaries)
        else:
            decisions = {}
            for tc in calls:
                decisions.update(self._from_handler(tc, await _maybe_await(self.approval_handler(tc))))
        rechecked = [(tc, await _maybe_await(r)) for tc, r in self._check_calls(self._accepted(calls, decisions), s)]
        return self._decided(s, decisions, rechecked)

    # ---------------------------------------------------------------- #
    # tools
    # ---------------------------------------------------------------- #

    @staticmethod
    def _tool_args(req: ToolRequest) -> dict[str, Any]:
        """The model's arguments, minus any it gave for injected ones, plus the injected values."""
        injections = getattr(req.tool, "injected", None) or {}
        args = {k: v for k, v in req.tool_call.args.items() if k not in injections}
        dropped = sorted(set(req.tool_call.args) & set(injections))
        if dropped:
            logger.warning("Ignored model-supplied value(s) for injected argument(s) %s of tool '%s'",
                           dropped, req.tool_call.name)
        return {**args, **resolve(injections, req.tool.func, req.context, req.state)}

    def _execute_tool(self, req: ToolRequest) -> str:
        return str(req.tool.execute(**self._tool_args(req)))

    async def _aexecute_tool(self, req: ToolRequest) -> str:
        return str(await req.tool.aexecute(**self._tool_args(req)))

    def _tool_message(self, tc: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content}

    def _precheck(self, tc: ToolCall, decisions: dict[str, dict[str, Any]]) -> str | None:
        """The result to report without running the tool (unknown tool, rejected call), if any."""
        emit_event(ToolCalled(tool_name=tc.name, tool_args=dict(tc.args), tool_call_id=tc.id))
        if tc.name not in self.tools:
            return f"Error: tool '{tc.name}' not found."
        decision = decisions.get(tc.id)
        if decision is not None and decision["type"] == "reject":
            return rejection_message(decision.get("feedback", ""), decision.get("source", "reviewer"))
        return None

    def _completed(self, tc: ToolCall, started: float, result: str, error: str | None) -> dict[str, Any]:
        emit_event(ToolCompleted(
            tool_name=tc.name,
            result=result,
            tool_call_id=tc.id,
            duration_seconds=time.perf_counter() - started,
            error=error,
        ))
        return self._tool_message(tc, result)

    def _tool_request(self, tc: ToolCall, s: AgentState) -> ToolRequest:
        return ToolRequest(
            tool_call=tc, tool=self.tools[tc.name], agent=self.ir.name,
            state=dict(s.get("parent_state") or {}), context=get_context(),
        )

    def _run_tool(self, tc: ToolCall, s: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        early = self._precheck(tc, s.get("decisions") or {})
        if early is not None:
            return self._completed(tc, started, early, early)
        try:
            result = self._tool_sync(self._tool_request(tc, s))
        except ToolExecutionError as e:
            return self._completed(tc, started, f"Error: {e}", str(e))
        return self._completed(tc, started, str(result), None)

    async def _arun_tool(self, tc: ToolCall, s: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        early = self._precheck(tc, s.get("decisions") or {})
        if early is not None:
            return self._completed(tc, started, early, early)
        try:
            result = await self._tool_async(self._tool_request(tc, s))
        except ToolExecutionError as e:
            return self._completed(tc, started, f"Error: {e}", str(e))
        return self._completed(tc, started, str(result), None)

    def run_tools(self, s: AgentState) -> dict[str, Any]:
        return {"turn": [self._run_tool(tc, s) for tc in _pending_calls(s)]}

    async def arun_tools(self, s: AgentState) -> dict[str, Any]:
        results = await asyncio.gather(*(self._arun_tool(tc, s) for tc in _pending_calls(s)))
        return {"turn": list(results)}

    # ---------------------------------------------------------------- #
    # finalize
    # ---------------------------------------------------------------- #

    def _parse(self, content: str) -> tuple[Any, str | None]:
        """(parsed, error): parses `content` into the structured output type."""
        schema = self.structured_output
        try:
            data = json.loads(strip_code_fences(content))
        except (json.JSONDecodeError, TypeError) as e:
            return None, f"the reply is not valid JSON ({e})"
        try:
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data), None
            if dataclasses.is_dataclass(schema) and isinstance(data, dict):
                return schema(**data), None
        except Exception as e:  # noqa: BLE001 - any validation failure is reported back to the model
            return None, str(e)
        return data, None

    def _schema_name(self) -> str:
        return getattr(self.structured_output, "__name__", "structured")

    def _retry_request(self, s: AgentState, messages: list[Any], content: str, error: str) -> tuple[ModelRequest, list[Any]]:
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": (
                f"Your reply could not be parsed as {self._schema_name()}: {error}\n"
                "Reply with only a JSON object that matches the required schema."
            )},
        ]
        req = ModelRequest(
            messages=messages,
            structured_output=self.structured_output,
            agent=self.ir.name,
            state=dict(s.get("parent_state") or {}),
            iteration=s.get("iterations") or 0,
            context=get_context(),
        )
        return req, messages

    def _build_update(self, s: AgentState, content: str, parsed: Any) -> dict[str, Any]:
        update: dict[str, Any] = {"response": content, "output": content}
        if self.structured_output is not None:
            if parsed is None:
                raise ExecutionError(f"Agent '{self.ir.name}' could not produce valid {self._schema_name()} output")
            update["structured_output"] = parsed
        parent_state = s.get("parent_state") or {}
        if "messages" in parent_state:
            turn = _with_edits(s.get("turn") or [], s.get("edited") or {})
            update["messages"] = turn + [{"role": "assistant", "content": content}]
        return update

    def finalize(self, s: AgentState) -> dict[str, Any]:
        content = (s.get("final") or {}).get("content", "")
        parsed = None
        if self.structured_output is not None:
            parsed, error = self._parse(content)
            messages = self._messages(s)
            for _ in range(self.ir.structured_output_retries):
                if error is None:
                    break
                req, messages = self._retry_request(s, messages, content, error)
                resp = self._invoke_model(req)
                content = resp.content or _serialize(resp.parsed)
                parsed, error = self._parse(content)
        update = self._build_update(s, content, parsed)
        parent_state = s.get("parent_state") or {}
        for h in self.stack.after_agent:
            out = _sync(h(parent_state, update, self.ir.name))
            update = out if out is not None else update
        return {"update": update}

    async def afinalize(self, s: AgentState) -> dict[str, Any]:
        content = (s.get("final") or {}).get("content", "")
        parsed = None
        if self.structured_output is not None:
            parsed, error = self._parse(content)
            messages = self._messages(s)
            for _ in range(self.ir.structured_output_retries):
                if error is None:
                    break
                req, messages = self._retry_request(s, messages, content, error)
                resp = await self._ainvoke_model(req)
                content = resp.content or _serialize(resp.parsed)
                parsed, error = self._parse(content)
        update = self._build_update(s, content, parsed)
        parent_state = s.get("parent_state") or {}
        for h in self.stack.after_agent:
            out = await _maybe_await(h(parent_state, update, self.ir.name))
            update = out if out is not None else update
        return {"update": update}


def _serialize(parsed: Any) -> str:
    if parsed is None:
        return ""
    if hasattr(parsed, "model_dump_json"):
        return parsed.model_dump_json()
    if dataclasses.is_dataclass(parsed) and not isinstance(parsed, type):
        return json.dumps(dataclasses.asdict(parsed), default=str)
    return json.dumps(parsed, default=str)


def build_agent_graph(ir: AgentIR, registry: Registry, interrupts_enabled: bool) -> Any:
    """Compiles the agent's subgraph (no checkpointer: when nested it uses the parent's)."""
    rt = AgentRuntime(ir, registry, interrupts_enabled)
    g = StateGraph(AgentState)
    g.add_node("prepare", RunnableLambda(rt.prepare, afunc=rt.aprepare, name="prepare"))
    g.add_node("model", RunnableLambda(rt.call_model, afunc=rt.acall_model, name="model"))
    g.add_node("screen", RunnableLambda(rt.screen, afunc=rt.ascreen, name="screen"))
    g.add_node("approve", RunnableLambda(rt.approve, afunc=rt.aapprove, name="approve"))
    g.add_node("tools", RunnableLambda(rt.run_tools, afunc=rt.arun_tools, name="tools"))
    g.add_node("finalize", RunnableLambda(rt.finalize, afunc=rt.afinalize, name="finalize"))
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "model")
    g.add_conditional_edges("model", rt.route_after_model, ["finalize", "screen", "tools"])
    g.add_conditional_edges("screen", rt.route_after_screen, ["approve", "tools"])
    g.add_edge("approve", "tools")
    g.add_edge("tools", "model")
    g.add_edge("finalize", END)
    return g.compile(name=ir.name)


class AgentExecutor:
    """Runs a compiled agent subgraph as one workflow node (or standalone)."""

    def __init__(self, graph: Any, ir: AgentIR):
        self.graph = graph
        self.ir = ir
        # prepare + finalize + (model, screen, approve, tools) per round + the forced final model call
        self._config = {"recursion_limit": 4 * ir.max_tool_iterations + 10}

    def _update(self, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("__interrupt__"):
            raise ToolExecutionError(
                f"Agent '{self.ir.name}' paused for approval, but there is no checkpointer to resume from."
            )
        return result["update"]

    @staticmethod
    def _kwargs(context: Any) -> dict[str, Any]:
        # Inside a workflow the run context is inherited; only standalone runs pass one.
        return {} if context is None else {"context": context}

    def execute(self, state: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        return self._update(self.graph.invoke({"parent_state": dict(state)}, self._config, **self._kwargs(context)))

    async def aexecute(self, state: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        return self._update(
            await self.graph.ainvoke({"parent_state": dict(state)}, self._config, **self._kwargs(context))
        )
