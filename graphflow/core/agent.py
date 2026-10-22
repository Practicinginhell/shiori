"""Graphflow Agent abstraction.

An Agent is a definition: model + instructions + tools. In a workflow it
compiles into its own LangGraph subgraph (see ``graphflow.graph.agent_graph``),
so every model call and tool round is checkpointed separately. ``execute`` /
``aexecute`` run the same subgraph standalone.

State contract (all keys optional):
  reads   ``messages`` (conversation history), ``input`` / ``query`` (current turn)
  writes  ``response`` and ``output`` (final text), ``structured_output`` (parsed
          object, when requested), and appends this turn's assistant/tool
          messages to ``messages`` if the state has that field.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from graphflow.approvals import DECISION_TYPES
from graphflow.errors import ConfigurationError
from graphflow.ir.agent import AgentIR, ToolIR
from graphflow.ir.lower import type_repr
from graphflow.ir.registry import Registry
from graphflow.messages import normalize_message
from graphflow.models.base import BaseModelAdapter, ToolCall
from graphflow.models.registry import resolve_model
from graphflow.tools.base import Tool, tool

__all__ = ["Agent", "normalize_message"]

ApprovalHandler = Callable[[ToolCall], Any]


def _in_graph_context() -> bool:
    try:
        from langgraph.config import get_config

        get_config()
    except (ImportError, RuntimeError):
        return False
    return True


def _normalize_interrupt_on(
    agent: str, interrupt_on: dict[str, bool | dict[str, Any] | None]
) -> dict[str, dict[str, Any] | None]:
    """``{tool: True | False | {"allowed_decisions": [...], "description", "check", "describe"}}``, normalized."""
    out: dict[str, dict[str, Any] | None] = {}
    for tool_name, policy in interrupt_on.items():
        if policy is None or policy is False:
            out[tool_name] = None
            continue
        if policy is True:
            policy = {}
        if not isinstance(policy, dict):
            raise ConfigurationError(f"Agent '{agent}': interrupt_on['{tool_name}'] must be a bool or a dict")
        unknown_keys = set(policy) - {"allowed_decisions", "description", "check", "describe"}
        if unknown_keys:
            raise ConfigurationError(f"Agent '{agent}': unknown interrupt_on keys {sorted(unknown_keys)}")
        for hook in ("check", "describe"):
            if hook in policy and not callable(policy[hook]):
                raise ConfigurationError(
                    f"Agent '{agent}': interrupt_on['{tool_name}']['{hook}'] must be a function of the ToolRequest"
                )
        decisions = list(policy.get("allowed_decisions") or DECISION_TYPES)
        bad = [d for d in decisions if d not in DECISION_TYPES]
        if bad:
            raise ConfigurationError(
                f"Agent '{agent}': unknown decision(s) {bad} for '{tool_name}' (expected {list(DECISION_TYPES)})"
            )
        out[tool_name] = {
            "allowed_decisions": decisions,
            **{k: policy[k] for k in ("description", "check", "describe") if k in policy},
        }
    return out


def _lower_policy(policy: dict[str, Any] | None, registry: Registry) -> dict[str, Any] | None:
    """An approval policy as plain data: its check / describe functions become registry refs."""
    if policy is None:
        return None
    out = dict(policy)
    for hook in ("check", "describe"):
        if hook in out:
            out[hook] = registry.register(out[hook], "fn")
    return out


class Agent:
    """High-level autonomous agent.

    Args:
        name: Unique agent (and default node) name.
        model: A model adapter, a LangChain chat model, or a spec string such as
            ``"gpt-4o-mini"``, ``"openai:gpt-4o"``, ``"anthropic:claude-..."`` or ``"mock"``.
        instructions: System prompt.
        tools: Tools, plain functions, or names of tools registered on the Application.
        structured_output: Pydantic model / dataclass / dict type to parse the final answer into.
        memory: Include the state's ``messages`` history in the prompt.
        max_tool_iterations: Maximum tool-calling rounds before forcing a final answer.
        structured_output_retries: How many times to ask the model again, telling it
            the validation error, when its answer doesn't parse as `structured_output`.
        middleware: ``AgentMiddleware`` objects (or any objects with some of its hooks:
            ``before_model``, ``after_model``, ``wrap_model_call``, ``wrap_tool_call``,
            ``before_agent``/``after_agent``, ...); see ``graphflow.middleware``.
        interrupt_on: Per-tool approval policy, overriding the tools' ``requires_approval``:
            ``{"send_email": True, "delete": {"allowed_decisions": ["approve", "reject"]},
            "search": False}``. Decisions are "approve", "edit" (new arguments) and
            "reject" (with feedback for the model); all three are allowed by default.
            A policy can also have ``check(request) -> reason | None``, run before the
            reviewer is asked and again on their answer (a call that fails it is
            refused and the model told why), and ``describe(request) -> str``, the
            summary the reviewer sees (``action["summary"]``). Both receive a
            ``ToolRequest`` (arguments, run context, state) and may be async.
        approval_handler: Called with a ToolCall for each tool call needing approval;
            returns a ToolDecision, ApprovalDecision, bool or dict. Without one, the
            agent pauses the workflow for human approval (requires a checkpointer)
            and resumes exactly where it stopped.
    """

    def __init__(
        self,
        name: str,
        model: BaseModelAdapter | str | Any,
        instructions: str = "You are a helpful AI assistant.",
        tools: Sequence[Tool | Callable[..., Any] | str] | None = None,
        structured_output: type | None = None,
        memory: bool = True,
        max_tool_iterations: int = 10,
        middleware: list[Any] | None = None,
        approval_handler: ApprovalHandler | None = None,
        interrupt_on: dict[str, bool | dict[str, Any]] | None = None,
        structured_output_retries: int = 1,
    ):
        if max_tool_iterations < 0:
            raise ConfigurationError("max_tool_iterations must be >= 0")
        if structured_output_retries < 0:
            raise ConfigurationError("structured_output_retries must be >= 0")
        self.interrupt_on = _normalize_interrupt_on(name, interrupt_on or {})
        self.structured_output_retries = structured_output_retries
        self.name = name
        self.instructions = instructions
        self.structured_output = structured_output
        self.memory = memory
        self.max_tool_iterations = max_tool_iterations
        self.middleware = list(middleware or [])
        self.approval_handler = approval_handler
        self.model = resolve_model(model, owner=name)

        self._tool_refs: list[Tool | str] = []
        for t in tools or []:
            if isinstance(t, (Tool, str)):
                self._tool_refs.append(t)
            elif callable(t):
                self._tool_refs.append(tool(t))
            else:
                raise ConfigurationError(f"Agent '{name}': unsupported tool {t!r}")
        self._executors: dict[bool, Any] = {}

    @property
    def tools(self) -> list[Tool]:
        """The tools given as Tool objects or functions (not names registered on an Application)."""
        return [t for t in self._tool_refs if isinstance(t, Tool)]

    @property
    def tool_names(self) -> list[str]:
        return [t.name if isinstance(t, Tool) else t for t in self._tool_refs]

    # ------------------------------------------------------------------ #
    # Lowering
    # ------------------------------------------------------------------ #

    def lower(self, registry: Registry) -> AgentIR:
        """Describes this agent as plain data, registering its live objects in `registry`."""
        tools: list[ToolIR] = []
        for t in self._tool_refs:
            if isinstance(t, str):
                tools.append(ToolIR(name=t, ref=f"apptool:{t}"))
            else:
                tools.append(t.to_ir(registry))
        so = self.structured_output
        return AgentIR(
            name=self.name,
            instructions=self.instructions,
            model=registry.register(self.model, "model", self.name),
            model_repr=type(self.model).__name__,
            tools=tools,
            structured_output=registry.register(so, "type", type_repr(so)) if so is not None else None,
            structured_output_repr=type_repr(so) if so is not None else "",
            memory_enabled=self.memory,
            max_tool_iterations=self.max_tool_iterations,
            middleware=[registry.register(mw, "middleware") for mw in self.middleware],
            approval_handler=registry.register(self.approval_handler, "handler") if self.approval_handler else None,
            interrupt_on={k: _lower_policy(v, registry) for k, v in self.interrupt_on.items()},
            structured_output_retries=self.structured_output_retries,
        )

    def to_ir(self) -> AgentIR:
        """Plain-data description of this agent (for inspection)."""
        return self.lower(Registry())

    # ------------------------------------------------------------------ #
    # Standalone execution
    # ------------------------------------------------------------------ #

    def _executor(self) -> Any:
        # Inside a running graph the agent nests into it (inheriting its checkpointer),
        # so pausing for approval is possible; standalone it is not.
        in_graph = _in_graph_context()
        if in_graph not in self._executors:
            from graphflow.graph.agent_graph import AgentExecutor, build_agent_graph

            unresolved = [t for t in self._tool_refs if isinstance(t, str)]
            if unresolved:
                raise ConfigurationError(
                    f"Agent '{self.name}' references tool(s) {', '.join(unresolved)} by name; "
                    "register them with Application.add_tool() and run through the Application."
                )
            registry = Registry()
            ir = self.lower(registry)
            self._executors[in_graph] = AgentExecutor(build_agent_graph(ir, registry, in_graph), ir)
        return self._executors[in_graph]

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute agent synchronously on state."""
        return self.execute(state)

    def execute(self, state: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        """Runs the agent on `state` (with an optional run context) and returns its state update."""
        return self._executor().execute(dict(state), context=context)

    async def aexecute(self, state: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        """Asynchronously runs the agent on `state` and returns its state update."""
        return await self._executor().aexecute(dict(state), context=context)

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def clone(self, name: str | None = None, **overrides: Any) -> Agent:
        """Returns an independent copy (with its own model instance), optionally renamed/overridden."""
        params: dict[str, Any] = {
            "name": name or self.name,
            "model": self.model.clone(),
            "instructions": self.instructions,
            "tools": list(self._tool_refs),
            "structured_output": self.structured_output,
            "memory": self.memory,
            "max_tool_iterations": self.max_tool_iterations,
            "middleware": list(self.middleware),
            "approval_handler": self.approval_handler,
            "interrupt_on": dict(self.interrupt_on),
            "structured_output_retries": self.structured_output_retries,
        }
        params.update(overrides)
        return Agent(**params)

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={type(self.model).__name__}, tools={self.tool_names})"
