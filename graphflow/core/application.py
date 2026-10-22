"""Graphflow Application.

The primary entry point for developers. Separates Application Definition
from Compilation and Execution, providing a clean, high-level API with
full access to underlying LangGraph escape hatches.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from graphflow.approvals import ApprovalDecision, ToolDecision
from graphflow.core.agent import Agent
from graphflow.core.thread import MessageLog, Thread
from graphflow.core.workflow import Workflow
from graphflow.errors import ConfigurationError, StateValidationError
from graphflow.events import StreamEvent
from graphflow.graph.compiler import LangGraphCompiler
from graphflow.graph.state_builder import build_validator, effective_schema
from graphflow.ir.graph import GraphIR, LoweredWorkflow
from graphflow.ir.registry import Registry
from graphflow.persistence.base import BaseCheckpointStore, checkpointer_supports_async
from graphflow.runtime.checkpoints import Checkpoint, ThreadHistory
from graphflow.runtime.runner import RunResult, RuntimeRunner
from graphflow.tools.base import Tool, tool

if TYPE_CHECKING:
    from graphflow.config.loader import Config

logger = logging.getLogger("graphflow.application")

EventListener = Callable[[StreamEvent], Any]
Decision = ApprovalDecision | ToolDecision | bool | dict[str, Any] | list[Any]


def _with_thread(config: dict[str, Any] | None, thread_id: str | None) -> dict[str, Any]:
    cfg = dict(config or {})
    if thread_id:
        cfg["configurable"] = {**(cfg.get("configurable") or {}), "thread_id": thread_id}
    return cfg


def _has_thread(config: dict[str, Any]) -> bool:
    return bool((config.get("configurable") or {}).get("thread_id"))


class CompiledApplication:
    """An immutable, compiled Application ready for execution.

    ``run``/``stream`` use LangGraph's native sync API and ``arun``/``astream``
    its async API. Results are ``RunResult`` dicts; check ``result.interrupted``
    and ``result.interrupts`` after runs that may pause for approval.
    """

    def __init__(
        self,
        name: str,
        compiled_graph: Any,
        ir: GraphIR,
        registry: Registry | None = None,
        listeners: list[EventListener] | None = None,
        sync_only: bool = False,
    ):
        self.name = name
        self.compiled_graph = compiled_graph  # Escape Hatch: Direct access to Pregel runnable
        self.ir = ir  # plain-data description of the compiled workflow
        self.registry = registry or Registry()
        self._schema = effective_schema(ir.state_schema)
        self._validator = build_validator(ir.state_schema, self.registry)
        self._runner = RuntimeRunner(compiled_graph=compiled_graph, name=name, listeners=listeners, sync_only=sync_only)
        self._threads = ThreadHistory(compiled_graph, name, sync_only=sync_only)
        self._messages = MessageLog(getattr(compiled_graph, "store", None))
        self._thread_locks: dict[str, threading.Lock] = {}
        self._thread_locks_guard = threading.Lock()

    def thread(self, thread_id: str, context: Any = None) -> Thread:
        """One conversation: send input, answer pending decisions, browse and fork its history.

        See ``graphflow.core.thread.Thread``. `context` is the run context for every
        run this object makes (not saved with the thread).
        """
        return Thread(self, thread_id, context=context)

    def _thread_lock(self, thread_id: str) -> threading.Lock:
        with self._thread_locks_guard:
            return self._thread_locks.setdefault(thread_id, threading.Lock())

    def _config(self, config: dict[str, Any] | None, thread_id: str | None) -> dict[str, Any]:
        cfg = _with_thread(config, thread_id)
        if getattr(self.compiled_graph, "checkpointer", None) and not _has_thread(cfg):
            raise ConfigurationError(
                f"Application '{self.name}' has a checkpointer, so each run needs a thread_id "
                "(e.g. app.run(input, thread_id='user-123'))"
            )
        return cfg

    def _check_input(self, input_data: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(input_data or {})
        unknown = sorted(k for k in data if k not in self._schema.fields)
        if unknown:
            message = (
                f"Input key(s) {unknown} are not declared in state schema "
                f"'{self._schema.name}' and will be ignored"
            )
            if self._schema.strict:
                raise StateValidationError(message)
            logger.warning(message)
        if self._schema.validation != "off":
            data = self._validator.validate(data, "input")
        return self._record_question(data)

    def _record_question(self, data: dict[str, Any]) -> dict[str, Any]:
        """Saves this run's `input` / `query` to the conversation, so the next turn's history has it.

        Done once per run, at the input, rather than by each agent: agents only add their own turn.
        """
        field = self._schema.fields.get("messages")
        question = data.get("input") or data.get("query")
        if field is None or field.reducer != "append" or not question or "messages" in data:
            return data
        return {**data, "messages": [{"role": "user", "content": str(question)}]}

    def run(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        context: Any = None,
    ) -> RunResult:
        """Synchronously run the application to completion (or until it pauses for approval).

        `context` is the run context (who the user is, ...; see ``graphflow.context``).
        It is not saved with the thread, so pass it again on every resume.
        """
        return self._runner.run(self._check_input(input_data), self._config(config, thread_id), context=context)

    async def arun(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        context: Any = None,
    ) -> RunResult:
        """Asynchronously run the application."""
        return await self._runner.arun(self._check_input(input_data), self._config(config, thread_id), context=context)

    def stream(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        context: Any = None,
    ) -> Iterator[StreamEvent]:
        """Synchronously stream high-level execution events as they happen."""
        return self._runner.stream(self._check_input(input_data), self._config(config, thread_id), context=context)

    async def astream(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        context: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        """Asynchronously stream high-level execution events."""
        async for event in self._runner.astream(
            self._check_input(input_data), self._config(config, thread_id), context=context
        ):
            yield event

    def _check_waiting(self, thread_id: str, snapshot: Any) -> None:
        # Resuming a thread with no pending interrupt would silently "complete" (a typo'd id,
        # an approval answered twice); LangGraph doesn't complain, so we do.
        # After update_state on a paused thread the snapshot lists no interrupt, but the paused step is still next.
        if not getattr(snapshot, "interrupts", None) and not getattr(snapshot, "next", None):
            raise ConfigurationError(f"Thread '{thread_id}' has nothing waiting for a decision")

    def _waiting(self, thread_id: str, config: dict[str, Any]) -> dict[str, Any]:
        self._check_waiting(thread_id, self.compiled_graph.get_state(config))
        return config

    async def _awaiting(self, thread_id: str, config: dict[str, Any]) -> dict[str, Any]:
        if self._runner.sync_only:
            snapshot = await asyncio.to_thread(self.compiled_graph.get_state, config)
        else:
            snapshot = await self.compiled_graph.aget_state(config)
        self._check_waiting(thread_id, snapshot)
        return config

    def resume(
        self,
        thread_id: str,
        decision: Decision,
        config: dict[str, Any] | None = None,
        interrupt_id: str | None = None,
        *,
        context: Any = None,
    ) -> RunResult:
        """Resume a paused thread. Pass `interrupt_id` when several approvals are pending."""
        cfg = self._waiting(thread_id, self._config(config, thread_id))
        return self._runner.resume(decision, cfg, interrupt_id=interrupt_id, context=context)

    async def aresume(
        self,
        thread_id: str,
        decision: Decision,
        config: dict[str, Any] | None = None,
        interrupt_id: str | None = None,
        *,
        context: Any = None,
    ) -> RunResult:
        """Asynchronously resume a paused thread."""
        cfg = await self._awaiting(thread_id, self._config(config, thread_id))
        return await self._runner.aresume(decision, cfg, interrupt_id=interrupt_id, context=context)

    def resume_all(
        self,
        thread_id: str,
        decisions: Mapping[str, Decision],
        config: dict[str, Any] | None = None,
        *,
        context: Any = None,
    ) -> RunResult:
        """Answer several pending interrupts (e.g. parallel branches) at once: ``{interrupt_id: decision}``."""
        cfg = self._waiting(thread_id, self._config(config, thread_id))
        return self._runner.run(RuntimeRunner.resume_all_command(decisions), cfg, context=context)

    async def aresume_all(
        self,
        thread_id: str,
        decisions: Mapping[str, Decision],
        config: dict[str, Any] | None = None,
        *,
        context: Any = None,
    ) -> RunResult:
        """Asynchronously answer several pending interrupts at once."""
        cfg = await self._awaiting(thread_id, self._config(config, thread_id))
        return await self._runner.arun(RuntimeRunner.resume_all_command(decisions), cfg, context=context)

    def stream_resume(
        self,
        thread_id: str,
        decision: Decision,
        config: dict[str, Any] | None = None,
        interrupt_id: str | None = None,
        *,
        context: Any = None,
    ) -> Iterator[StreamEvent]:
        """Resume a paused thread, streaming events."""
        cfg = self._waiting(thread_id, self._config(config, thread_id))
        return self._runner.stream(RuntimeRunner.resume_command(decision, interrupt_id), cfg, context=context)

    def get_state(self, thread_id: str) -> Any:
        """Inspect the current LangGraph state snapshot for a thread (time-travel / state inspection)."""
        return self.compiled_graph.get_state({"configurable": {"thread_id": thread_id}})

    # ------------------------------------------------------------------ #
    # Time travel
    # ------------------------------------------------------------------ #

    def _check_update(self, values: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(k for k in values if k not in self._schema.fields)
        if unknown:
            raise StateValidationError(f"update_state: unknown field(s) {unknown} for schema '{self._schema.name}'")
        if self._schema.validation != "off":
            return self._validator.validate(dict(values), "update_state")
        return dict(values)

    def history(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        """The thread's checkpoints, newest first."""
        return self._threads.history(thread_id, limit)

    async def ahistory(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        """Asynchronously list the thread's checkpoints, newest first."""
        return await self._threads.ahistory(thread_id, limit)

    def update_state(
        self,
        thread_id: str,
        values: dict[str, Any],
        *,
        checkpoint_id: str | None = None,
        as_node: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """Writes `values` into the thread (at its latest or a given checkpoint); returns the new checkpoint id.

        Values pass through the fields' reducers (an ``append`` field appends) unless
        ``overwrite=True``. `as_node` makes the write count as that node's output,
        which decides what runs next.
        """
        values = self._check_update(values)
        cfg = (
            self._threads.checkpoint_config(thread_id, checkpoint_id)
            if checkpoint_id else self._threads.config(thread_id)
        )
        return self._threads.update(cfg, values, as_node, overwrite)

    async def aupdate_state(
        self,
        thread_id: str,
        values: dict[str, Any],
        *,
        checkpoint_id: str | None = None,
        as_node: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """Asynchronous ``update_state``."""
        values = self._check_update(values)
        cfg = (
            await self._threads.acheckpoint_config(thread_id, checkpoint_id)
            if checkpoint_id else self._threads.config(thread_id)
        )
        return await self._threads.aupdate(cfg, values, as_node, overwrite)

    def replay(self, thread_id: str, checkpoint_id: str, *, context: Any = None) -> RunResult:
        """Re-runs the thread from a past checkpoint (the steps after it run again)."""
        return self._runner.run(None, self._threads.checkpoint_config(thread_id, checkpoint_id), context=context)

    async def areplay(self, thread_id: str, checkpoint_id: str, *, context: Any = None) -> RunResult:
        """Asynchronous ``replay``."""
        cfg = await self._threads.acheckpoint_config(thread_id, checkpoint_id)
        return await self._runner.arun(None, cfg, context=context)

    def fork(
        self,
        thread_id: str,
        checkpoint_id: str,
        values: dict[str, Any],
        *,
        as_node: str | None = None,
        overwrite: bool = False,
        context: Any = None,
    ) -> RunResult:
        """Writes `values` at a past checkpoint and runs on from there, leaving the original history intact."""
        new_id = self.update_state(
            thread_id, values, checkpoint_id=checkpoint_id, as_node=as_node, overwrite=overwrite
        )
        return self._runner.run(None, self._threads.config(thread_id, new_id), context=context)

    async def afork(
        self,
        thread_id: str,
        checkpoint_id: str,
        values: dict[str, Any],
        *,
        as_node: str | None = None,
        overwrite: bool = False,
        context: Any = None,
    ) -> RunResult:
        """Asynchronous ``fork``."""
        new_id = await self.aupdate_state(
            thread_id, values, checkpoint_id=checkpoint_id, as_node=as_node, overwrite=overwrite
        )
        return await self._runner.arun(None, self._threads.config(thread_id, new_id), context=context)

    def visualize(self, format: str = "mermaid") -> str:
        """Visualize the compiled graph in Mermaid or ASCII format."""
        drawable = self.compiled_graph.get_graph()
        fmt = format.lower()
        if fmt == "mermaid":
            return drawable.draw_mermaid()
        if fmt == "ascii":
            try:
                return drawable.draw_ascii()
            except ImportError:  # draw_ascii needs the optional `grandalf` package
                return self._text_outline()
        raise ValueError(f"Unsupported format '{format}' (expected 'mermaid' or 'ascii')")

    def _text_outline(self) -> str:
        lines = [f"Graph: {self.name}"]
        lines += [f"  Node: {n}" for n in self.ir.nodes]
        lines += [f"  Edge: {e.source} -> {e.target}" for e in self.ir.edges]
        lines += [f"  Branch: {c.source} -> {c.route_map}" for c in self.ir.conditional_edges]
        lines += [f"  Parallel: {p.source} -> {p.branch_nodes} -> {p.fan_in or '(join)'}" for p in self.ir.parallel_branches]
        lines += [f"  Loop: {loop.body_node} (max {loop.max_iterations}) -> {loop.exit_node or 'END'}" for loop in self.ir.loops]
        return "\n".join(lines)


class Application:
    """High-level application definition container.

    Args:
        name: Application name.
        checkpointer: A BaseCheckpointStore or raw LangGraph checkpointer; required
            for multi-turn threads and human-in-the-loop resume.
        store: A BaseMemoryStore or raw LangGraph store for long-term memory,
            available to nodes through LangGraph's ``get_store()``.
        listeners: Callables (or objects with ``on_event``) receiving every
            StreamEvent of every run, e.g. ``ExecutionTracer`` or ``StructuredLogger``.
    """

    def __init__(
        self,
        name: str = "graphflow-app",
        checkpointer: Any | BaseCheckpointStore | None = None,
        store: Any | None = None,
        listeners: list[EventListener] | None = None,
    ):
        self.name = name
        self.checkpointer = checkpointer
        self.store = store
        self.listeners = list(listeners or [])
        self._workflows: dict[str, Workflow] = {}
        self._agents: dict[str, Agent] = {}
        self._tools: dict[str, Tool] = {}
        self._compiled: dict[str, CompiledApplication] = {}
        self._compile_lock = threading.RLock()  # concurrent first requests must share one compiled app

    @classmethod
    def from_config(cls, config: Config | dict[str, Any] | str | None = None, **overrides: Any) -> Application:
        """Creates an Application from a config file/dict (see ``graphflow.config.load_config``)."""
        from graphflow.config.loader import Config, load_config

        cfg = config if isinstance(config, Config) else load_config(config)
        params: dict[str, Any] = {"name": cfg.app_name, "checkpointer": cfg.create_checkpointer()}
        params.update(overrides)
        return cls(**params)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def _register(self, registry: dict[str, Any], key: str, item: Any, kind: str) -> Application:
        with self._compile_lock:
            existing = registry.get(key)
            if existing is not None and existing is not item:
                raise ConfigurationError(f"A different {kind} named '{key}' is already registered")
            registry[key] = item
            self._compiled.clear()
        return self

    def add_agent(self, agent: Agent) -> Application:
        """Register an agent with the application."""
        return self._register(self._agents, agent.name, agent, "agent")

    def add_tool(self, tool_or_fn: Tool | Callable[..., Any]) -> Application:
        """Register a reusable tool; agents can reference it by name (``tools=["name"]``)."""
        t = tool_or_fn if isinstance(tool_or_fn, Tool) else tool(tool_or_fn)
        return self._register(self._tools, t.name, t, "tool")

    def add_workflow(self, workflow: Workflow) -> Application:
        """Register a workflow."""
        return self._register(self._workflows, workflow.name, workflow, "workflow")

    def register(self, item: Workflow | Agent | Tool) -> Application:
        """Generic register method for a workflow, agent or tool."""
        if isinstance(item, Workflow):
            return self.add_workflow(item)
        if isinstance(item, Agent):
            return self.add_agent(item)
        if isinstance(item, Tool):
            return self.add_tool(item)
        raise TypeError(f"Cannot register item of type {type(item)}")

    # ------------------------------------------------------------------ #
    # Compilation
    # ------------------------------------------------------------------ #

    def _select_workflow(self, workflow_name: str | None) -> Workflow:
        if workflow_name is not None:
            if workflow_name in self._workflows:
                return self._workflows[workflow_name]
            if workflow_name in self._agents:
                return Workflow(name=workflow_name).then(self._agents[workflow_name])
            raise ConfigurationError(
                f"Application '{self.name}' has no workflow or agent named '{workflow_name}' "
                f"(available: {sorted(self._workflows) + sorted(self._agents)})"
            )
        if len(self._workflows) == 1:
            return next(iter(self._workflows.values()))
        if len(self._workflows) > 1:
            raise ConfigurationError(
                f"Application '{self.name}' has several workflows {sorted(self._workflows)}; "
                "pass workflow_name=... to choose one"
            )
        if len(self._agents) == 1:
            agent = next(iter(self._agents.values()))
            return Workflow(name=f"{agent.name}_flow").then(agent)
        if len(self._agents) > 1:
            raise ConfigurationError(
                f"Application '{self.name}' has several agents {sorted(self._agents)} and no workflow; "
                "compose them in a Workflow (or pass workflow_name=<agent name>)"
            )
        raise ConfigurationError(f"Application '{self.name}' has no registered workflows or agents.")

    def _lower(self, wf: Workflow) -> LoweredWorkflow:
        program = wf.to_program()
        for name, t in self._tools.items():
            program.registry.put(f"apptool:{name}", t)
        return program

    def _resolve_checkpointer(self) -> tuple[Any, bool]:
        cp = self.checkpointer
        if isinstance(cp, BaseCheckpointStore):
            return cp.get_langgraph_checkpointer(), cp.supports_async
        return cp, checkpointer_supports_async(cp)

    def _resolve_store(self) -> Any:
        store = self.store
        if store is not None and hasattr(store, "get_langgraph_store"):
            return store.get_langgraph_store()
        return store

    def compile(self, workflow_name: str | None = None) -> CompiledApplication:
        """Compile a registered workflow (or single agent) into a CompiledApplication.

        Results are cached per workflow; registering anything clears the cache. Safe to
        call from several threads at once: they all get the same CompiledApplication
        (which holds the per-conversation locks ``Thread`` relies on).
        """
        with self._compile_lock:
            return self._compile(workflow_name)

    def _compile(self, workflow_name: str | None) -> CompiledApplication:
        wf = self._select_workflow(workflow_name)
        if wf.name in self._compiled:
            return self._compiled[wf.name]

        program = self._lower(wf)
        checkpointer, supports_async = self._resolve_checkpointer()
        compiled_graph = LangGraphCompiler().compile(
            program.ir, checkpointer=checkpointer, store=self._resolve_store(), registry=program.registry
        )
        compiled = CompiledApplication(
            name=self.name,
            compiled_graph=compiled_graph,
            ir=program.ir,
            registry=program.registry,
            listeners=self.listeners,
            sync_only=not supports_async,
        )
        self._compiled[wf.name] = compiled
        return compiled

    # ------------------------------------------------------------------ #
    # Convenience execution (compile on first use)
    # ------------------------------------------------------------------ #

    def run(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        workflow: str | None = None,
        context: Any = None,
    ) -> RunResult:
        """Compile (if needed) and synchronously run the application."""
        return self.compile(workflow).run(input_data, thread_id=thread_id, config=config, context=context)

    async def arun(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        workflow: str | None = None,
        context: Any = None,
    ) -> RunResult:
        """Compile (if needed) and asynchronously run the application."""
        return await self.compile(workflow).arun(input_data, thread_id=thread_id, config=config, context=context)

    def stream(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        workflow: str | None = None,
        context: Any = None,
    ) -> Iterator[StreamEvent]:
        """Compile (if needed) and stream execution events."""
        return self.compile(workflow).stream(input_data, thread_id=thread_id, config=config, context=context)

    async def astream(
        self,
        input_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        workflow: str | None = None,
        context: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        """Compile (if needed) and asynchronously stream execution events."""
        async for ev in self.compile(workflow).astream(input_data, thread_id=thread_id, config=config, context=context):
            yield ev

    def resume(
        self,
        thread_id: str,
        decision: Decision,
        config: dict[str, Any] | None = None,
        *,
        interrupt_id: str | None = None,
        workflow: str | None = None,
        context: Any = None,
    ) -> RunResult:
        """Resume an interrupted execution."""
        return self.compile(workflow).resume(thread_id, decision, config=config, interrupt_id=interrupt_id, context=context)

    async def aresume(
        self,
        thread_id: str,
        decision: Decision,
        config: dict[str, Any] | None = None,
        *,
        interrupt_id: str | None = None,
        workflow: str | None = None,
        context: Any = None,
    ) -> RunResult:
        """Asynchronously resume an interrupted execution."""
        return await self.compile(workflow).aresume(thread_id, decision, config=config, interrupt_id=interrupt_id, context=context)

    def resume_all(
        self,
        thread_id: str,
        decisions: Mapping[str, Decision],
        config: dict[str, Any] | None = None,
        *,
        workflow: str | None = None,
        context: Any = None,
    ) -> RunResult:
        """Answer several pending interrupts at once: ``{interrupt_id: decision}``."""
        return self.compile(workflow).resume_all(thread_id, decisions, config=config, context=context)

    async def aresume_all(
        self,
        thread_id: str,
        decisions: Mapping[str, Decision],
        config: dict[str, Any] | None = None,
        *,
        workflow: str | None = None,
        context: Any = None,
    ) -> RunResult:
        """Asynchronously answer several pending interrupts at once."""
        return await self.compile(workflow).aresume_all(thread_id, decisions, config=config, context=context)

    def thread(self, thread_id: str, context: Any = None, *, workflow: str | None = None) -> Thread:
        """One conversation of the (compiled) workflow; see ``CompiledApplication.thread``."""
        return self.compile(workflow).thread(thread_id, context=context)

    def history(self, thread_id: str, limit: int | None = None, *, workflow: str | None = None) -> list[Checkpoint]:
        """The thread's checkpoints, newest first."""
        return self.compile(workflow).history(thread_id, limit)

    async def ahistory(
        self, thread_id: str, limit: int | None = None, *, workflow: str | None = None
    ) -> list[Checkpoint]:
        """Asynchronously list the thread's checkpoints, newest first."""
        return await self.compile(workflow).ahistory(thread_id, limit)

    def update_state(self, thread_id: str, values: dict[str, Any], *, workflow: str | None = None, **kwargs: Any) -> str:
        """See ``CompiledApplication.update_state``."""
        return self.compile(workflow).update_state(thread_id, values, **kwargs)

    async def aupdate_state(
        self, thread_id: str, values: dict[str, Any], *, workflow: str | None = None, **kwargs: Any
    ) -> str:
        """See ``CompiledApplication.aupdate_state``."""
        return await self.compile(workflow).aupdate_state(thread_id, values, **kwargs)

    def replay(
        self, thread_id: str, checkpoint_id: str, *, workflow: str | None = None, context: Any = None
    ) -> RunResult:
        """Re-runs the thread from a past checkpoint."""
        return self.compile(workflow).replay(thread_id, checkpoint_id, context=context)

    async def areplay(
        self, thread_id: str, checkpoint_id: str, *, workflow: str | None = None, context: Any = None
    ) -> RunResult:
        """Asynchronous ``replay``."""
        return await self.compile(workflow).areplay(thread_id, checkpoint_id, context=context)

    def fork(
        self, thread_id: str, checkpoint_id: str, values: dict[str, Any], *, workflow: str | None = None, **kwargs: Any
    ) -> RunResult:
        """Writes `values` at a past checkpoint and runs on from there."""
        return self.compile(workflow).fork(thread_id, checkpoint_id, values, **kwargs)

    async def afork(
        self, thread_id: str, checkpoint_id: str, values: dict[str, Any], *, workflow: str | None = None, **kwargs: Any
    ) -> RunResult:
        """Asynchronous ``fork``."""
        return await self.compile(workflow).afork(thread_id, checkpoint_id, values, **kwargs)

    def get_state(self, thread_id: str, *, workflow: str | None = None) -> Any:
        """Inspect the current state snapshot for a thread."""
        return self.compile(workflow).get_state(thread_id)

    def visualize(self, format: str = "mermaid", *, workflow: str | None = None) -> str:
        """Visualize graph."""
        return self.compile(workflow).visualize(format=format)

    def to_langgraph(self, workflow_name: str | None = None) -> Any:
        """Escape Hatch: Returns the uncompiled StateGraph of the selected workflow."""
        program = self._lower(self._select_workflow(workflow_name))
        return LangGraphCompiler().compile_to_state_graph(program.ir, program.registry)
