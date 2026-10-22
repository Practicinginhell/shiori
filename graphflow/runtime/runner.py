"""Graphflow Runtime Runner.

Drives a compiled LangGraph graph and translates its native stream into
Graphflow events. ``run`` and ``stream`` share one code path, so listeners
(tracers, loggers) observe every execution, streamed or not.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

from langgraph.types import Command

from graphflow.approvals import ApprovalDecision, ApprovalRequest, to_resume_value
from graphflow.errors import ExecutionError, GraphflowError
from graphflow.events import (
    CustomEvent,
    HumanApprovalRequired,
    ModelToken,
    NodeCompleted,
    NodeStarted,
    RunCompleted,
    RunFailed,
    RunInterrupted,
    RunStarted,
    StateUpdated,
    StreamEvent,
)

logger = logging.getLogger("graphflow.runtime")

EventListener = Callable[[StreamEvent], Any]


def _is_hidden(name: str) -> bool:
    return name.startswith("__")


def _visible(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    return {k: v for k, v in state.items() if not _is_hidden(k)}


def approval_request(interrupt: Any, node_name: str = "", thread_id: str = "") -> ApprovalRequest:
    """An ApprovalRequest for a LangGraph ``Interrupt`` (from a run's stream or a saved snapshot)."""
    value = getattr(interrupt, "value", interrupt)
    payload = value if isinstance(value, dict) else {"value": value}
    return ApprovalRequest(
        prompt=str(payload.get("prompt", value if not isinstance(value, dict) else "Approval required")),
        description=str(payload.get("description", "")),
        payload=value,
        node_name=str(payload.get("node_name", node_name)),
        thread_id=thread_id,
        interrupt_id=getattr(interrupt, "id", "") or "",
    )


class RunResult(dict):
    """Final state of a run (a plain dict), plus run metadata.

    ``status`` is ``"completed"``, ``"interrupted"`` (``interrupts`` lists the
    pending approval requests to resume) or ``"duplicate"`` (``Thread.send`` skipped
    a message id it had already processed; the dict is the thread's current state).
    """

    def __init__(
        self,
        state: dict[str, Any],
        status: str = "completed",
        interrupts: list[ApprovalRequest] | None = None,
        run_id: str = "",
    ):
        super().__init__(state)
        self.status = status
        self.interrupts = interrupts or []
        self.run_id = run_id

    @property
    def interrupted(self) -> bool:
        return self.status == "interrupted"

    @property
    def duplicate(self) -> bool:
        return self.status == "duplicate"

    @property
    def state(self) -> dict[str, Any]:
        return dict(self)


class _EventTranslator:
    """Converts LangGraph (mode, chunk) pairs into Graphflow events for one run."""

    def __init__(self, run_id: str, thread_id: str, initial_input: Any):
        self.run_id = run_id
        self.thread_id = thread_id
        self.state: dict[str, Any] = _visible(initial_input)
        self.interrupts: list[ApprovalRequest] = []
        self.started_at = time.perf_counter()
        self.last_node = ""
        self._task_started: dict[str, float] = {}
        self._completed: list[tuple[str, dict[str, Any], float]] = []

    def feed(self, namespace: tuple[str, ...], mode: str, data: Any) -> list[StreamEvent]:
        """Translates one chunk. `namespace` is non-empty for chunks from nested graphs
        (agents, sub-workflows); those only contribute custom events and tokens,
        attributed to the top-level node that ran them."""
        outer_node = namespace[0].split(":", 1)[0] if namespace else ""
        if outer_node and mode not in ("custom", "messages"):
            return []
        if mode == "tasks":
            return self._on_task(data)
        if mode == "values":
            return self._on_values(data)
        if mode == "custom":
            event = data if isinstance(data, StreamEvent) else CustomEvent(data=data)
            event.run_id = self.run_id
            if outer_node and hasattr(event, "node_name"):
                event.node_name = outer_node  # type: ignore[attr-defined]
            return [event]
        if mode == "messages":
            chunk, metadata = data
            text = getattr(chunk, "content", "")
            node = outer_node or (metadata or {}).get("langgraph_node", "")
            if isinstance(text, str) and text and not _is_hidden(node):
                return [ModelToken(run_id=self.run_id, token=text, node_name=node, agent_name=node)]
        return []

    def _on_task(self, task: dict[str, Any]) -> list[StreamEvent]:
        name = task.get("name", "")
        if _is_hidden(name):
            return []
        if "result" not in task:  # task start
            self._task_started[task["id"]] = time.perf_counter()
            self.last_node = name
            return [NodeStarted(run_id=self.run_id, node_name=name, input_state=_visible(task.get("input")))]
        if task.get("error") or task.get("interrupts"):
            return []
        duration = time.perf_counter() - self._task_started.pop(task["id"], time.perf_counter())
        result = task.get("result")
        updates = _visible(result) if isinstance(result, dict) else {}
        self._completed.append((name, updates, duration))
        return []

    def _on_values(self, values: dict[str, Any]) -> list[StreamEvent]:
        for intr in values.get("__interrupt__", ()) or ():
            self._record_interrupt(intr)
        self.state = _visible(values)
        return self.flush()

    def _record_interrupt(self, intr: Any) -> None:
        request = approval_request(intr, self.last_node, self.thread_id)
        if request.interrupt_id and any(r.interrupt_id == request.interrupt_id for r in self.interrupts):
            return
        self.interrupts.append(request)

    def flush(self) -> list[StreamEvent]:
        """Emits StateUpdated + NodeCompleted for tasks finished in the last superstep."""
        events: list[StreamEvent] = []
        for name, updates, duration in self._completed:
            events.append(StateUpdated(run_id=self.run_id, node_name=name, updates=updates, full_state=dict(self.state)))
            events.append(NodeCompleted(run_id=self.run_id, node_name=name, duration_seconds=duration, output_updates=updates))
        self._completed.clear()
        return events

    def finish(self) -> list[StreamEvent]:
        events = self.flush()
        elapsed = time.perf_counter() - self.started_at
        if self.interrupts:
            for req in self.interrupts:
                events.append(HumanApprovalRequired(
                    run_id=self.run_id,
                    prompt=req.prompt,
                    node_name=req.node_name,
                    thread_id=self.thread_id,
                    payload=req.payload,
                    interrupt_id=req.interrupt_id,
                ))
            events.append(RunInterrupted(
                run_id=self.run_id, state=dict(self.state), interrupts=list(self.interrupts),
                total_duration_seconds=elapsed,
            ))
        else:
            events.append(RunCompleted(run_id=self.run_id, final_output=dict(self.state), total_duration_seconds=elapsed))
        return events

    def result(self) -> RunResult:
        status = "interrupted" if self.interrupts else "completed"
        return RunResult(self.state, status=status, interrupts=list(self.interrupts), run_id=self.run_id)


class RuntimeRunner:
    """Manages the execution lifecycle of a compiled LangGraph graph."""

    def __init__(
        self,
        compiled_graph: Any,
        name: str = "app",
        listeners: list[EventListener] | None = None,
        sync_only: bool = False,
    ):
        self.compiled_graph = compiled_graph
        self.name = name
        self.listeners = list(listeners or [])
        # A sync-only checkpointer (e.g. SqliteSaver) can't serve the async API directly;
        # async calls then run the sync graph in a worker thread.
        self.sync_only = sync_only

    # ------------------------------------------------------------------ #
    # Listener dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(self, event: StreamEvent) -> StreamEvent:
        for listener in self.listeners:
            try:
                handler = getattr(listener, "on_event", listener)
                handler(event)
            except Exception:  # listeners must never break a run
                logger.exception("Event listener %r failed", listener)
        return event

    @staticmethod
    def _modes(tokens: bool) -> list[str]:
        return ["tasks", "values", "custom", "messages"] if tokens else ["tasks", "values", "custom"]

    # ------------------------------------------------------------------ #
    # Core event streams
    # ------------------------------------------------------------------ #

    def _stream(
        self, graph_input: Any, config: dict[str, Any], tokens: bool, translator: _EventTranslator, context: Any = None
    ) -> Iterator[StreamEvent]:
        yield self._dispatch(RunStarted(
            run_id=translator.run_id, workflow_name=self.name,
            initial_input=_visible(graph_input) if isinstance(graph_input, dict) else {},
        ))
        try:
            for namespace, mode, data in self.compiled_graph.stream(
                graph_input, config, context=context, stream_mode=self._modes(tokens), subgraphs=True
            ):
                for ev in translator.feed(namespace, mode, data):
                    yield self._dispatch(ev)
        except Exception as e:
            yield self._dispatch(RunFailed(run_id=translator.run_id, error=str(e), failed_node=translator.last_node))
            if isinstance(e, GraphflowError):
                raise
            raise ExecutionError(f"Application execution failed: {e}") from e
        for ev in translator.finish():
            yield self._dispatch(ev)

    async def _astream(
        self, graph_input: Any, config: dict[str, Any], tokens: bool, translator: _EventTranslator, context: Any = None
    ) -> AsyncIterator[StreamEvent]:
        if self.sync_only:
            async for ev in _iterate_in_thread(lambda: self._stream(graph_input, config, tokens, translator, context)):
                yield ev
            return

        yield self._dispatch(RunStarted(
            run_id=translator.run_id, workflow_name=self.name,
            initial_input=_visible(graph_input) if isinstance(graph_input, dict) else {},
        ))
        try:
            async for namespace, mode, data in self.compiled_graph.astream(
                graph_input, config, context=context, stream_mode=self._modes(tokens), subgraphs=True
            ):
                for ev in translator.feed(namespace, mode, data):
                    yield self._dispatch(ev)
        except Exception as e:
            yield self._dispatch(RunFailed(run_id=translator.run_id, error=str(e), failed_node=translator.last_node))
            if isinstance(e, GraphflowError):
                raise
            raise ExecutionError(f"Application execution failed: {e}") from e
        for ev in translator.finish():
            yield self._dispatch(ev)

    @staticmethod
    def _translator(graph_input: Any, config: dict[str, Any]) -> _EventTranslator:
        thread_id = str((config.get("configurable") or {}).get("thread_id", ""))
        initial = graph_input if isinstance(graph_input, dict) else {}
        return _EventTranslator(run_id=str(uuid.uuid4()), thread_id=thread_id, initial_input=initial)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def stream(
        self, graph_input: Any, config: dict[str, Any] | None = None, tokens: bool = True, *, context: Any = None
    ) -> Iterator[StreamEvent]:
        config = config or {}
        yield from self._stream(graph_input, config, tokens, self._translator(graph_input, config), context)

    async def astream(
        self, graph_input: Any, config: dict[str, Any] | None = None, tokens: bool = True, *, context: Any = None
    ) -> AsyncIterator[StreamEvent]:
        config = config or {}
        async for ev in self._astream(graph_input, config, tokens, self._translator(graph_input, config), context):
            yield ev

    def run(self, graph_input: Any, config: dict[str, Any] | None = None, *, context: Any = None) -> RunResult:
        config = config or {}
        translator = self._translator(graph_input, config)
        for _ in self._stream(graph_input, config, False, translator, context):
            pass
        return translator.result()

    async def arun(self, graph_input: Any, config: dict[str, Any] | None = None, *, context: Any = None) -> RunResult:
        config = config or {}
        translator = self._translator(graph_input, config)
        async for _ in self._astream(graph_input, config, False, translator, context):
            pass
        return translator.result()

    @staticmethod
    def resume_command(
        decision: ApprovalDecision | bool | dict[str, Any] | Any, interrupt_id: str | None = None
    ) -> Command:
        value = to_resume_value(decision)
        return Command(resume={interrupt_id: value} if interrupt_id else value)

    @staticmethod
    def resume_all_command(decisions: Mapping[str, Any]) -> Command:
        """Answers several pending interrupts at once (interrupt_id -> decision)."""
        if not decisions:
            raise ValueError("resume_all needs at least one interrupt_id -> decision")
        return Command(resume={interrupt_id: to_resume_value(d) for interrupt_id, d in decisions.items()})

    def resume(
        self, decision: Any, config: dict[str, Any], interrupt_id: str | None = None, *, context: Any = None
    ) -> RunResult:
        return self.run(self.resume_command(decision, interrupt_id), config, context=context)

    async def aresume(
        self, decision: Any, config: dict[str, Any], interrupt_id: str | None = None, *, context: Any = None
    ) -> RunResult:
        return await self.arun(self.resume_command(decision, interrupt_id), config, context=context)


_DONE = object()


async def _iterate_in_thread(make_iter: Callable[[], Iterator[Any]]) -> AsyncIterator[Any]:
    """Consumes a blocking iterator in a worker thread, yielding its items asynchronously."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def worker() -> None:
        try:
            for item in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as e:  # noqa: BLE001 - forwarded to the consumer, which re-raises it
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    thread = threading.Thread(target=worker, name="graphflow-sync-runner", daemon=True)
    thread.start()
    while True:
        item = await queue.get()
        if item is _DONE:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
