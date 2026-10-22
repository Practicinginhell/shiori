"""One conversation (a LangGraph thread) as an object.

    thread = app.thread("customer:+15550100", context={"customer_phone": "+15550100"})
    result = thread.send("Hi, can I book a manicure?", message_id="wamid.123")
    if thread.pending:                       # read from the checkpoint
        result = thread.respond(True)

Everything is read from and written to the checkpointer, so a new ``Thread``
object in another request (or worker) continues the same conversation. Runs on
one thread happen one at a time within a process; a ``message_id`` already
processed is skipped (``status == "duplicate"``). Processed ids are kept in the
application's store when it has one (surviving restarts), otherwise in memory.
Across several worker processes, route each thread to one worker: the lock is
per process.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

from graphflow.approvals import ApprovalRequest
from graphflow.errors import ConfigurationError
from graphflow.runtime.checkpoints import Checkpoint
from graphflow.runtime.runner import RunResult, approval_request

if TYPE_CHECKING:
    from graphflow.core.application import CompiledApplication

_MEMORY_LIMIT = 1000  # processed message ids remembered per thread without a store


def _label(text: str) -> str:
    # LangGraph store namespace labels may not contain '.'.
    return text.replace("%", "%25").replace(".", "%2E")


class MessageLog:
    """Which message ids each thread has processed."""

    def __init__(self, store: Any = None):
        self._store = store
        self._memory: dict[str, OrderedDict[str, None]] = {}
        self._guard = threading.Lock()

    def _namespace(self, thread_id: str) -> tuple[str, ...]:
        return ("graphflow", "processed_messages", _label(thread_id))

    def seen(self, thread_id: str, message_id: str) -> bool:
        if self._store is not None:
            return self._store.get(self._namespace(thread_id), message_id) is not None
        with self._guard:
            return message_id in self._memory.get(thread_id, ())

    def record(self, thread_id: str, message_id: str) -> None:
        if self._store is not None:
            self._store.put(self._namespace(thread_id), message_id, {"processed": True})
            return
        with self._guard:
            ids = self._memory.setdefault(thread_id, OrderedDict())
            ids[message_id] = None
            while len(ids) > _MEMORY_LIMIT:
                ids.popitem(last=False)

    async def aseen(self, thread_id: str, message_id: str) -> bool:
        if self._store is not None:
            return await self._store.aget(self._namespace(thread_id), message_id) is not None
        return self.seen(thread_id, message_id)

    async def arecord(self, thread_id: str, message_id: str) -> None:
        if self._store is not None:
            await self._store.aput(self._namespace(thread_id), message_id, {"processed": True})
        else:
            self.record(thread_id, message_id)


class Thread:
    """One conversation of a compiled application. Create it with ``app.thread(...)``.

    Args:
        thread_id: The conversation's id (e.g. ``"customer:+15550100"``).
        context: The run context passed to every run and resume of this object
            (see ``graphflow.context``); it is not saved with the thread.
    """

    def __init__(self, app: CompiledApplication, thread_id: str, context: Any = None):
        if getattr(app.compiled_graph, "checkpointer", None) is None:
            raise ConfigurationError(
                f"Application '{app.name}' has no checkpointer; threads need one "
                "(e.g. Application(checkpointer=InMemoryCheckpointStore()))"
            )
        self._app = app
        self.id = thread_id
        self.context = context

    def __repr__(self) -> str:
        return f"Thread({self.id!r})"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def _config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.id}}

    @contextmanager
    def _serialized(self) -> Iterator[None]:
        lock = self._app._thread_lock(self.id)
        with lock:
            yield

    @asynccontextmanager
    async def _aserialized(self) -> AsyncIterator[None]:
        lock = self._app._thread_lock(self.id)
        while not lock.acquire(blocking=False):  # never blocks the event loop; safe to cancel
            await asyncio.sleep(0.005)
        try:
            yield
        finally:
            lock.release()

    def _input(self, data: Mapping[str, Any] | str | None) -> dict[str, Any]:
        if isinstance(data, str):
            if "messages" not in self._app._schema.fields:
                raise ConfigurationError(
                    f"send('...') adds a user message, but state '{self._app._schema.name}' has no 'messages' field; "
                    "pass a dict of inputs instead"
                )
            return {"messages": [{"role": "user", "content": data}]}
        return dict(data or {})

    def _pending_from(self, snapshot: Any) -> list[ApprovalRequest]:
        node_of = {i.id: task.name for task in snapshot.tasks for i in task.interrupts}
        return [approval_request(i, node_of.get(i.id, ""), self.id) for i in snapshot.interrupts]

    def _duplicate(self, snapshot: Any) -> RunResult:
        values = {k: v for k, v in (snapshot.values or {}).items() if not k.startswith("__")}
        return RunResult(values, status="duplicate", interrupts=self._pending_from(snapshot))

    def _check_not_waiting(self, pending: list[ApprovalRequest]) -> None:
        if pending:
            raise ConfigurationError(
                f"Thread '{self.id}' is waiting for a decision ({pending[0].prompt!r}); "
                "answer it with respond(...) (see thread.pending) before sending new input"
            )

    def _check_waiting(self, pending: list[ApprovalRequest]) -> None:
        if not pending:
            raise ConfigurationError(f"Thread '{self.id}' has nothing waiting for a decision")

    async def _asnapshot(self) -> Any:
        graph = self._app.compiled_graph
        if self._app._runner.sync_only:
            return await asyncio.to_thread(graph.get_state, self._config)
        return await graph.aget_state(self._config)

    # ------------------------------------------------------------------ #
    # Conversation
    # ------------------------------------------------------------------ #

    @property
    def pending(self) -> list[ApprovalRequest]:
        """The decisions this thread is waiting for, read from the checkpoint."""
        return self._pending_from(self._app.compiled_graph.get_state(self._config))

    async def apending(self) -> list[ApprovalRequest]:
        return self._pending_from(await self._asnapshot())

    @property
    def state(self) -> dict[str, Any]:
        """The thread's current state (internal channels hidden)."""
        values = self._app.compiled_graph.get_state(self._config).values or {}
        return {k: v for k, v in values.items() if not k.startswith("__")}

    def send(self, data: Mapping[str, Any] | str | None = None, *, message_id: str | None = None) -> RunResult:
        """Runs the thread with new input (a dict, or text added as a user message).

        With `message_id`, a message this thread already processed is skipped and
        the result has ``status == "duplicate"``. Only successful runs are recorded,
        so a redelivery after a failure runs again.
        """
        graph_input = self._input(data)
        log = self._app._messages
        with self._serialized():
            snapshot = self._app.compiled_graph.get_state(self._config)
            if message_id and log.seen(self.id, message_id):
                return self._duplicate(snapshot)
            self._check_not_waiting(self._pending_from(snapshot))
            result = self._app.run(graph_input, thread_id=self.id, context=self.context)
            if message_id:
                log.record(self.id, message_id)
            return result

    async def asend(self, data: Mapping[str, Any] | str | None = None, *, message_id: str | None = None) -> RunResult:
        graph_input = self._input(data)
        log = self._app._messages
        async with self._aserialized():
            snapshot = await self._asnapshot()
            if message_id and await log.aseen(self.id, message_id):
                return self._duplicate(snapshot)
            self._check_not_waiting(self._pending_from(snapshot))
            result = await self._app.arun(graph_input, thread_id=self.id, context=self.context)
            if message_id:
                await log.arecord(self.id, message_id)
            return result

    def respond(self, decision: Any, *, interrupt_id: str | None = None) -> RunResult:
        """Answers the pending decision (pass `interrupt_id` when several are pending)."""
        with self._serialized():
            self._check_waiting(self.pending)
            return self._app.resume(self.id, decision, interrupt_id=interrupt_id, context=self.context)

    async def arespond(self, decision: Any, *, interrupt_id: str | None = None) -> RunResult:
        async with self._aserialized():
            self._check_waiting(await self.apending())
            return await self._app.aresume(self.id, decision, interrupt_id=interrupt_id, context=self.context)

    def respond_all(self, decisions: Mapping[str, Any]) -> RunResult:
        """Answers several pending decisions at once: ``{interrupt_id: decision}``."""
        with self._serialized():
            self._check_waiting(self.pending)
            return self._app.resume_all(self.id, decisions, context=self.context)

    async def arespond_all(self, decisions: Mapping[str, Any]) -> RunResult:
        async with self._aserialized():
            self._check_waiting(await self.apending())
            return await self._app.aresume_all(self.id, decisions, context=self.context)

    # ------------------------------------------------------------------ #
    # History (see CompiledApplication for the details of each operation)
    # ------------------------------------------------------------------ #

    def history(self, limit: int | None = None) -> list[Checkpoint]:
        return self._app.history(self.id, limit)

    async def ahistory(self, limit: int | None = None) -> list[Checkpoint]:
        return await self._app.ahistory(self.id, limit)

    def update_state(self, values: dict[str, Any], **kwargs: Any) -> str:
        with self._serialized():
            return self._app.update_state(self.id, values, **kwargs)

    async def aupdate_state(self, values: dict[str, Any], **kwargs: Any) -> str:
        async with self._aserialized():
            return await self._app.aupdate_state(self.id, values, **kwargs)

    def replay(self, checkpoint_id: str) -> RunResult:
        with self._serialized():
            return self._app.replay(self.id, checkpoint_id, context=self.context)

    async def areplay(self, checkpoint_id: str) -> RunResult:
        async with self._aserialized():
            return await self._app.areplay(self.id, checkpoint_id, context=self.context)

    def fork(self, checkpoint_id: str, values: dict[str, Any], **kwargs: Any) -> RunResult:
        with self._serialized():
            return self._app.fork(self.id, checkpoint_id, values, context=self.context, **kwargs)

    async def afork(self, checkpoint_id: str, values: dict[str, Any], **kwargs: Any) -> RunResult:
        async with self._aserialized():
            return await self._app.afork(self.id, checkpoint_id, values, context=self.context, **kwargs)
