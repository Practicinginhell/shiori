"""Threads: one conversation as a first-class object.

``app.thread(thread_id, context=...)`` owns everything about one conversation:
sending input, what it is waiting for (read from the checkpoint, so any process
can answer), responding, history and forking. Runs on one thread happen one at a
time, and a repeated ``message_id`` (a webhook delivered twice) is skipped.
"""

import asyncio
import threading
import time
from typing import Annotated

import pytest

from graphflow import (
    Agent,
    Application,
    ConfigurationError,
    Field,
    FromContext,
    InMemoryCheckpointStore,
    MockModel,
    ModelResponse,
    State,
    ToolCall,
    Workflow,
    get_context,
    tool,
)
from graphflow.memory import LongTermMemory


def calls(name, **args):
    return ModelResponse(tool_calls=[ToolCall(id="c1", name=name, args=args)])


@pytest.fixture
def ran():
    return []


@pytest.fixture
def approval_app(ran):
    @tool(requires_approval=True)
    def delete(path: str, owner: Annotated[str, FromContext("user")]) -> str:
        ran.append((path, owner))
        return f"deleted {path}"

    def build(store=None, checkpointer=None):
        model = MockModel(lambda messages: "done" if messages[-1]["role"] == "tool" else calls("delete", path="a"))
        return Application(
            "a", checkpointer=checkpointer or InMemoryCheckpointStore(), store=store
        ).register(Agent("ops", model=model, tools=[delete]))

    return build


class Chat(State):
    messages: list = Field.reducer("append")
    log: list = Field.reducer("append")
    output: str = ""
    response: str = ""


def chat_app(model=None, checkpointer=None, store=None, node=None):
    wf = Workflow("chat", state_schema=Chat)
    wf.then(node or Agent("bot", model=model or MockModel("hi!")))
    return Application("c", checkpointer=checkpointer or InMemoryCheckpointStore(), store=store).register(wf)


# --------------------------------------------------------------------------- #
# Sending, pending, responding
# --------------------------------------------------------------------------- #

def test_pending_is_read_from_the_checkpoint_so_any_process_can_answer(approval_app, ran):
    app = approval_app()
    first = app.thread("t", context={"user": "ann"}).send({"query": "clean"})
    assert first.interrupted

    other_worker = app.thread("t", context={"user": "ann"})  # a new object, e.g. another webhook call
    [request] = other_worker.pending
    assert request.interrupt_id == first.interrupts[0].interrupt_id
    assert request.actions[0]["tool"] == "delete" and request.node_name == "ops"

    result = other_worker.respond(True)
    assert result["output"] == "done" and not result.interrupted
    assert ran == [("a", "ann")]  # the thread's context reached the tool on resume
    assert other_worker.pending == []


def test_responding_when_nothing_is_pending_is_an_error(approval_app):
    thread = approval_app().thread("t", context={"user": "ann"})
    with pytest.raises(ConfigurationError, match="nothing"):
        thread.respond(True)


def test_sending_while_a_decision_is_pending_is_an_error(approval_app):
    thread = approval_app().thread("t", context={"user": "ann"})
    thread.send({"query": "clean"})
    with pytest.raises(ConfigurationError, match="waiting"):
        thread.send({"query": "something else"})


def test_text_input_becomes_a_user_message():
    model = MockModel(["hi!", "again!"])
    thread = chat_app(model).thread("t")
    assert thread.send("hello")["output"] == "hi!"
    thread.send("second")
    assert [m["content"] for m in model.history[1] if m["role"] == "user"] == ["hello", "second"]


def test_text_input_needs_a_messages_field():
    class NoMessages(State):
        count: int = 0

    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(
        Workflow("w", state_schema=NoMessages).then(lambda s: {}, name="n")
    )
    with pytest.raises(ConfigurationError, match="messages"):
        app.thread("t").send("hello")


def test_state_history_and_fork_belong_to_the_thread():
    def count(state):
        return {"log": [len(state.get("log") or [])]}

    thread = chat_app(node=count).thread("t")
    thread.send({})
    thread.send({})
    assert thread.state["log"] == [0, 1]
    assert thread.history()[0].values["log"] == [0, 1]
    first_done = next(c for c in thread.history() if c.values.get("log") == [0] and c.done)
    assert thread.fork(first_done.checkpoint_id, {"log": [9]})["log"] == [0, 9]


def test_threads_need_a_checkpointer():
    app = Application("a").register(Workflow("w").then(lambda s: {}, name="n"))
    with pytest.raises(ConfigurationError, match="checkpointer"):
        app.thread("t")


def test_respond_all_answers_parallel_pauses(ran):
    @tool(requires_approval=True)
    def delete(path: str) -> str:
        ran.append(path)
        return "deleted"

    def agent(name):
        return Agent(name, model=MockModel([calls("delete", path=name), "ok"]), tools=[delete])

    class P(State):
        output: list = Field.reducer("append")
        response: list = Field.reducer("append")

    wf = Workflow("w", state_schema=P).then(lambda s: {}, name="start").parallel([agent("left"), agent("right")])
    thread = Application("a", checkpointer=InMemoryCheckpointStore()).register(wf).thread("t")
    thread.send({})
    pending = thread.pending
    assert len(pending) == 2
    thread.respond_all({p.interrupt_id: p.actions[0]["args"]["path"] == "left" for p in pending})
    assert ran == ["left"] and thread.pending == []


# --------------------------------------------------------------------------- #
# Repeated messages
# --------------------------------------------------------------------------- #

def test_a_repeated_message_id_is_skipped():
    runs = []
    thread = chat_app(node=lambda s: runs.append(1) or {}).thread("t")
    assert thread.send({}, message_id="m1").status == "completed"
    duplicate = thread.send({}, message_id="m1")
    assert duplicate.status == "duplicate" and duplicate.duplicate
    assert thread.send({}, message_id="m2").status == "completed"
    assert len(runs) == 2


def test_message_ids_are_per_thread():
    runs = []
    app = chat_app(node=lambda s: runs.append(1) or {})
    app.thread("a").send({}, message_id="m1")
    app.thread("b").send({}, message_id="m1")
    assert len(runs) == 2


def test_a_failed_run_is_not_recorded_so_a_redelivery_runs_again():
    attempts = []

    def flaky(state):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return {}

    thread = chat_app(node=flaky).thread("t")
    with pytest.raises(Exception, match="model unavailable"):
        thread.send({}, message_id="m1")
    assert thread.send({}, message_id="m1").status == "completed"


def test_with_a_store_repeated_messages_are_skipped_across_restarts():
    runs = []
    checkpointer, store = InMemoryCheckpointStore(), LongTermMemory()
    chat_app(node=lambda s: runs.append(1) or {}, checkpointer=checkpointer, store=store).thread("t").send(
        {}, message_id="m1")
    restarted = chat_app(node=lambda s: runs.append(1) or {}, checkpointer=checkpointer, store=store)
    assert restarted.thread("t").send({}, message_id="m1").duplicate
    assert len(runs) == 1


# --------------------------------------------------------------------------- #
# One run at a time per thread
# --------------------------------------------------------------------------- #

def overlap_probe():
    active, peak, lock = [0], [0], threading.Lock()

    def slow(state):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.15)
        with lock:
            active[0] -= 1
        return {}

    return slow, peak


def test_runs_on_one_thread_happen_one_at_a_time():
    slow, peak = overlap_probe()
    app = chat_app(node=slow)
    workers = [threading.Thread(target=lambda: app.thread("t").send({})) for _ in range(3)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert peak[0] == 1
    assert len([c for c in app.thread("t").history() if c.source == "input"]) == 3


def test_different_threads_still_run_in_parallel():
    slow, peak = overlap_probe()
    app = chat_app(node=slow)
    workers = [threading.Thread(target=lambda i=i: app.thread(f"t{i}").send({})) for i in range(3)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert peak[0] > 1


def test_async_runs_on_one_thread_happen_one_at_a_time():
    active, peak = [0], [0]

    async def slow(state):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.05)
        active[0] -= 1
        return {}

    app = chat_app(node=slow)

    async def go():
        await asyncio.gather(*(app.thread("t").asend({}) for _ in range(3)))

    asyncio.run(go())
    assert peak[0] == 1


# --------------------------------------------------------------------------- #
# Async API and context
# --------------------------------------------------------------------------- #

def test_async_send_pending_and_respond(approval_app, ran):
    app = approval_app()

    async def go():
        thread = app.thread("t", context={"user": "bob"})
        await thread.asend({"query": "clean"})
        pending = await thread.apending()
        result = await thread.arespond(True)
        return pending, result

    pending, result = asyncio.run(go())
    assert len(pending) == 1 and result["output"] == "done"
    assert ran == [("a", "bob")]


def test_the_thread_context_reaches_nodes():
    thread = chat_app(node=lambda s: {"log": [get_context()]}).thread("t", context={"user": "ann"})
    assert thread.send({})["log"] == [{"user": "ann"}]


def test_async_threads_work_with_a_sync_only_checkpointer(tmp_path, approval_app, ran):
    from graphflow import SqliteCheckpointStore

    store = SqliteCheckpointStore(str(tmp_path / "threads.db"))
    app = approval_app(checkpointer=store)

    async def go():
        thread = app.thread("t", context={"user": "sql"})
        await thread.asend({"query": "clean"})
        assert len(await thread.apending()) == 1
        await thread.arespond(True)
        return await thread.ahistory()

    try:
        history = asyncio.run(go())
    finally:
        store.close()
    assert history[0].done and ran == [("a", "sql")]
