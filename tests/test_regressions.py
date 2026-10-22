"""Regression tests for defects found in the architecture review, plus the behavior added to fix them."""

import asyncio
import os
import tempfile
import time

import pytest
from langgraph.types import Command, interrupt

from graphflow import (
    Agent,
    Application,
    ApprovalDecision,
    ConfigurationError,
    ExecutionError,
    Field,
    HumanApprovalRequired,
    InMemoryCheckpointStore,
    MockModel,
    ModelResponse,
    NodeCompleted,
    NodeStarted,
    RunCompleted,
    RunInterrupted,
    State,
    StateUpdated,
    StateValidationError,
    TimeoutError,
    ToolCall,
    Workflow,
    tool,
)
from graphflow.observability import ExecutionTracer
from graphflow.testing import TestHarness


class Counter(State):
    x: int = 0
    log: list = Field.reducer("append")


def _app(wf: Workflow, **kwargs) -> Application:
    return Application("test", **kwargs).register(wf)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

def test_state_defaults_are_applied_at_runtime():
    class Greeting(State):
        name: str = "World"
        greeting: str = ""

    def hello(state):
        return {"greeting": f"Hello {state['name']}"}

    assert _app(Workflow("w", state_schema=Greeting).then(hello)).run({})["greeting"] == "Hello World"


def test_defaults_do_not_overwrite_persisted_thread_values():
    class S(State):
        mode: str = "initial"
        turns: int = Field.reducer("sum")

    def step(state):
        return {"mode": "changed", "turns": 1}

    app = _app(Workflow("w", state_schema=S).then(step), checkpointer=InMemoryCheckpointStore())
    app.run({}, thread_id="t")
    res = app.run({}, thread_id="t")
    assert res["mode"] == "changed"
    assert res["turns"] == 2


def test_mutable_defaults_are_not_shared():
    class S(State):
        items: list = []  # noqa: RUF012 - the mutable default under test

    a, b = S(), S()
    a["items"].append(1)
    assert b["items"] == []


def test_typed_state_node_receives_state_instance_and_returns_diff():
    def step(state: Counter) -> Counter:
        state.x += 5
        state.log.append("step")
        return state

    res = _app(Workflow("w", state_schema=Counter).then(step)).run({"log": ["start"]})
    assert res["x"] == 5
    assert res["log"] == ["start", "step"]  # appended once, not duplicated


def test_strict_schema_rejects_unknown_input():
    class Strict(State):
        val: int = 0

        class Config:
            strict = True

    app = _app(Workflow("w", state_schema=Strict).then(lambda s: {}, name="noop"))
    with pytest.raises(StateValidationError):
        app.run({"nope": 1})


# --------------------------------------------------------------------------- #
# Nodes: timeouts, retries, interrupts, Command
# --------------------------------------------------------------------------- #

def test_async_node_with_timeout_succeeds():
    async def fast(state):
        return {"x": 1}

    wf = Workflow("w", state_schema=Counter).then(fast, timeout=5)
    assert _app(wf).run({})["x"] == 1


def test_sync_node_timeout_is_enforced():
    def sleepy(state):
        time.sleep(1.0)
        return {"x": 2}

    wf = Workflow("w", state_schema=Counter).then(sleepy, timeout=0.1)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        _app(wf).run({})
    assert time.perf_counter() - started < 0.9


def test_interrupt_inside_user_node_pauses_and_resumes():
    def ask(state):
        answer = interrupt({"prompt": "value?"})
        return {"x": answer["value"]}

    app = _app(Workflow("w", state_schema=Counter).then(ask), checkpointer=InMemoryCheckpointStore())
    first = app.run({}, thread_id="t")
    assert first.interrupted
    assert first.interrupts[0].prompt == "value?"
    assert app.resume("t", {"value": 42})["x"] == 42


def test_node_returning_command_routes():
    def go(state):
        return Command(update={"x": 9}, goto="finish")

    def finish(state):
        return {"x": state["x"] + 1}

    wf = Workflow("w", state_schema=Counter)
    wf.add_node("go", go).add_node("finish", finish)
    wf.then("go")
    assert _app(wf).run({})["x"] == 10


def test_unsupported_node_return_type_fails_clearly():
    wf = Workflow("w", state_schema=Counter).then(lambda s: object(), name="bad")
    with pytest.raises(ExecutionError, match="unsupported type"):
        _app(wf).run({})


def test_node_on_error_fallback():
    def boom(state):
        raise ValueError("nope")

    wf = Workflow("w", state_schema=Counter).then(boom, on_error=lambda s, e: {"log": [f"handled {e}"]})
    assert _app(wf).run({})["log"] == ["handled nope"]


# --------------------------------------------------------------------------- #
# Builder / compiler topology
# --------------------------------------------------------------------------- #

def test_then_after_branch_joins_branch_targets():
    def start(s):
        return {"x": 1}

    def left(s):
        return {"x": 2}

    def right(s):
        return {"x": 3}

    def after(s):
        return {"x": s["x"] * 100}

    wf = Workflow("w", state_schema=Counter).then(start).branch(lambda s: "l", {"l": left, "r": right}).then(after)
    assert _app(wf).run({})["x"] == 200


def test_branch_route_to_end():
    def start(s):
        return {"x": 1}

    def never(s):
        return {"x": -1}

    wf = Workflow("w", state_schema=Counter).then(start).branch(lambda s: "stop", {"stop": None, "go": never})
    assert _app(wf).run({})["x"] == 1


def test_boolean_branch_conditions():
    def start(s):
        return {}

    def yes(s):
        return {"log": ["yes"]}

    def no(s):
        return {"log": ["no"]}

    wf = Workflow("w", state_schema=Counter).then(start).branch(lambda s: s["x"] > 0, {True: yes, False: no})
    assert _app(wf).run({"x": 1})["log"] == ["yes"]


def test_loop_max_iterations_is_enforced():
    def start(s):
        return {"x": 0}

    def inc(s):
        return {"x": s["x"] + 1}

    wf = Workflow("w", state_schema=Counter).then(start).loop(inc, lambda s: True, max_iterations=3)
    assert _app(wf).run({})["x"] == 3


def test_loop_without_exit_continues_to_next_then():
    def inc(s):
        return {"x": s["x"] + 1}

    def done(s):
        return {"log": [f"done at {s['x']}"]}

    wf = Workflow("w", state_schema=Counter).loop(inc, lambda s: s["x"] < 4).then(done)
    assert _app(wf).run({})["log"] == ["done at 4"]


def test_loop_counter_resets_between_runs_on_same_thread():
    def inc(s):
        return {"x": s["x"] + 1}

    wf = Workflow("w", state_schema=Counter).loop(inc, lambda s: True, max_iterations=2)
    app = _app(wf, checkpointer=InMemoryCheckpointStore())
    assert app.run({"x": 0}, thread_id="t")["x"] == 2
    assert app.run({"x": 0}, thread_id="t")["x"] == 2


def test_parallel_without_fan_in_then_joins_all_branches():
    def src(s):
        return {}

    def a(s):
        return {"log": ["a"]}

    def b(s):
        return {"log": ["b"]}

    def join(s):
        return {"x": len(s["log"])}

    wf = Workflow("w", state_schema=Counter).then(src).parallel([a, b]).then(join)
    res = _app(wf).run({})
    assert res["x"] == 2


def test_fan_out_with_no_tasks_still_reaches_fan_in():
    def src(s):
        return {}

    def worker(task):
        return {"log": [task["i"]]}

    def reduce_(s):
        return {"x": len(s["log"])}

    wf = Workflow("w", state_schema=Counter).then(src).fan_out(lambda s: [], worker, reduce_)
    assert _app(wf).run({})["x"] == 0


def test_fan_out_maps_tasks():
    def src(s):
        return {}

    def worker(task):
        return {"log": [task["i"] * 10]}

    def reduce_(s):
        return {"x": sum(s["log"])}

    wf = Workflow("w", state_schema=Counter).then(src).fan_out(lambda s: [{"i": i} for i in range(3)], worker, reduce_)
    assert _app(wf).run({})["x"] == 30


def test_duplicate_node_name_rejected():
    def a(s):
        return {}

    wf = Workflow("w", state_schema=Counter).then(a)
    with pytest.raises(ConfigurationError):
        wf.then(lambda s: {}, name="a")


def test_subworkflow_appends_are_not_duplicated():
    child = Workflow("child", state_schema=Counter).then(lambda s: {"log": ["c1"]}, name="c1")
    parent = Workflow("parent", state_schema=Counter)
    parent.then(lambda s: {"log": ["p1"]}, name="p1").then(child).then(lambda s: {"log": ["p2"]}, name="p2")
    assert _app(parent).run({})["log"] == ["p1", "c1", "p2"]


# --------------------------------------------------------------------------- #
# Human in the loop
# --------------------------------------------------------------------------- #

class Tx(State):
    amount: int = 100
    status: str = "pending"
    reason: str = ""


def _tx_workflow(on_reject=None) -> Workflow:
    def prepare(s):
        return {"amount": 500}

    def execute(s):
        return {"status": "executed"}

    wf = Workflow("tx", state_schema=Tx).then(prepare)
    wf.require_approval(before=execute, prompt="ok?", on_reject=on_reject)
    return wf.then(execute)


def test_hitl_reject_route():
    def cancelled(s):
        return {"status": "cancelled"}

    app = _app(_tx_workflow(on_reject=cancelled), checkpointer=InMemoryCheckpointStore())
    first = app.run({}, thread_id="t")
    assert first.interrupted and first.interrupts[0].node_name == "approval_before_execute"
    res = app.resume("t", ApprovalDecision(approved=False, edits={"reason": "too big"}))
    assert res["status"] == "cancelled"
    assert res["reason"] == "too big"


def test_hitl_rejects_unknown_edit_fields():
    app = _app(_tx_workflow(), checkpointer=InMemoryCheckpointStore())
    app.run({}, thread_id="t")
    again = app.resume("t", ApprovalDecision(approved=True, edits={"nonexistent": 1}))
    assert again.interrupted and "unknown field(s): nonexistent" in again.interrupts[0].payload["error"]


def test_interrupted_stream_ends_with_run_interrupted():
    app = _app(_tx_workflow(), checkpointer=InMemoryCheckpointStore())
    events = list(app.stream({}, thread_id="t"))
    assert isinstance(events[-1], RunInterrupted)
    approval = next(e for e in events if isinstance(e, HumanApprovalRequired))
    assert approval.node_name == "approval_before_execute"
    assert approval.interrupt_id


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

def test_agent_receives_user_query_with_default_schema():
    model = MockModel("hi there")
    res = Application("a").register(Agent("bot", model=model)).run({"query": "hello"})
    assert res["output"] == "hi there"
    assert model.history[0][-1] == {"role": "user", "content": "hello"}


def test_agent_tool_calls_use_structured_messages():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    model = MockModel([
        {"content": "", "tool_calls": [{"id": "call_1", "name": "add", "args": {"a": 1, "b": 2}}]},
        "The answer is 3",
    ])
    Agent("calc", model=model, tools=[add]).execute({"input": "1+2?"})
    second_prompt = model.history[1]
    assert second_prompt[-2] == {
        "role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "add", "args": {"a": 1, "b": 2}}],
    }
    assert second_prompt[-1] == {"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "3"}


def test_agent_memory_flag_controls_history():
    history = {"messages": [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}], "input": "new"}
    with_memory, without = MockModel("x"), MockModel("x")
    Agent("a", model=with_memory).execute(history)
    Agent("b", model=without, memory=False).execute(history)
    assert len(with_memory.history[0]) == 4
    assert [m["content"] for m in without.history[0]] == ["You are a helpful AI assistant.", "new"]


def test_agent_tool_approval_handler():
    @tool(requires_approval=True)
    def delete(target: str) -> str:
        return f"deleted {target}"

    model = MockModel([ModelResponse(tool_calls=[ToolCall(id="c", name="delete", args={"target": "db"})]), "ok"])
    denied = Agent("a", model=model, tools=[delete], approval_handler=lambda tc: False)
    res = denied.execute({"input": "go", "messages": []})
    assert res["messages"][1]["content"] == "Tool call rejected by the approval handler."


def test_agent_tool_approval_interrupts_inside_workflow():
    @tool(requires_approval=True)
    def delete(target: str) -> str:
        return f"deleted {target}"

    model = MockModel(lambda msgs: "done" if msgs[-1]["role"] == "tool" else
                      ModelResponse(tool_calls=[ToolCall(id="c", name="delete", args={"target": "db"})]))
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(Agent("ops", model=model, tools=[delete]))
    first = app.run({"query": "clean up"}, thread_id="t")
    assert first.interrupted and first.interrupts[0].payload["tool"] == "delete"
    res = app.resume("t", True)
    assert res["output"] == "done"
    assert any(m.get("content") == "deleted db" for m in res["messages"])


def test_agent_tool_names_resolve_from_application_registry():
    @tool
    def ping() -> str:
        return "pong"

    model = MockModel([ModelResponse(tool_calls=[ToolCall(id="c", name="ping", args={})]), "done"])
    app = Application("a").register(Agent("bot", model=model, tools=["ping"])).register(ping)
    assert app.run({"query": "ping"})["output"] == "done"
    assert model.history[1][-1]["content"] == "pong"


def test_stream_emits_tool_events_in_node_order():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    model = MockModel([{"content": "", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]}, "3"])
    app = Application("a").register(Agent("calc", model=model, tools=[add]))
    names = [type(e).__name__ for e in app.stream({"query": "1+2"})]
    assert names == [
        "RunStarted", "NodeStarted", "ToolCalled", "ToolCompleted", "StateUpdated", "NodeCompleted", "RunCompleted",
    ]


# --------------------------------------------------------------------------- #
# Application / runtime
# --------------------------------------------------------------------------- #

def test_multiple_agents_without_workflow_is_an_error():
    app = Application("a").register(Agent("one", model=MockModel("1"))).register(Agent("two", model=MockModel("2")))
    with pytest.raises(ConfigurationError, match="several agents"):
        app.run({"query": "q"})
    assert app.run({"query": "q"}, workflow="two")["output"] == "2"


def test_registering_after_compile_invalidates_cache():
    app = Application("a").register(Agent("one", model=MockModel("1")))
    app.run({"query": "q"})
    wf = Workflow("main").then(Agent("two", model=MockModel("2")))
    app.register(wf)
    assert app.run({"query": "q"}, workflow="main")["output"] == "2"


def test_final_state_uses_reducers():
    def a(s):
        return {"log": ["a"]}

    def b(s):
        return {"log": ["b"]}

    wf = Workflow("w", state_schema=Counter).then(a).then(b)
    events = list(_app(wf).stream({}))
    assert next(e for e in events if isinstance(e, RunCompleted)).final_output["log"] == ["a", "b"]
    updated = [e for e in events if isinstance(e, StateUpdated)]
    assert updated[-1].full_state["log"] == ["a", "b"]


def test_node_timings_are_measured():
    def slow(s):
        time.sleep(0.05)
        return {}

    events = list(_app(Workflow("w", state_schema=Counter).then(slow)).stream({}))
    done = next(e for e in events if isinstance(e, NodeCompleted))
    assert done.duration_seconds >= 0.04
    assert [type(e) for e in events].index(NodeStarted) < [type(e) for e in events].index(NodeCompleted)


def test_harness_runs_app_once():
    calls = []

    def step(s):
        calls.append(1)
        return {"x": 1}

    TestHarness.test_app(_app(Workflow("w", state_schema=Counter).then(step)), {})
    assert len(calls) == 1


def test_tracer_as_listener_records_runs():
    tracer = ExecutionTracer()

    def a(s):
        return {}

    _app(Workflow("w", state_schema=Counter).then(a), listeners=[tracer]).run({})
    trace = tracer.last_trace()
    assert trace.status == "completed"
    assert [s.node_name for s in trace.steps] == ["a"]


def test_sqlite_store_works_with_async_api():
    from graphflow import SqliteCheckpointStore

    def hit(s):
        return {"log": ["hit"]}

    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteCheckpointStore(os.path.join(tmp, "cp.db"))
        app = _app(Workflow("w", state_schema=Counter).then(hit), checkpointer=store)

        async def go():
            await app.arun({}, thread_id="t")
            return await app.arun({}, thread_id="t")

        assert asyncio.run(go())["log"] == ["hit", "hit"]
        store.close()


def test_sync_run_inside_running_event_loop():
    async def node(s):
        return {"x": 7}

    app = _app(Workflow("w", state_schema=Counter).then(node))

    async def caller():
        return app.run({})  # e.g. a notebook cell

    assert asyncio.run(caller())["x"] == 7


def test_application_from_config():
    app = Application.from_config({"app": {"name": "cfg-app", "persistence": {"backend": "memory"}}})
    assert app.name == "cfg-app"
    assert isinstance(app.checkpointer, InMemoryCheckpointStore)


def test_import_works_without_sqlite_extra():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['langgraph.checkpoint.sqlite'] = None\n"
        "import graphflow\n"
        "try:\n"
        "    graphflow.SqliteCheckpointStore(':memory:')\n"
        "except ImportError as e:\n"
        "    print('ok' if 'graphflow[sqlite]' in str(e) else e)\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, check=False)
    assert out.stdout.strip() == "ok", out.stderr


# --------------------------------------------------------------------------- #
# Bug hunt, 2026-09-22
# --------------------------------------------------------------------------- #

class _Counter(State):
    n: int = 0
    out: str = ""


def _approval_app():
    wf = Workflow("w", state_schema=_Counter).then(lambda s: {}, name="start").require_approval(
        before=lambda s: {"out": f"n={s['n']}"})
    return Application("x", checkpointer=InMemoryCheckpointStore()).register(wf)


def test_invalid_approval_edits_ask_again_instead_of_sticking_to_the_thread():
    app = _approval_app()
    app.run({}, thread_id="t")
    again = app.resume("t", ApprovalDecision(True, edits={"n": "not a number"}))
    assert again.interrupted and "n" in again.interrupts[0].payload["error"]
    assert app.resume("t", ApprovalDecision(True, edits={"n": 7}))["out"] == "n=7"


def test_the_query_is_saved_to_the_conversation():
    model = MockModel(["r1", "r2"])
    app = Application("d", checkpointer=InMemoryCheckpointStore()).register(Agent("bot", model=model))
    app.run({"query": "q1"}, thread_id="t")
    result = app.run({"query": "q2"}, thread_id="t")
    assert [(m["role"], m["content"]) for m in model.history[1][1:]] == [("user", "q1"), ("assistant", "r1"), ("user", "q2")]
    assert [m["content"] for m in result["messages"]] == ["q1", "r1", "q2", "r2"]


def test_a_tool_timeout_is_reported_to_the_model():
    @tool(timeout=0.05)
    def slow(x: str) -> str:
        time.sleep(0.3)
        return "late"

    model = MockModel([ModelResponse(tool_calls=[ToolCall(id="1", name="slow", args={"x": "a"})]), "handled"])
    assert Agent("a", model=model, tools=[slow]).execute({"input": "go"})["output"] == "handled"
    assert "timed out" in model.history[-1][-1]["content"]
    with pytest.raises(TimeoutError):  # still a TimeoutError when called directly
        slow.execute(x="a")


@pytest.mark.parametrize("thread_id", ["never-started", "finished"])
def test_resuming_a_thread_that_is_not_waiting_is_an_error(thread_id):
    app = _approval_app()
    app.run({}, thread_id="finished")
    app.resume("finished", True)
    with pytest.raises(ConfigurationError, match="nothing waiting"):
        app.resume(thread_id, True)
    with pytest.raises(ConfigurationError, match="nothing waiting"):
        asyncio.run(app.aresume(thread_id, True))
