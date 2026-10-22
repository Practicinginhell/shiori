"""Tests for the plain-data IR, agents compiled as subgraphs, and state validation modes."""

import asyncio
import json
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from graphflow import (
    Agent,
    Application,
    ApprovalDecision,
    CompilationError,
    Field,
    InMemoryCheckpointStore,
    MockModel,
    ModelResponse,
    State,
    StateValidationError,
    ToolCall,
    ToolCalled,
    ToolCompleted,
    Workflow,
    tool,
)
from graphflow.graph.compiler import LangGraphCompiler
from graphflow.ir import Registry, validate_graph


class Report(BaseModel):
    title: str
    score: int


def _max(current, update):
    return update if current is None else max(current, update)


class RichState(State):
    query: str = ""
    best: int = Field(default=0, reducer=_max)
    tags: list[str] = Field.reducer("append")
    report: Report | None = None
    messages: list = Field.reducer("append")


@tool
def lookup(term: str) -> str:
    """Looks a term up."""
    return f"definition of {term}"


def _rich_workflow() -> Workflow:
    child = Workflow("child", state_schema=RichState).then(lambda s: {"tags": ["child"]}, name="child_step")
    agent = Agent("writer", model=MockModel("done"), tools=[lookup], structured_output=None)

    def start(s):
        return {"best": 1}

    def check(s):
        return {"best": s["best"] + 1}

    wf = Workflow("rich", state_schema=RichState)
    wf.then(start).branch(lambda s: "write", {"write": agent, "skip": None}).then(child)
    wf.loop(check, lambda s: s["best"] < 3)
    return wf


# --------------------------------------------------------------------------- #
# IR holds only data
# --------------------------------------------------------------------------- #

def _walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def test_lowered_ir_is_json_serializable_plain_data():
    data = _rich_workflow().to_ir().to_dict()
    json.dumps(data)  # raises if anything isn't plain data
    assert not any(callable(v) or isinstance(v, type) for v in _walk(data))


def test_ir_references_resolve_through_registry():
    program = _rich_workflow().to_program()
    ir, registry = program.ir, program.registry
    assert ir.nodes["writer"].kind == "agent"
    assert ir.nodes["writer"].agent.tools[0].ref in registry
    assert ir.nodes["child"].kind == "subgraph"
    assert ir.state_schema.fields["best"].reducer.startswith("reducer:")
    assert ir.state_schema.fields["tags"].type_repr == "list[str]"
    validate_graph(ir, registry)  # every reference resolves


def test_unresolved_reference_fails_validation():
    program = _rich_workflow().to_program()
    with pytest.raises(CompilationError, match="unresolved reference"):
        validate_graph(program.ir, Registry())


def test_lowered_ir_compiles_and_runs():
    program = _rich_workflow().to_program()
    graph = LangGraphCompiler().compile(program.ir, registry=program.registry)
    result = graph.invoke({"query": "q"})
    assert result["best"] == 3
    assert result["tags"] == ["child"]


# --------------------------------------------------------------------------- #
# Agents compiled as subgraphs
# --------------------------------------------------------------------------- #

def test_agent_resume_does_not_repeat_model_calls_or_tools():
    ran = []

    @tool
    def read(path: str) -> str:
        ran.append(("read", path))
        return "contents"

    @tool(requires_approval=True)
    def delete(path: str) -> str:
        ran.append(("delete", path))
        return "deleted"

    model = MockModel([
        ModelResponse(tool_calls=[
            ToolCall(id="r", name="read", args={"path": "a"}),
            ToolCall(id="d", name="delete", args={"path": "a"}),
        ]),
        "cleaned up",
    ])
    app = Application("ops", checkpointer=InMemoryCheckpointStore()).register(
        Agent("ops", model=model, tools=[read, delete])
    )
    first = app.run({"query": "clean"}, thread_id="t")
    assert first.interrupted
    assert ran == []  # nothing runs before the approval decision
    assert model.call_count == 1

    result = app.resume("t", ApprovalDecision(approved=True))
    assert result["output"] == "cleaned up"
    assert model.call_count == 2  # resumed at the approval step, no repeated model call
    assert ran == [("read", "a"), ("delete", "a")]


def test_rejected_tool_is_not_executed():
    ran = []

    @tool(requires_approval=True)
    def delete(path: str) -> str:
        ran.append(path)
        return "deleted"

    model = MockModel([ModelResponse(tool_calls=[ToolCall(id="d", name="delete", args={"path": "a"})]), "ok"])
    app = Application("ops", checkpointer=InMemoryCheckpointStore()).register(Agent("ops", model=model, tools=[delete]))
    app.run({"query": "clean"}, thread_id="t")
    app.resume("t", False)
    assert ran == []
    assert "rejected" in model.history[-1][-1]["content"]


def test_agent_approval_without_checkpointer_fails_clearly():
    @tool(requires_approval=True)
    def delete(path: str) -> str:
        return "deleted"

    model = MockModel([ModelResponse(tool_calls=[ToolCall(id="d", name="delete", args={"path": "a"})]), "ok"])
    app = Application("ops").register(Agent("ops", model=model, tools=[delete]))
    with pytest.raises(Exception, match="checkpointer"):
        app.run({"query": "clean"})


def test_nested_agent_events_are_attributed_to_the_workflow_node():
    @tool
    def ping() -> str:
        return "pong"

    agent = Agent("helper", model=MockModel([ModelResponse(tool_calls=[ToolCall(id="p", name="ping", args={})]), "ok"]), tools=[ping])
    wf = Workflow("w").then(agent, name="assistant")
    events = list(Application("a").register(wf).stream({"query": "hi"}))
    tool_events = [e for e in events if isinstance(e, (ToolCalled, ToolCompleted))]
    assert [e.node_name for e in tool_events] == ["assistant", "assistant"]


def test_agent_parallel_tool_calls_async():
    @tool
    async def slow(n: int) -> int:
        await asyncio.sleep(0.2)
        return n

    calls = [ToolCall(id=str(i), name="slow", args={"n": i}) for i in range(3)]
    agent = Agent("par", model=MockModel([ModelResponse(tool_calls=calls), "done"]), tools=[slow])

    async def run():
        loop = asyncio.get_running_loop()
        started = loop.time()
        await agent.aexecute({"input": "go"})
        return loop.time() - started

    assert asyncio.run(run()) < 0.5  # three 0.2s tools ran concurrently


def test_agent_structured_output_in_workflow():
    agent = Agent("reporter", model=MockModel('{"title": "Q3", "score": 7}'), structured_output=Report)
    res = Application("a").register(agent).run({"query": "report"})
    assert res["structured_output"] == Report(title="Q3", score=7)


def test_agent_in_function_node_still_works():
    agent = Agent("inner", model=MockModel("inner answer"))

    def wrapper(state):
        return {"output": agent.execute(state)["output"].upper()}

    assert Application("a").register(Workflow("w").then(wrapper)).run({"query": "x"})["output"] == "INNER ANSWER"


# --------------------------------------------------------------------------- #
# Validation modes
# --------------------------------------------------------------------------- #

class Typed(State):
    count: int = 0
    tags: list[str] = Field.reducer("append")
    status: str = "new"


def _typed_app(validation: str, node_update: dict) -> Application:
    Schema = type(f"Typed_{validation}", (Typed,), {"Config": type("Config", (), {"validation": validation})})
    wf = Workflow("w", state_schema=Schema).then(lambda s: node_update, name="step")
    return Application("a").register(wf)


def test_input_is_validated_and_coerced_by_default():
    app = _typed_app("boundaries", {})
    assert app.run({"count": "5"})["count"] == 5
    with pytest.raises(StateValidationError, match="count"):
        app.run({"count": "five"})


def test_append_fields_validate_items():
    app = _typed_app("boundaries", {})
    assert app.run({"tags": "one"})["tags"] == ["one"]
    with pytest.raises(StateValidationError, match="tags"):
        app.run({"tags": [1, {"not": "a string"}]})


def test_boundaries_mode_does_not_check_node_updates():
    assert _typed_app("boundaries", {"count": "not a number"}).run({})["count"] == "not a number"


def test_updates_mode_checks_node_updates():
    with pytest.raises(StateValidationError, match="update from node 'step'"):
        _typed_app("updates", {"count": "not a number"}).run({})


def test_off_mode_skips_type_checks():
    assert _typed_app("off", {}).run({"count": "five"})["count"] == "five"


def test_approval_edits_are_type_checked():
    def prepare(s):
        return {}

    def act(s):
        return {"status": "done"}

    wf = Workflow("w", state_schema=Typed).then(prepare)
    wf.require_approval(before=act).then(act)
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(wf)
    app.run({}, thread_id="t")
    again = app.resume("t", ApprovalDecision(approved=True, edits={"count": "lots"}))
    assert again.interrupted and "count" in again.interrupts[0].payload["error"]  # asks again


def test_dataclass_and_model_fields_validate_without_writing_models():
    @dataclass
    class Point:
        x: int
        y: int

    class Shapes(State):
        origin: Point | None = None
        report: Report | None = None

    def noop(s):
        return {}

    app = Application("a").register(Workflow("w", state_schema=Shapes).then(noop))
    res = app.run({"origin": {"x": 1, "y": "2"}, "report": {"title": "t", "score": 3}})
    assert res["origin"] == Point(1, 2)
    assert res["report"] == Report(title="t", score=3)


def test_state_validate_update_checks_types():
    class S(State):
        count: int = 0

    with pytest.raises(StateValidationError, match="count"):
        S().validate_update({"count": "many"})
    assert S().apply({"count": "3"})["count"] == 3


def test_invalid_validation_mode_rejected():
    from graphflow import ConfigurationError

    with pytest.raises(ConfigurationError):
        type("Bad", (State,), {"__annotations__": {"x": int}, "Config": type("Config", (), {"validation": "sometimes"})})


def test_checkpointed_app_requires_thread_id():
    from graphflow import ConfigurationError

    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(Workflow("w").then(lambda s: {}, name="n"))
    with pytest.raises(ConfigurationError, match="thread_id"):
        app.run({})
