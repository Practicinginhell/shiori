"""Tool-call approval: approve / edit / reject decisions, per-tool policies, batched requests.

Mirrors LangChain's HumanInTheLoopMiddleware semantics on graphflow agents, and
LangGraph's validation-loop pattern (an invalid answer re-asks the human).
"""

import pytest

from graphflow import (
    Agent,
    Application,
    ConfigurationError,
    Field,
    InMemoryCheckpointStore,
    MockModel,
    ModelResponse,
    State,
    ToolCall,
    ToolDecision,
    Workflow,
    tool,
)


def calls(*items):
    return ModelResponse(tool_calls=[ToolCall(id=i, name=n, args=a) for i, n, a in items])


def make_app(agent):
    return Application("a", checkpointer=InMemoryCheckpointStore()).register(agent)


@pytest.fixture
def ran():
    return []


@pytest.fixture
def delete(ran):
    @tool(requires_approval=True)
    def delete(path: str) -> str:
        """Deletes a file."""
        ran.append(path)
        return f"deleted {path}"

    return delete


def test_edit_runs_the_tool_with_the_edited_arguments(ran, delete):
    model = MockModel([calls(("d", "delete", {"path": "/"})), "done"])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    first = app.run({"query": "clean"}, thread_id="t")
    request = first.interrupts[0].payload["action_requests"][0]
    assert request == {
        "tool_call_id": "d",
        "tool": "delete",
        "args": {"path": "/"},
        "description": "Deletes a file.",
        "allowed_decisions": ["approve", "edit", "reject"],
    }

    result = app.resume("t", ToolDecision.edit({"path": "/tmp/x"}))
    assert result["output"] == "done"
    assert ran == ["/tmp/x"]
    assistant = model.history[1][-2]
    assert assistant["tool_calls"][0]["args"] == {"path": "/tmp/x"}  # history shows what actually ran


def test_reject_with_feedback_reaches_the_model(ran, delete):
    model = MockModel([calls(("d", "delete", {"path": "/"})), "ok"])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    app.run({"query": "clean"}, thread_id="t")
    app.resume("t", ToolDecision.reject("never delete the root directory"))
    assert ran == []
    tool_message = model.history[1][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == (
        "Tool call rejected by human reviewer. Feedback: never delete the root directory"
    )


def test_several_pending_calls_are_one_request_answered_in_order(ran, delete):
    model = MockModel([calls(("1", "delete", {"path": "a"}), ("2", "delete", {"path": "b"})), "ok"])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    first = app.run({"query": "clean"}, thread_id="t")
    assert len(first.interrupts) == 1
    assert [r["tool_call_id"] for r in first.interrupts[0].actions] == ["1", "2"]

    app.resume("t", [ToolDecision.approve(), ToolDecision.reject("keep b")])
    assert ran == ["a"]


def test_decisions_can_be_keyed_by_tool_call_id(ran, delete):
    model = MockModel([calls(("1", "delete", {"path": "a"}), ("2", "delete", {"path": "b"})), "ok"])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    app.run({"query": "clean"}, thread_id="t")
    app.resume("t", {"decisions": {"2": {"type": "approve"}, "1": {"type": "reject"}}})
    assert ran == ["b"]


def test_disallowed_decision_asks_again(ran, delete):
    model = MockModel([calls(("d", "delete", {"path": "a"})), "ok"])
    agent = Agent("ops", model=model, tools=[delete], interrupt_on={"delete": {"allowed_decisions": ["approve", "reject"]}})
    app = make_app(agent)
    app.run({"query": "clean"}, thread_id="t")

    again = app.resume("t", ToolDecision.edit({"path": "b"}))
    assert again.interrupted
    assert "not allowed" in again.interrupts[0].payload["error"]
    assert ran == []

    assert app.resume("t", ToolDecision.approve())["output"] == "ok"
    assert ran == ["a"]


def test_invalid_edited_arguments_ask_again(ran, delete):
    model = MockModel([calls(("d", "delete", {"path": "a"})), "ok"])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    app.run({"query": "clean"}, thread_id="t")
    again = app.resume("t", ToolDecision.edit({"wrong": 1}))
    assert again.interrupted and "path" in again.interrupts[0].payload["error"]
    app.resume("t", ToolDecision.edit({"path": "c"}))
    assert ran == ["c"]


def test_wrong_number_of_decisions_asks_again(ran, delete):
    model = MockModel([calls(("1", "delete", {"path": "a"}), ("2", "delete", {"path": "b"})), "ok"])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    app.run({"query": "clean"}, thread_id="t")
    again = app.resume("t", [ToolDecision.approve()])
    assert again.interrupted and "2" in again.interrupts[0].payload["error"]


def test_interrupt_on_enables_and_disables_approval(ran, delete):
    @tool
    def send(to: str) -> str:
        ran.append(to)
        return "sent"

    model = MockModel([calls(("s", "send", {"to": "bob"}), ("d", "delete", {"path": "a"})), "ok"])
    agent = Agent("ops", model=model, tools=[send, delete], interrupt_on={"send": True, "delete": False})
    app = make_app(agent)
    first = app.run({"query": "go"}, thread_id="t")
    assert [r["tool"] for r in first.interrupts[0].actions] == ["send"]
    app.resume("t", True)
    assert ran == ["bob", "a"]


def test_interrupt_on_unknown_tool_is_a_configuration_error(delete):
    with pytest.raises(ConfigurationError, match="nope"):
        Agent("ops", model=MockModel("x"), tools=[delete], interrupt_on={"nope": True}).execute({"input": "x"})


def test_approval_handler_can_edit(ran, delete):
    model = MockModel([calls(("d", "delete", {"path": "/"})), "ok"])
    agent = Agent("ops", model=model, tools=[delete], approval_handler=lambda tc: ToolDecision.edit({"path": "/tmp"}))
    agent.execute({"input": "clean"})
    assert ran == ["/tmp"]


def test_boolean_decisions_still_work(ran, delete):
    model = MockModel([calls(("d", "delete", {"path": "a"})), "ok"])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    app.run({"query": "clean"}, thread_id="t")
    app.resume("t", True)
    assert ran == ["a"]


def test_resume_all_answers_parallel_interrupts(ran, delete):
    def agent(name, path):
        return Agent(name, model=MockModel([calls((name, "delete", {"path": path})), "ok"]), tools=[delete])

    class Parallel(State):
        query: str = ""
        output: list = Field.reducer("append")  # both branches write their answer
        response: list = Field.reducer("append")

    wf = Workflow("w", state_schema=Parallel).then(lambda s: {}, name="start")
    wf.parallel([agent("left", "l"), agent("right", "r")])
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(wf)
    first = app.run({"query": "clean"}, thread_id="t")
    assert len(first.interrupts) == 2
    by_tool_call = {r.actions[0]["tool_call_id"]: r.interrupt_id for r in first.interrupts}
    result = app.resume_all("t", {by_tool_call["left"]: True, by_tool_call["right"]: ToolDecision.reject()})
    assert not result.interrupted
    assert ran == ["l"]


def test_workflow_approval_step_understands_tool_decisions():
    def ship(state):
        return {"output": "shipped"}

    wf = Workflow("w").then(lambda s: {"output": "draft"}, name="draft").require_approval(ship, "Ship it?")
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(wf)
    app.run({}, thread_id="t")
    with pytest.raises(Exception, match="rejected"):
        app.resume("t", ToolDecision.reject("not yet"))


def test_an_edit_does_not_leak_into_a_later_round_reusing_the_call_id(ran, delete):
    # Some models number tool calls per message ("call_0" every round).
    model = MockModel([
        {"tool_calls": [{"name": "delete", "args": {"path": "first"}}]},
        {"tool_calls": [{"name": "delete", "args": {"path": "second"}}]},
        "done",
    ])
    app = make_app(Agent("ops", model=model, tools=[delete]))
    app.run({"query": "clean"}, thread_id="t")
    second_round = app.resume("t", ToolDecision.edit({"path": "edited"}))
    assert second_round.interrupts[0].actions[0]["args"] == {"path": "second"}
    app.resume("t", True)
    assert ran == ["edited", "second"]
    rounds = [m for m in model.history[2] if m.get("tool_calls")]
    assert [r["tool_calls"][0]["args"]["path"] for r in rounds] == ["edited", "second"]


def test_decision_objects_inside_a_mapping_are_stored_as_plain_data(ran, delete):
    from graphflow.approvals import to_resume_value

    value = to_resume_value({"decisions": {"1": ToolDecision.reject("no"), "2": True}})
    assert value == {"decisions": {"1": {"type": "reject", "feedback": "no"}, "2": {"approved": True, "feedback": "", "edits": {}}}}
    assert to_resume_value({"decisions": [ToolDecision.approve()]}) == {"decisions": [{"type": "approve"}]}
