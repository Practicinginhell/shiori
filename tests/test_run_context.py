"""Per-run context: data that comes with each run (who the user is, their tenant,
their permissions) rather than living in the conversation state.

It maps to LangGraph's run context (``invoke(..., context=...)``): it reaches every
node, nested agent subgraphs included, and is never saved in checkpoints, so each
run and each resume supplies it again. Tools receive values from it through
arguments the model can't see or set.
"""

import asyncio
import json
from typing import Annotated

import pytest

from graphflow import (
    Agent,
    AgentMiddleware,
    Application,
    ConfigurationError,
    Field,
    FromContext,
    FromState,
    InMemoryCheckpointStore,
    MockModel,
    ModelResponse,
    State,
    ToolCall,
    Workflow,
    get_context,
    tool,
)


class S(State):
    seen: list = Field.reducer("append")
    tenant: str = ""


def calls(name, **args):
    return ModelResponse(tool_calls=[ToolCall(id="c1", name=name, args=args)])


@pytest.fixture
def whoami():
    @tool
    def whoami(greeting: str, phone: Annotated[str, FromContext("customer_phone")]) -> str:
        """Says who is asking."""
        return f"{greeting}, {phone}"

    return whoami


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def test_nodes_read_the_run_context():
    wf = Workflow("w", state_schema=S).then(lambda s: {"seen": [get_context()]}, name="look")
    app = Application("a").register(wf)
    assert app.run({}, context={"user": "ann"})["seen"] == [{"user": "ann"}]
    assert asyncio.run(app.arun({}, context={"user": "bob"}))["seen"] == [{"user": "bob"}]
    assert app.run({})["seen"] == [None]


def test_context_is_not_saved_so_every_resume_passes_it_again():
    def publish(state):
        return {"seen": [get_context()]}

    wf = Workflow("w", state_schema=S).then(lambda s: {}, name="draft").require_approval(publish, "Publish?")
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(wf)
    app.run({}, thread_id="t", context={"user": "ann"})
    assert "ann" not in json.dumps(app.get_state("t").values, default=str)
    assert app.resume("t", True, context={"user": "ann"})["seen"] == [{"user": "ann"}]

    app.run({}, thread_id="t2", context={"user": "ann"})
    assert app.resume("t2", True)["seen"] == [None]


def test_stream_passes_the_context():
    wf = Workflow("w", state_schema=S).then(lambda s: {"seen": [get_context()]}, name="look")
    events = list(Application("a").register(wf).stream({}, context={"user": "ann"}))
    assert events[-1].final_output["seen"] == [{"user": "ann"}]


def test_get_context_outside_a_run_is_none():
    assert get_context() is None


# --------------------------------------------------------------------------- #
# Injected tool arguments
# --------------------------------------------------------------------------- #

def test_injected_arguments_are_hidden_from_the_model(whoami):
    assert set(whoami.args_schema.model_fields) == {"greeting"}
    assert "phone" not in json.dumps(whoami.to_langchain_tool().args)
    assert whoami.to_ir().injected == {"phone": "context:customer_phone"}


def test_the_agent_fills_injected_arguments_from_the_run_context(whoami):
    model = MockModel([calls("whoami", greeting="hi"), "done"])
    app = Application("a").register(Agent("bot", model=model, tools=[whoami]))
    app.run({"query": "who am I?"}, context={"customer_phone": "+1555"})
    assert model.history[1][-1]["content"] == "hi, +1555"


def test_a_value_the_model_supplies_for_an_injected_argument_is_ignored(whoami):
    model = MockModel([calls("whoami", greeting="hi", phone="+1999 (someone else)"), "done"])
    Application("a").register(Agent("bot", model=model, tools=[whoami])).run(
        {"query": "x"}, context={"customer_phone": "+1555"}
    )
    assert model.history[1][-1]["content"] == "hi, +1555"


def test_async_runs_fill_injected_arguments(whoami):
    model = MockModel([calls("whoami", greeting="hi"), "done"])
    app = Application("a").register(Agent("bot", model=model, tools=[whoami]))
    asyncio.run(app.arun({"query": "x"}, context={"customer_phone": "+1555"}))
    assert model.history[1][-1]["content"] == "hi, +1555"


def test_context_reaches_agents_inside_parallel_branches(whoami):
    def agent(name):
        return Agent(name, model=MockModel([calls("whoami", greeting=name), "done"]), tools=[whoami])

    class P(State):
        output: list = Field.reducer("append")
        response: list = Field.reducer("append")
        messages: list = Field.reducer("append")

    left, right = agent("left"), agent("right")
    wf = Workflow("w", state_schema=P).then(lambda s: {}, name="start").parallel([left, right])
    result = Application("a").register(wf).run({}, context={"customer_phone": "+1555"})
    tool_results = sorted(m["content"] for m in result["messages"] if m["role"] == "tool")
    assert tool_results == ["left, +1555", "right, +1555"]


def test_injected_from_state():
    @tool
    def tenant_name(tenant: Annotated[str, FromState("tenant")]) -> str:
        return f"tenant={tenant}"

    model = MockModel([calls("tenant_name"), "done"])
    wf = Workflow("w", state_schema=S).then(Agent("bot", model=model, tools=[tenant_name]))
    Application("a").register(wf).run({"tenant": "acme"})
    assert model.history[1][-1]["content"] == "tenant=acme"


def test_a_missing_context_value_fails_the_run_instead_of_being_guessed(whoami):
    model = MockModel([calls("whoami", greeting="hi"), "done"])
    app = Application("a").register(Agent("bot", model=model, tools=[whoami]))
    with pytest.raises(ConfigurationError, match="customer_phone"):
        app.run({"query": "x"})


def test_a_default_is_used_when_the_context_has_no_value():
    @tool
    def greet(name: Annotated[str, FromContext("user_name")] = "guest") -> str:
        return f"hello {name}"

    model = MockModel([calls("greet"), "done"])
    Application("a").register(Agent("bot", model=model, tools=[greet])).run({"query": "x"})
    assert model.history[1][-1]["content"] == "hello guest"


def test_context_objects_can_be_dataclasses(whoami):
    from dataclasses import dataclass

    @dataclass
    class Caller:
        customer_phone: str

    model = MockModel([calls("whoami", greeting="hi"), "done"])
    Application("a").register(Agent("bot", model=model, tools=[whoami])).run(
        {"query": "x"}, context=Caller("+1555")
    )
    assert model.history[1][-1]["content"] == "hi, +1555"


def test_trusted_code_can_pass_injected_arguments_directly(whoami):
    assert whoami.execute(greeting="hi", phone="+1555") == "hi, +1555"
    with pytest.raises(ConfigurationError, match="phone"):
        whoami.execute(greeting="hi")


def test_standalone_agents_take_a_context(whoami):
    agent = Agent("bot", model=MockModel([calls("whoami", greeting="hi"), "done"]), tools=[whoami])
    assert agent.execute({"input": "x"}, context={"customer_phone": "+1555"})["output"] == "done"
    agent2 = Agent("bot", model=MockModel([calls("whoami", greeting="yo"), "done"]), tools=[whoami])
    assert asyncio.run(agent2.aexecute({"input": "x"}, context={"customer_phone": "+1"}))["output"] == "done"


def test_middleware_sees_the_context(whoami):
    seen = []

    class Record(AgentMiddleware):
        def before_model(self, request):
            seen.append(("model", request.context))

        def wrap_tool_call(self, request, handler):
            seen.append(("tool", request.context))
            return handler(request)

    model = MockModel([calls("whoami", greeting="hi"), "done"])
    Application("a").register(Agent("bot", model=model, tools=[whoami], middleware=[Record()])).run(
        {"query": "x"}, context={"customer_phone": "+1555"}
    )
    assert seen == [("model", {"customer_phone": "+1555"}), ("tool", {"customer_phone": "+1555"}),
                    ("model", {"customer_phone": "+1555"})]


def test_approval_resume_with_context_runs_the_tool_with_it():
    @tool(requires_approval=True)
    def charge(amount: int, account: Annotated[str, FromContext("account")]) -> str:
        return f"charged {amount} to {account}"

    model = MockModel([calls("charge", amount=5), "done"])
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(Agent("bot", model=model, tools=[charge]))
    assert app.run({"query": "x"}, thread_id="t", context={"account": "acc-1"}).interrupted
    app.resume("t", True, context={"account": "acc-1"})
    assert model.history[1][-1]["content"] == "charged 5 to acc-1"
