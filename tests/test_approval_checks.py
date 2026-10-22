"""Approval checks: ``interrupt_on={tool: {"check": fn, "describe": fn}}``.

``check(request)`` returns a reason to refuse a call (or None). It runs before
the reviewer is asked, so a call that can't succeed is never put to them, and
again on their answer, because the world may have changed while they decided
(or they edited the arguments). ``describe(request)`` writes the summary the
reviewer sees. Refusals tell the model who refused and why.
"""

import asyncio
from typing import Annotated

import pytest

from graphflow import (
    Agent,
    Application,
    ConfigurationError,
    FromContext,
    InMemoryCheckpointStore,
    MockModel,
    ModelResponse,
    ToolCall,
    ToolDecision,
    tool,
)


def calls(*items):
    return ModelResponse(tool_calls=[ToolCall(id=i, name=n, args=a) for i, n, a in items])


@pytest.fixture
def ran():
    return []


@pytest.fixture
def booking(ran):
    free = {"9:00", "10:00"}

    @tool
    def book(slot: str, user: Annotated[str, FromContext("user")]) -> str:
        """Books a slot."""
        free.discard(slot)
        ran.append((slot, user))
        return f"booked {slot}"

    def check(request):
        slot = request.tool_call.args["slot"]
        return None if slot in free else f"{slot} is not free"

    def describe(request):
        return f"Book {request.tool_call.args['slot']} for {request.context['user']}"

    return book, check, describe, free


def app_for(agent, **kw):
    return Application("a", checkpointer=InMemoryCheckpointStore(), **kw).register(agent)


def tool_messages(model):
    return [m["content"] for m in model.history[-1] if m["role"] == "tool"]


def test_describe_writes_the_summary_the_reviewer_sees(booking):
    book, check, describe, _ = booking
    model = MockModel([calls(("1", "book", {"slot": "9:00"})), "done"])
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check, "describe": describe}})
    paused = app_for(agent).run({"query": "book"}, thread_id="t", context={"user": "ann"})
    [action] = paused.interrupts[0].actions
    assert action["summary"] == "Book 9:00 for ann"


def test_a_call_failing_its_check_is_refused_without_asking(booking, ran):
    book, check, describe, _ = booking
    model = MockModel([calls(("1", "book", {"slot": "7:00"})), "sorry"])
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check, "describe": describe}})
    result = app_for(agent).run({"query": "book"}, thread_id="t", context={"user": "ann"})
    assert not result.interrupted and result["output"] == "sorry"
    assert ran == []
    [message] = tool_messages(model)
    assert "7:00 is not free" in message and "human" not in message


def test_only_the_calls_that_pass_are_put_to_the_reviewer(booking, ran):
    book, check, _describe, _ = booking
    model = MockModel([calls(("1", "book", {"slot": "7:00"}), ("2", "book", {"slot": "9:00"})), "ok"])
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check}})
    app = app_for(agent)
    paused = app.run({"query": "book"}, thread_id="t", context={"user": "ann"})
    assert [a["tool_call_id"] for a in paused.interrupts[0].actions] == ["2"]
    app.resume("t", [ToolDecision.approve()], context={"user": "ann"})
    assert ran == [("9:00", "ann")]
    assert "not free" in tool_messages(model)[0]


def test_the_check_runs_again_on_the_answer(booking, ran):
    """Someone else took the slot while the reviewer was deciding."""
    book, check, _describe, free = booking
    model = MockModel([calls(("1", "book", {"slot": "9:00"})), "that time was just taken"])
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check}})
    app = app_for(agent)
    app.run({"query": "book"}, thread_id="t", context={"user": "ann"})
    free.discard("9:00")
    result = app.resume("t", True, context={"user": "ann"})
    assert ran == [] and result["output"] == "that time was just taken"
    assert "9:00 is not free" in tool_messages(model)[0]


def test_edited_arguments_are_checked_too(booking, ran):
    book, check, _describe, _ = booking
    model = MockModel([calls(("1", "book", {"slot": "9:00"})), "ok"])
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check}})
    app = app_for(agent)
    app.run({"query": "book"}, thread_id="t", context={"user": "ann"})
    app.resume("t", ToolDecision.edit({"slot": "7:00"}), context={"user": "ann"})
    assert ran == [] and "7:00 is not free" in tool_messages(model)[0]


def test_refusals_before_asking_stay_fixed_across_the_resume(booking, ran):
    """The reviewer answers the calls they were shown, even if a refused one became possible."""
    book, check, _describe, free = booking
    model = MockModel([calls(("1", "book", {"slot": "7:00"}), ("2", "book", {"slot": "9:00"})), "ok"])
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check}})
    app = app_for(agent)
    app.run({"query": "book"}, thread_id="t", context={"user": "ann"})
    free.add("7:00")
    result = app.resume("t", {"decisions": {"2": ToolDecision.approve()}}, context={"user": "ann"})
    assert not result.interrupted and ran == [("9:00", "ann")]


def test_human_and_handler_rejections_say_who_rejected(booking, ran):
    book, *_ = booking
    human = MockModel([calls(("1", "book", {"slot": "9:00"})), "ok"])
    app = app_for(Agent("desk", model=human, tools=[book], interrupt_on={"book": True}))
    app.run({"query": "book"}, thread_id="t", context={"user": "ann"})
    app.resume("t", ToolDecision.reject("not now"), context={"user": "ann"})
    assert tool_messages(human) == ["Tool call rejected by human reviewer. Feedback: not now"]

    auto = MockModel([calls(("1", "book", {"slot": "9:00"})), "ok"])
    agent = Agent("desk", model=auto, tools=[book], interrupt_on={"book": True},
                  approval_handler=lambda call: ToolDecision.reject("policy"))
    agent.execute({"query": "book"}, context={"user": "ann"})
    assert tool_messages(auto) == ["Tool call rejected by the approval handler. Feedback: policy"]


def test_checks_apply_with_an_approval_handler(booking, ran):
    book, check, *_ = booking
    model = MockModel([calls(("1", "book", {"slot": "7:00"})), "ok"])
    asked = []
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check}},
                  approval_handler=lambda call: asked.append(call) or True)
    agent.execute({"query": "book"}, context={"user": "ann"})
    assert asked == [] and ran == []


def test_async_check_and_describe(booking, ran):
    book, check, describe, _ = booking

    async def acheck(request):
        return check(request)

    async def adescribe(request):
        return describe(request)

    model = MockModel([calls(("1", "book", {"slot": "9:00"})), "done"])
    agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": acheck, "describe": adescribe}})
    app = app_for(agent)

    async def go():
        paused = await app.arun({"query": "book"}, thread_id="t", context={"user": "ann"})
        return paused, await app.aresume("t", True, context={"user": "ann"})

    paused, result = asyncio.run(go())
    assert paused.interrupts[0].actions[0]["summary"] == "Book 9:00 for ann"
    assert result["output"] == "done" and ran == [("9:00", "ann")]
    # and the sync path runs async hooks too
    model2 = MockModel([calls(("1", "book", {"slot": "7:00"})), "no"])
    agent2 = Agent("desk", model=model2, tools=[book], interrupt_on={"book": {"check": acheck}})
    assert app_for(agent2).run({"query": "b"}, thread_id="t", context={"user": "ann"})["output"] == "no"


def test_check_and_describe_must_be_callable(booking):
    book, *_ = booking
    with pytest.raises(ConfigurationError, match="check"):
        Agent("desk", model=MockModel("x"), tools=[book], interrupt_on={"book": {"check": "yes"}})


def test_check_and_describe_appear_in_the_ir_as_references(booking):
    book, check, describe, _ = booking
    agent = Agent("desk", model=MockModel("x"), tools=[book], interrupt_on={"book": {"check": check, "describe": describe}})
    policy = agent.to_ir().interrupt_on["book"]
    assert policy["check"].startswith("fn:") and policy["describe"].startswith("fn:")
