"""Agent middleware: before/after model hooks and model/tool call wrappers.

Mirrors LangChain's agent middleware hooks (before_model, after_model,
wrap_model_call, wrap_tool_call, before_agent, after_agent).
"""

import asyncio

import pytest

from graphflow import (
    Agent,
    AgentMiddleware,
    Application,
    ApprovalDecision,
    ExecutionError,
    InMemoryCheckpointStore,
    MockModel,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolRequest,
    tool,
)


def call(tool_name, /, **args):
    return ModelResponse(tool_calls=[ToolCall(id=f"{tool_name}-1", name=tool_name, args=args)])


def test_before_model_can_rewrite_the_request():
    class KeepLastUserMessage(AgentMiddleware):
        def before_model(self, request):
            system = [m for m in request.messages if m["role"] == "system"]
            return request.override(messages=system + [request.messages[-1]])

    model = MockModel("ok")
    history = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old reply"}]
    Agent("a", model=model, middleware=[KeepLastUserMessage()]).execute({"messages": history, "input": "new"})
    assert [m["content"] for m in model.history[0]][-1] == "new"
    assert len(model.history[0]) == 2


def test_after_model_can_rewrite_the_response():
    class Redact(AgentMiddleware):
        def after_model(self, request, response):
            return ModelResponse(content=response.content.replace("secret", "[redacted]"))

    out = Agent("a", model=MockModel("the secret is 42"), middleware=[Redact()]).execute({"input": "?"})
    assert out["output"] == "the [redacted] is 42"


def test_wrap_model_call_can_retry_a_failing_model():
    attempts = []

    def flaky(messages):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("rate limited")
        return "ok"

    class Retry(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except ConnectionError:
                return handler(request)

    agent = Agent("a", model=MockModel(flaky), middleware=[Retry()])
    assert agent.execute({"input": "hi"})["output"] == "ok"
    attempts.clear()
    assert asyncio.run(agent.aexecute({"input": "hi"}))["output"] == "ok"  # sync hook also serves async runs


def test_async_wrap_hook_is_used_by_async_runs():
    seen = []

    class Both(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            seen.append("sync")
            return handler(request)

        async def awrap_model_call(self, request, handler):
            seen.append("async")
            return await handler(request)

    agent = Agent("a", model=MockModel("ok"), middleware=[Both()])
    agent.execute({"input": "hi"})
    asyncio.run(agent.aexecute({"input": "hi"}))
    assert seen == ["sync", "async"]


def test_wrap_order_first_middleware_is_outermost():
    order = []

    class Named(AgentMiddleware):
        def __init__(self, name):
            self.name = name

        def wrap_model_call(self, request, handler):
            order.append(f"{self.name}:in")
            response = handler(request)
            order.append(f"{self.name}:out")
            return response

    Agent("a", model=MockModel("ok"), middleware=[Named("outer"), Named("inner")]).execute({"input": "hi"})
    assert order == ["outer:in", "inner:in", "inner:out", "outer:out"]


def test_wrap_tool_call_can_short_circuit_a_tool():
    ran = []

    @tool
    def drop_table(name: str) -> str:
        ran.append(name)
        return "dropped"

    class Guard(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            assert isinstance(request, ToolRequest)
            if request.tool_call.name == "drop_table":
                return "Blocked by policy."
            return handler(request)

    model = MockModel([call("drop_table", name="users"), "done"])
    Agent("a", model=model, tools=[drop_table], middleware=[Guard()]).execute({"input": "go"})
    assert ran == []
    assert model.history[1][-1]["content"] == "Blocked by policy."


def test_wrap_tool_call_sees_tool_errors_and_can_retry():
    attempts = []

    @tool
    def fetch(url: str) -> str:
        attempts.append(url)
        if len(attempts) == 1:
            raise ValueError("boom")
        return "page"

    class RetryTool(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            try:
                return handler(request)
            except Exception:
                return handler(request)

    model = MockModel([call("fetch", url="u"), "done"])
    Agent("a", model=model, tools=[fetch], middleware=[RetryTool()]).execute({"input": "go"})
    assert model.history[1][-1]["content"] == "page"


def test_wrap_tool_call_can_rewrite_arguments():
    @tool
    def search(query: str) -> str:
        return f"results for {query}"

    class Normalize(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            tc = request.tool_call
            return handler(request.override(tool_call=ToolCall(tc.id, tc.name, {"query": tc.args["query"].lower()})))

    model = MockModel([call("search", query="LangGraph"), "done"])
    Agent("a", model=model, tools=[search], middleware=[Normalize()]).execute({"input": "go"})
    assert model.history[1][-1]["content"] == "results for langgraph"


def test_before_and_after_agent_hooks():
    class Tag(AgentMiddleware):
        def before_agent(self, state, agent_name):
            return {**state, "input": state["input"] + "!"}

        def after_agent(self, state, update, agent_name):
            return {**update, "output": f"[{agent_name}] {update['output']}"}

    model = MockModel(lambda messages: messages[-1]["content"])
    out = Agent("echo", model=model, middleware=[Tag()]).execute({"input": "hi"})
    assert out["output"] == "[echo] hi!"


def test_duck_typed_middleware_without_base_class():
    class Plain:
        def before_model(self, request):
            return request.override(messages=request.messages + [{"role": "user", "content": "be brief"}])

    model = MockModel("ok")
    Agent("a", model=model, middleware=[Plain()]).execute({"input": "hi"})
    assert model.history[0][-1]["content"] == "be brief"


def test_model_request_exposes_agent_context():
    seen = {}

    @tool
    def ping() -> str:
        return "pong"

    class Inspect(AgentMiddleware):
        def before_model(self, request: ModelRequest):
            seen.setdefault("iterations", []).append(request.iteration)
            seen["agent"] = request.agent
            seen["tools"] = [t.name for t in request.tools]
            seen["state_input"] = request.state.get("input")

    Agent("inspector", model=MockModel([call("ping"), "done"]), tools=[ping], middleware=[Inspect()]).execute(
        {"input": "hi"}
    )
    assert seen == {"iterations": [0, 1], "agent": "inspector", "tools": ["ping"], "state_input": "hi"}


def test_model_hooks_do_not_rerun_on_approval_resume():
    before = []

    @tool(requires_approval=True)
    def delete(path: str) -> str:
        return "deleted"

    class Count(AgentMiddleware):
        def before_model(self, request):
            before.append(request.iteration)

    model = MockModel([call("delete", path="a"), "done"])
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(
        Agent("ops", model=model, tools=[delete], middleware=[Count()])
    )
    assert app.run({"query": "x"}, thread_id="t").interrupted
    assert app.resume("t", ApprovalDecision(approved=True))["output"] == "done"
    assert before == [0, 1] and model.call_count == 2


def test_middleware_errors_fail_the_agent():
    class Broken(AgentMiddleware):
        def before_model(self, request):
            raise RuntimeError("broken middleware")

    with pytest.raises(Exception, match="broken middleware"):
        Agent("a", model=MockModel("ok"), middleware=[Broken()]).execute({"input": "hi"})


def test_structured_output_retry_tells_the_model_what_was_wrong():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    model = MockModel(['{"value": "not a number"}', '{"value": 7}'])
    out = Agent("a", model=model, structured_output=Answer).execute({"input": "?"})
    assert out["structured_output"] == Answer(value=7)
    feedback = model.history[1][-1]
    assert feedback["role"] == "user" and "Answer" in feedback["content"] and "value" in feedback["content"]


def test_structured_output_gives_up_after_its_retries():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    model = MockModel("not json")
    with pytest.raises(ExecutionError, match="valid Answer"):
        Agent("a", model=model, structured_output=Answer, structured_output_retries=2).execute({"input": "?"})
    assert model.call_count == 3
