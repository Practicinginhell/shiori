"""Errors keep their type across node boundaries, and model failures are typed.

An application can tell "the model provider is down" (``ModelError``) from "a
tool failed" (``ToolExecutionError``) from "my code has a bug" (``ExecutionError``
wrapping the original), whichever node or nested workflow it happened in, and
every one says which node (``error.node``) after how many attempts.
"""

import asyncio

import pytest

from graphflow import (
    Agent,
    AgentMiddleware,
    Application,
    ExecutionError,
    MockModel,
    ModelError,
    ModelResponse,
    RetryPolicy,
    ToolExecutionError,
    Workflow,
    tool,
)
from graphflow.retry import is_transient_error


class ProviderError(Exception):
    """Shaped like an SDK's HTTP error."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def failing_model(status=503, calls=None):
    def respond(messages):
        if calls is not None:
            calls.append(1)
        raise ProviderError(status)

    return MockModel(respond)


def run_agent(agent, **node):
    wf = Workflow("w").then(agent, **node)
    return Application("a").register(wf).run({"query": "hi"})


def test_a_model_failure_is_a_model_error_at_the_top():
    with pytest.raises(ModelError) as info:
        run_agent(Agent("bot", model=failing_model(503)))
    error = info.value
    assert isinstance(error, ExecutionError)
    assert error.agent == "bot" and error.status_code == 503 and error.transient
    assert error.node == "bot" and error.attempts == 1
    assert isinstance(error.__cause__, ProviderError) and "HTTP 503" in str(error)


def test_model_errors_are_retried_only_when_transient():
    calls = []
    with pytest.raises(ModelError):
        run_agent(Agent("bot", model=failing_model(503, calls)),
                  retry=RetryPolicy(max_attempts=3, initial_interval=0.001))
    assert len(calls) == 3

    calls.clear()
    with pytest.raises(ModelError) as info:
        run_agent(Agent("bot", model=failing_model(400, calls)),
                  retry=RetryPolicy(max_attempts=3, initial_interval=0.001))
    assert len(calls) == 1 and not info.value.transient and not is_transient_error(info.value)


def test_a_standalone_agent_raises_model_error():
    with pytest.raises(ModelError):
        Agent("bot", model=failing_model()).execute({"input": "hi"})


def test_middleware_sees_the_providers_own_exception():
    """wrap_model_call hooks (retries, fallbacks) catch what the provider raised; ModelError is for the app."""
    class Fallback(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except ProviderError:
                return ModelResponse(content="Sorry, I can't answer right now.")

    result = run_agent(Agent("bot", model=failing_model(), middleware=[Fallback()]))
    assert result["output"] == "Sorry, I can't answer right now."


def test_a_tool_error_raised_in_a_node_keeps_its_type():
    @tool
    def lookup(key: str) -> str:
        raise ToolExecutionError(f"no record {key}")

    wf = Workflow("w").then(lambda s: {"output": lookup.execute(key="k")}, name="fetch")
    with pytest.raises(ToolExecutionError, match="no record k") as info:
        Application("a").register(wf).run({})
    assert info.value.node == "fetch" and isinstance(info.value, ExecutionError)


def test_other_errors_are_wrapped_with_the_node_and_cause():
    def broken(state):
        raise KeyError("missing")

    with pytest.raises(ExecutionError, match="Node 'step' failed after 1 attempt") as info:
        Application("a").register(Workflow("w").then(broken, name="step")).run({})
    assert info.value.node == "step" and isinstance(info.value.__cause__, KeyError)


def test_the_innermost_node_is_reported_from_a_nested_workflow():
    inner = Workflow("inner").then(Agent("bot", model=failing_model(400)))
    outer = Workflow("outer").then(lambda s: {}, name="start").then(inner, name="sub")
    with pytest.raises(ModelError) as info:
        Application("a").register(outer).run({"query": "hi"})
    assert info.value.node == "bot"


def test_on_error_receives_the_typed_error():
    seen = []

    def handle(state, error):
        seen.append(error)
        return {"output": "fallback"}

    assert run_agent(Agent("bot", model=failing_model()), on_error=handle)["output"] == "fallback"
    assert isinstance(seen[0], ModelError)


def test_async_runs_raise_model_error():
    app = Application("a").register(Workflow("w").then(Agent("bot", model=failing_model())))
    with pytest.raises(ModelError) as info:
        asyncio.run(app.arun({"query": "hi"}))
    assert info.value.node == "bot"
