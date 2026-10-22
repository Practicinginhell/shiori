"""Tests for Agent middleware, tool approval, and advanced agent features.

These tests verify:
- Agent middleware is invoked during execution
- Tools with requires_approval=True trigger approval before execution
- Agent.clone() creates independent copies
- Agent structured output extraction
"""

import pytest

from graphflow import Agent, MockModel, tool
from graphflow.core.errors import ToolExecutionError
from graphflow.models.base import ModelResponse, ToolCall


def test_agent_middleware_pre_and_post_hooks():
    """Middleware functions registered on an agent should be called pre and post execution."""
    call_log = []

    class LoggingMiddleware:
        def pre_execute(self, state: dict, agent_name: str) -> dict:
            call_log.append(f"pre:{agent_name}")
            state["_enriched"] = True
            return state

        def post_execute(self, state: dict, result: dict, agent_name: str) -> dict:
            call_log.append(f"post:{agent_name}")
            result["_audited"] = True
            return result

    mock = MockModel(responses=["Middleware test response"])
    agent = Agent(
        name="middleware_agent",
        model=mock,
        middleware=[LoggingMiddleware()],
    )

    result = agent.execute({"input": "test"})

    assert "pre:middleware_agent" in call_log
    assert "post:middleware_agent" in call_log
    assert result.get("_audited") is True


def test_tool_requires_approval_blocks_execution():
    """A tool with requires_approval=True should not execute without approval callback."""
    @tool(requires_approval=True)
    def dangerous_action(target: str) -> str:
        """Performs a dangerous action that requires human approval."""
        return f"Destroyed {target}"

    # The tool's requires_approval flag should be checked
    # Without an approval handler, executing the tool should raise an error
    mock = MockModel(responses=[
        ModelResponse(
            content="",
            tool_calls=[ToolCall(id="call_1", name="dangerous_action", args={"target": "production"})],
        ),
        ModelResponse(content="Action completed"),
    ])

    agent = Agent(
        name="careful_agent",
        model=mock,
        tools=[dangerous_action],
    )

    # Without approval handler, agent should refuse to execute the tool
    with pytest.raises((ToolExecutionError, RuntimeError), match="approval"):
        agent.execute({"input": "do it"})


def test_agent_clone_creates_independent_copy():
    """Agent.clone() should create a deep copy with an optional name override."""
    original = Agent(
        name="original",
        model=MockModel(responses=["original"]),
        instructions="Be helpful",
        max_tool_iterations=5,
    )

    cloned = original.clone(name="cloned_agent")

    assert cloned.name == "cloned_agent"
    assert cloned.instructions == "Be helpful"
    assert cloned.max_tool_iterations == 5
    # They should be independent
    assert cloned is not original
    assert cloned.model is not original.model


def test_agent_with_structured_output():
    """An agent with structured_output should extract and return structured data."""
    from pydantic import BaseModel

    class TaskResult(BaseModel):
        title: str
        priority: int
        completed: bool

    mock = MockModel(responses=[
        ModelResponse(content='{"title": "Fix bug", "priority": 1, "completed": false}')
    ])

    agent = Agent(
        name="structured_agent",
        model=mock,
        structured_output=TaskResult,
    )

    result = agent.execute({"input": "parse this task"})

    # The agent should parse and return the structured output
    assert "structured_output" in result
    parsed = result["structured_output"]
    assert isinstance(parsed, TaskResult)
    assert parsed.title == "Fix bug"
    assert parsed.priority == 1
    assert parsed.completed is False


def test_agent_max_tool_iterations_enforced():
    """Agent should stop after max_tool_iterations even if model keeps requesting tools."""
    @tool
    def infinite_tool() -> str:
        return "still going"

    # Model always requests tool calls — never gives a final response
    mock = MockModel(responses=[
        ModelResponse(
            content="",
            tool_calls=[ToolCall(id=f"call_{i}", name="infinite_tool", args={})],
        )
        for i in range(20)
    ])

    agent = Agent(
        name="runaway_agent",
        model=mock,
        tools=[infinite_tool],
        max_tool_iterations=3,
    )

    # Should terminate after 3 iterations, not loop forever
    agent.execute({"input": "go"})
    # The mock model's call count should be capped at max_tool_iterations
    assert mock._call_count <= 4  # max_tool_iterations + potential final call
