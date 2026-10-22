"""Tests for error handling, retry policies, and resilience features.

These tests verify:
- Node-level retry with exponential backoff works end-to-end
- Workflow-level error handling with fallback nodes
- Timeout enforcement on nodes
- Graceful degradation when a node in a parallel branch fails
"""


import pytest

from graphflow import Application, Field, State, Workflow
from graphflow.core.errors import ExecutionError, TimeoutError


class ResilienceState(State):
    val: int = 0
    output: str = ""
    errors: list = Field.reducer("append")
    history: list = Field.reducer("append")


def test_node_retry_with_exponential_backoff():
    """A node configured with retries should retry on failure before raising."""
    attempt_count = 0

    def flaky_node(state: dict) -> dict:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Flaky failure #{attempt_count}")
        return {"output": "success after retries", "history": ["flaky_node"]}

    wf = Workflow("retry_wf", state_schema=ResilienceState)
    wf.then(flaky_node, retries=3)

    app = Application("retry_app")
    app.register(wf)
    res = app.run({"val": 0})

    assert res["output"] == "success after retries"
    assert attempt_count == 3


def test_node_retry_exhaustion_raises_execution_error():
    """When retries are exhausted, the node should raise ExecutionError."""
    def always_fails(state: dict) -> dict:
        raise RuntimeError("Permanent failure")

    wf = Workflow("exhaust_wf", state_schema=ResilienceState)
    wf.then(always_fails, retries=2)

    app = Application("exhaust_app")
    app.register(wf)

    with pytest.raises(ExecutionError, match="failed after"):
        app.run({"val": 0})


def test_node_timeout_enforcement():
    """A node with a timeout should be interrupted if it exceeds the time limit."""
    import asyncio

    async def slow_node(state: dict) -> dict:
        await asyncio.sleep(10)  # Way too slow
        return {"output": "should not reach here"}

    wf = Workflow("timeout_wf", state_schema=ResilienceState)
    wf.then(slow_node, timeout=0.1)  # 100ms timeout

    app = Application("timeout_app")
    app.register(wf)

    with pytest.raises((TimeoutError, ExecutionError, asyncio.TimeoutError)):
        app.run({"val": 0})


def test_workflow_error_handler_fallback():
    """A workflow should support an on_error handler that catches failures gracefully."""
    def failing_step(state: dict) -> dict:
        raise ValueError("Something broke")

    def error_handler(state: dict, error: Exception) -> dict:
        return {"output": f"Handled error: {error}", "errors": [str(error)]}

    wf = Workflow("fallback_wf", state_schema=ResilienceState)
    wf.then(failing_step)
    wf.on_error(error_handler)  # Register a fallback error handler

    app = Application("fallback_app")
    app.register(wf)
    res = app.run({"val": 0})

    assert "Handled error" in res["output"]
    assert len(res["errors"]) > 0


def test_parallel_branch_partial_failure_isolation():
    """If one branch in a parallel execution fails, others should still complete."""
    def worker_ok(state: dict) -> dict:
        return {"history": ["ok_worker"]}

    def worker_fail(state: dict) -> dict:
        raise RuntimeError("Worker crashed")

    def aggregator(state: dict) -> dict:
        return {"output": f"Got {len(state.get('history', []))} results"}

    wf = Workflow("partial_fail_wf", state_schema=ResilienceState)

    def init(state: dict) -> dict:
        return {}

    wf.then(init)
    wf.parallel(
        branches=[worker_ok, worker_fail],
        fan_in=aggregator,
        from_node="init",
    )

    app = Application("partial_fail_app")
    app.register(wf)

    # With error isolation, the workflow should still produce partial results
    res = app.run({})
    assert "ok_worker" in res.get("history", [])
    assert "results" in res.get("output", "")
