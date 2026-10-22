"""Tests for Config loading, observability tracing, and the TestHarness.

These tests verify:
- YAML config loading populates Application/Agent/Model settings
- Environment variable override of config values
- Execution tracing records step-level telemetry
- TestHarness utility correctly captures events and asserts behavior
"""

import os
import tempfile

import pytest

from graphflow import Application, Field, State, Workflow
from graphflow.config.loader import load_config
from graphflow.testing.asserts import assert_node_executed, assert_state_contains
from graphflow.testing.harness import TestHarness

# =========================================================================
# Config Loader Tests
# =========================================================================


class ConfigState(State):
    output: str = ""


def test_load_config_from_yaml():
    """load_config should parse a YAML file and return a structured config dict."""
    yaml_content = """
app:
  name: "test-app"
  model:
    provider: "openai"
    model_name: "gpt-4o-mini"
    temperature: 0.7
  persistence:
    backend: "sqlite"
    db_path: "/tmp/test.db"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        path = f.name

    try:
        config = load_config(path)
        assert config["app"]["name"] == "test-app"
        assert config["app"]["model"]["provider"] == "openai"
        assert config["app"]["model"]["temperature"] == 0.7
        assert config["app"]["persistence"]["backend"] == "sqlite"
    finally:
        os.remove(path)


def test_load_config_with_env_var_override():
    """Environment variables prefixed with GRAPHFLOW_ should override YAML values."""
    yaml_content = """
app:
  name: "yaml-name"
  model:
    provider: "openai"
    model_name: "gpt-4o-mini"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        path = f.name

    try:
        os.environ["GRAPHFLOW_APP_NAME"] = "env-override-name"
        config = load_config(path)
        assert config["app"]["name"] == "env-override-name"
    finally:
        os.environ.pop("GRAPHFLOW_APP_NAME", None)
        os.remove(path)


def test_load_config_from_dict():
    """load_config should accept a plain dict as input."""
    config = load_config({"app": {"name": "dict-app"}})
    assert config["app"]["name"] == "dict-app"


# =========================================================================
# Observability / Tracing Tests
# =========================================================================


def test_tracing_records_execution_steps():
    """The tracer should record start/end times, node names, and durations."""
    from graphflow.observability.tracing import ExecutionTracer

    tracer = ExecutionTracer()

    tracer.start_run("run-1", "test_wf")
    tracer.start_node("run-1", "step_a")
    tracer.end_node("run-1", "step_a", updates={"val": 1})
    tracer.start_node("run-1", "step_b")
    tracer.end_node("run-1", "step_b", updates={"val": 2})
    tracer.end_run("run-1", final_state={"val": 2})

    trace = tracer.get_trace("run-1")
    assert trace is not None
    assert trace.run_id == "run-1"
    assert trace.workflow_name == "test_wf"
    assert len(trace.steps) == 2
    assert trace.steps[0].node_name == "step_a"
    assert trace.steps[1].node_name == "step_b"
    assert trace.total_duration_seconds > 0


def test_tracing_captures_errors():
    """The tracer should record error information when a node fails."""
    from graphflow.observability.tracing import ExecutionTracer

    tracer = ExecutionTracer()
    tracer.start_run("run-err", "error_wf")
    tracer.start_node("run-err", "failing_node")
    tracer.record_error("run-err", "failing_node", error="Something went wrong")

    trace = tracer.get_trace("run-err")
    assert trace.steps[0].error == "Something went wrong"


# =========================================================================
# TestHarness Tests
# =========================================================================


class HarnessState(State):
    val: int = 0
    items: list = Field.reducer("append")


def step_inc(state: dict) -> dict:
    return {"val": state.get("val", 0) + 1, "items": ["inc"]}


def step_double(state: dict) -> dict:
    return {"val": state.get("val", 0) * 2, "items": ["double"]}


def test_test_harness_captures_events_and_nodes():
    """TestHarness.test_app should capture all stream events and final state."""
    wf = Workflow("harness_wf", state_schema=HarnessState)
    wf.then(step_inc).then(step_double)

    app = Application("harness_app")
    app.register(wf)

    result = TestHarness.test_app(app, input_data={"val": 5})

    assert result.final_state["val"] == 12  # (5+1)*2
    executed = result.get_executed_nodes()
    assert "step_inc" in executed
    assert "step_double" in executed


def test_assert_node_executed_helper():
    """assert_node_executed should pass for executed nodes and fail for non-executed."""
    wf = Workflow("assert_wf", state_schema=HarnessState)
    wf.then(step_inc)

    app = Application("assert_app")
    app.register(wf)

    result = TestHarness.test_app(app, input_data={"val": 0})

    assert_node_executed(result.events, "step_inc")

    with pytest.raises(AssertionError, match="step_double"):
        assert_node_executed(result.events, "step_double")


def test_assert_state_contains_helper():
    """assert_state_contains should check presence and optionally value."""
    state = {"val": 42, "items": [1, 2]}

    assert_state_contains(state, "val", 42)
    assert_state_contains(state, "items")

    with pytest.raises(AssertionError, match="missing_key"):
        assert_state_contains(state, "missing_key")
