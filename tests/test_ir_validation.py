"""Tests for Graphflow IR validation during compilation.

These tests verify that the compiler properly validates the GraphIR
before compiling to LangGraph, rejecting invalid structures with
clear error messages.
"""

import pytest

from graphflow import Application, State, Workflow
from graphflow.core.errors import CompilationError


class ValidateState(State):
    val: int = 0
    output: str = ""


def test_compile_empty_workflow_raises_compilation_error():
    """An empty workflow with no nodes should fail at compilation."""
    wf = Workflow("empty", state_schema=ValidateState)
    app = Application("empty_app")
    app.register(wf)

    with pytest.raises((CompilationError, ValueError)):
        app.compile()


def test_conditional_edge_referencing_nonexistent_node():
    """A branch route pointing to a node name not registered should fail."""
    def step(state: dict) -> dict:
        return {"val": 1}

    def route(state: dict) -> str:
        return "nonexistent_node"

    wf = Workflow("bad_branch", state_schema=ValidateState)
    wf.then(step)
    wf.branch(
        condition=route,
        routes={"nonexistent_node": "does_not_exist_node"},
        from_node="step",
    )

    app = Application("bad_branch_app")
    app.register(wf)

    # The compiler should validate that route targets exist as registered nodes
    with pytest.raises(CompilationError, match="node"):
        app.compile()


def test_loop_referencing_nonexistent_body_node():
    """A loop with a body_node that doesn't exist should fail validation."""
    wf = Workflow("bad_loop", state_schema=ValidateState)

    def fake_condition(state: dict) -> bool:
        return False

    # Manually craft a loop IR referencing a node that was never added
    from graphflow.ir.graph import LoopIR
    wf._loops.append(LoopIR(
        id="loop_1",
        body_node="ghost_node",
        condition=fake_condition,
        exit_node="",
        max_iterations=5,
    ))

    app = Application("bad_loop_app")
    app.register(wf)

    with pytest.raises(CompilationError, match="ghost_node"):
        app.compile()


def test_parallel_branch_referencing_nonexistent_fan_in():
    """A parallel block with a fan_in node that doesn't exist should fail."""
    def worker_a(state: dict) -> dict:
        return {"output": "a"}

    def worker_b(state: dict) -> dict:
        return {"output": "b"}

    wf = Workflow("bad_parallel", state_schema=ValidateState)
    wf.add_node("worker_a", worker_a)
    wf.add_node("worker_b", worker_b)

    from graphflow.ir.graph import ParallelBranchIR
    wf._parallel_branches.append(ParallelBranchIR(
        id="par_1",
        source="worker_a",
        branch_nodes=["worker_a", "worker_b"],
        fan_in="nonexistent_aggregator",
    ))

    app = Application("bad_par_app")
    app.register(wf)

    with pytest.raises(CompilationError, match="nonexistent_aggregator"):
        app.compile()


def test_duplicate_node_registration_raises_error():
    """Registering a node with the same name twice should raise an error or overwrite warning."""
    def step1(state: dict) -> dict:
        return {"val": 1}

    def step2(state: dict) -> dict:
        return {"val": 2}

    wf = Workflow("dup_nodes", state_schema=ValidateState)
    wf.add_node("step", step1)

    # Adding another node with the same name should raise a ConfigurationError
    from graphflow.core.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match="already registered"):
        wf.add_node("step", step2)
