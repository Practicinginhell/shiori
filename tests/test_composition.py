"""Tests for workflow composition — nesting sub-workflows and subgraphs.

These tests verify that workflows can be composed by embedding one workflow
inside another as a node, enabling modular, reusable workflow construction.
"""

from graphflow import Application, Field, State, Workflow


class ComposableState(State):
    val: int = 0
    history: list = Field.reducer("append")
    output: str = ""


def test_subworkflow_as_node():
    """A sub-workflow should be embeddable as a node in a parent workflow."""
    # Define a reusable sub-workflow
    sub_wf = Workflow("sub_pipeline", state_schema=ComposableState)

    def sub_step1(state: dict) -> dict:
        return {"val": state.get("val", 0) + 10, "history": ["sub_step1"]}

    def sub_step2(state: dict) -> dict:
        return {"val": state.get("val", 0) * 2, "history": ["sub_step2"]}

    sub_wf.then(sub_step1).then(sub_step2)

    # The parent workflow should embed the sub-workflow
    parent_wf = Workflow("parent", state_schema=ComposableState)

    def init_step(state: dict) -> dict:
        return {"val": 5, "history": ["init"]}

    parent_wf.then(init_step)
    parent_wf.then(sub_wf, name="sub_pipeline")  # Embed sub-workflow as a node

    def finalize(state: dict) -> dict:
        return {"output": f"Final val: {state.get('val')}", "history": ["finalize"]}

    parent_wf.then(finalize)

    app = Application("composed_app")
    app.register(parent_wf)
    res = app.run({"val": 0})

    assert res["val"] == 30  # (5 + 10) * 2
    assert "sub_step1" in res["history"]
    assert "sub_step2" in res["history"]
    assert res["output"] == "Final val: 30"


def test_workflow_merge():
    """Two independently defined workflows should be mergeable into one."""
    wf_a = Workflow("pipeline_a", state_schema=ComposableState)

    def step_a(state: dict) -> dict:
        return {"history": ["a"]}

    wf_a.then(step_a)

    wf_b = Workflow("pipeline_b", state_schema=ComposableState)

    def step_b(state: dict) -> dict:
        return {"history": ["b"]}

    wf_b.then(step_b)

    # Merge wf_b into wf_a sequentially
    merged = wf_a.merge(wf_b)

    app = Application("merged_app")
    app.register(merged)
    res = app.run({})

    assert res["history"] == ["a", "b"]


def test_application_with_named_workflow_selection():
    """An application with multiple registered workflows should allow selecting one by name."""
    class WfState(State):
        output: str = ""

    def fast_step(state: dict) -> dict:
        return {"output": "fast"}

    def slow_step(state: dict) -> dict:
        return {"output": "slow"}

    wf_fast = Workflow("fast_pipeline", state_schema=WfState)
    wf_fast.then(fast_step)

    wf_slow = Workflow("slow_pipeline", state_schema=WfState)
    wf_slow.then(slow_step)

    app = Application("multi_wf_app")
    app.register(wf_fast)
    app.register(wf_slow)

    # Should be able to compile and run a specific workflow by name
    compiled = app.compile(workflow_name="slow_pipeline")
    res = compiled.run({})
    assert res["output"] == "slow"

    compiled_fast = app.compile(workflow_name="fast_pipeline")
    res_fast = compiled_fast.run({})
    assert res_fast["output"] == "fast"
