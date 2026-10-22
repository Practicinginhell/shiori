"""Unit tests for Graphflow workflows and compiler escape hatches."""

from graphflow import Application, Field, State, Workflow


class FlowState(State):
    val: int = 0
    history: list = Field.reducer("append")
    category: str = ""
    items: list = Field.reducer("append")
    output: str = ""


def test_sequential_workflow():
    def step1(state: dict) -> dict:
        return {"val": state.get("val", 0) + 1, "history": ["step1"]}

    def step2(state: dict) -> dict:
        return {"val": state.get("val", 0) * 2, "history": ["step2"]}

    wf = Workflow("seq", state_schema=FlowState)
    wf.then(step1).then(step2)

    app = Application("seq_app")
    app.register(wf)
    res = app.run({"val": 5})

    assert res["val"] == 12  # (5 + 1) * 2
    assert res["history"] == ["step1", "step2"]


def test_conditional_branching():
    def classifier(state: dict) -> dict:
        val = state.get("val", 0)
        cat = "high" if val >= 10 else "low"
        return {"category": cat}

    def route(state: dict) -> str:
        return state.get("category", "low")

    def handle_high(state: dict) -> dict:
        return {"output": "Processed high value"}

    def handle_low(state: dict) -> dict:
        return {"output": "Processed low value"}

    wf = Workflow("branch_flow", state_schema=FlowState)
    wf.then(classifier)
    wf.branch(
        condition=route,
        routes={"high": handle_high, "low": handle_low},
        from_node="classifier",
    )

    app = Application("branch_app")
    app.register(wf)

    res_high = app.run({"val": 15})
    assert res_high["output"] == "Processed high value"

    res_low = app.run({"val": 3})
    assert res_low["output"] == "Processed low value"


def test_looping_workflow():
    def increment_node(state: dict) -> dict:
        current = state.get("val", 0) + 1
        return {"val": current, "history": [f"iter_{current}"]}

    def keep_looping(state: dict) -> bool:
        return state.get("val", 0) < 4

    def done_node(state: dict) -> dict:
        return {"output": f"Done at {state.get('val')}"}

    wf = Workflow("loop_flow", state_schema=FlowState)
    wf.loop(
        body_node=increment_node,
        while_condition=keep_looping,
        exit_node=done_node,
        max_iterations=10,
    )

    app = Application("loop_app")
    app.register(wf)
    res = app.run({"val": 0})

    assert res["val"] == 4
    assert len(res["history"]) == 4
    assert res["output"] == "Done at 4"


def test_parallel_workflow():
    def start_node(state: dict) -> dict:
        return {"history": ["start"]}

    def worker_a(state: dict) -> dict:
        return {"items": ["item_from_A"]}

    def worker_b(state: dict) -> dict:
        return {"items": ["item_from_B"]}

    def aggregator(state: dict) -> dict:
        items = state.get("items", [])
        return {"output": f"Combined {len(items)} items"}

    wf = Workflow("par_flow", state_schema=FlowState)
    wf.then(start_node)
    wf.parallel(branches=[worker_a, worker_b], fan_in=aggregator, from_node="start_node")

    app = Application("par_app")
    app.register(wf)
    res = app.run({})

    assert set(res["items"]) == {"item_from_A", "item_from_B"}
    assert res["output"] == "Combined 2 items"


def test_dynamic_fan_out_send_api():
    class MapReduceState(State):
        tasks: list = Field.reducer("append")
        results: list = Field.reducer("append")
        summary: str = ""

    def init_node(state: dict) -> dict:
        return {"tasks": ["task1", "task2", "task3"]}

    def partition_tasks(state: dict) -> list[dict]:
        return [{"task_name": t} for t in state.get("tasks", [])]

    def worker(task_state: dict) -> dict:
        return {"results": [f"completed_{task_state.get('task_name')}"]}

    def aggregator(state: dict) -> dict:
        res = state.get("results", [])
        return {"summary": f"Aggregated {len(res)} tasks"}

    wf = Workflow("dynamic_fan_out", state_schema=MapReduceState)
    wf.then(init_node)
    wf.fan_out(partition_fn=partition_tasks, worker_node=worker, fan_in=aggregator, from_node="init_node")

    app = Application("fan_out_app")
    app.register(wf)
    res = app.run({})

    assert len(res["results"]) == 3
    assert "completed_task1" in res["results"]
    assert res["summary"] == "Aggregated 3 tasks"


def test_raw_langgraph_node_escape_hatch():
    """Verify that an advanced user can inject raw LangGraph node functions expecting (state, config)."""
    from langchain_core.runnables import RunnableConfig

    def raw_langgraph_node(state: dict, config: RunnableConfig) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "none")
        return {"output": f"Raw node executed on thread: {thread_id}"}

    wf = Workflow("raw_escape", state_schema=FlowState)
    wf.add_raw_node("raw_step", raw_langgraph_node)

    app = Application("escape_app")
    app.register(wf)
    res = app.run({}, thread_id="special-thread-42")

    assert res["output"] == "Raw node executed on thread: special-thread-42"
