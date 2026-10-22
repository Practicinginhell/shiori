"""Unit tests for Multi-Agent patterns (Supervisor, Sequential, Scatter-Gather)."""

from graphflow import Agent, Application, MockModel
from graphflow.multiagent import (
    create_scatter_gather,
    create_sequential_pipeline,
    create_supervisor,
)


def test_sequential_multiagent_pipeline():
    agent_a = Agent(name="agent_a", model=MockModel(responses=["Analysis from A"]))
    agent_b = Agent(name="agent_b", model=MockModel(responses=["Synthesis from B based on A"]))

    wf = create_sequential_pipeline("pipeline", agents=[agent_a, agent_b])
    app = Application("seq_ma_app")
    app.register(wf)

    res = app.run({"query": "Initial task"})
    assert res["output"] == "Synthesis from B based on A"
    assert [m["content"] for m in res["messages"]] == ["Initial task", "Analysis from A", "Synthesis from B based on A"]


def test_scatter_gather_parallel_agents():
    agent_1 = Agent(name="worker_1", model=MockModel(responses=["Result 1"]))
    agent_2 = Agent(name="worker_2", model=MockModel(responses=["Result 2"]))

    wf = create_scatter_gather("scatter_gather", agents=[agent_1, agent_2])
    app = Application("scatter_app")
    app.register(wf)

    res = app.run({"query": "Parallel task"})
    assert len(res["agent_results"]) == 2
    assert "Result 1" in res["final_output"]
    assert "Result 2" in res["final_output"]


def test_supervisor_pattern():
    worker = Agent(name="coder", model=MockModel(responses=["Code generated"]))
    supervisor_model = MockModel(
        responses=[
            '{"next": "coder", "instruction": "Write code"}',
            '{"next": "FINISH", "instruction": "Done"}',
        ]
    )

    wf = create_supervisor("sup_team", model=supervisor_model, agents=[worker])
    app = Application("supervisor_app")
    app.register(wf)

    res = app.run({"query": "Write code"})
    assert res["next_agent"] == "FINISH"
    assert res["output"] == "Code generated"
