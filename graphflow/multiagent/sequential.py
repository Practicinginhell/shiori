"""Sequential multi-agent pipeline: Agent A -> Agent B -> Agent C."""

from __future__ import annotations

from collections.abc import Sequence

from graphflow.core.agent import Agent
from graphflow.core.state import Field, State
from graphflow.core.workflow import Workflow


class PipelineState(State):
    """Default state schema for sequential agent pipelines."""
    query: str = ""
    messages: list = Field.reducer("append")
    output: str = ""


def create_sequential_pipeline(
    name: str = "sequential_pipeline",
    agents: Sequence[Agent] = (),
    state_schema: type[State] | None = None,
) -> Workflow:
    """Chains multiple agents into a sequential pipeline where output flows forward."""
    wf = Workflow(name=name, state_schema=state_schema or PipelineState)
    for agent in agents:
        wf.then(agent)
    return wf
