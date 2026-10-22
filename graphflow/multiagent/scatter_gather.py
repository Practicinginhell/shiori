"""Parallel multi-agent scatter-gather: Input -> [Agent A, Agent B, Agent C] -> Aggregator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from graphflow.core.agent import Agent
from graphflow.core.state import Field, State
from graphflow.core.workflow import Workflow


class ScatterGatherState(State):
    """Default state schema for parallel scatter-gather workflows."""
    query: str = ""
    agent_results: list[dict[str, Any]] = Field.reducer("append")
    final_output: str = ""
    output: str = ""
    errors: list = Field.reducer("append")


class _CollectAgentResult:
    """Runs an agent and records its answer into `agent_results` (sync and async)."""

    def __init__(self, agent: Agent):
        self.agent = agent

    def _record(self, res: dict[str, Any]) -> dict[str, Any]:
        text = res.get("response", res.get("output", ""))
        return {"agent_results": [{"agent": self.agent.name, "result": text}]}

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._record(self.agent.execute(state))

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._record(await self.agent.aexecute(state))


def _default_aggregator(state: dict[str, Any]) -> dict[str, Any]:
    results = state.get("agent_results", [])
    combined = "\n\n".join(f"[{r.get('agent', 'Agent')}]: {r.get('result', '')}" for r in results)
    return {"final_output": combined, "output": combined}


def create_scatter_gather(
    name: str = "scatter_gather",
    agents: Sequence[Agent] = (),
    aggregator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    state_schema: type[State] | None = None,
) -> Workflow:
    """Executes multiple agents concurrently and aggregates their outputs.

    A failing agent is recorded in ``errors`` and the others still complete.
    """
    wf = Workflow(name=name, state_schema=state_schema or ScatterGatherState)

    def prepare(state: dict[str, Any]) -> dict[str, Any]:
        return {}

    wf.then(prepare)
    branch_names = []
    for ag in agents:
        node_name = f"parallel_{ag.name}"
        wf.add_node(node_name, _CollectAgentResult(ag))
        branch_names.append(node_name)

    wf.add_node("aggregate", aggregator or _default_aggregator)
    wf.parallel(branches=branch_names, fan_in="aggregate", from_node="prepare")
    return wf
