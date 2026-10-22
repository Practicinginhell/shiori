"""Multi-agent Supervisor orchestration pattern."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from graphflow.core.agent import Agent
from graphflow.core.state import Field, State
from graphflow.core.workflow import Workflow
from graphflow.models.base import BaseModelAdapter

logger = logging.getLogger("graphflow.multiagent")

FINISH = "FINISH"
_DELEGATION_PREFIX = "Supervisor -> "


class SupervisorState(State):
    """Default state schema for Supervisor multi-agent workflows."""
    query: str = ""
    messages: list[dict[str, Any]] = Field.reducer("append")
    next_agent: str = ""
    supervisor_rounds: int = Field.reducer("sum")
    final_response: str = ""
    output: str = ""


def _parse_decision(content: str, agent_names: list[str]) -> tuple[str, str]:
    """Extracts (next_agent, instruction) from the supervisor's reply."""
    clean = content.strip()
    if "```" in clean:
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        clean = clean.removeprefix("json").strip()
    try:
        parsed = json.loads(clean)
        return str(parsed.get("next", FINISH)), str(parsed.get("instruction", ""))
    except (json.JSONDecodeError, AttributeError):
        lowered = content.lower()
        for name in agent_names:
            if name.lower() in lowered:
                return name, ""
        return FINISH, ""


def create_supervisor(
    name: str = "supervisor",
    model: BaseModelAdapter | str = "gpt-4o-mini",
    agents: Sequence[Agent] = (),
    instructions: str = "",
    max_rounds: int = 10,
    state_schema: type[State] | None = None,
) -> Workflow:
    """Creates a multi-agent Supervisor workflow.

    The supervisor reads the conversation, delegates the next step to one agent
    (passing it an instruction), or answers FINISH. After `max_rounds`
    delegations it finishes regardless; a custom `state_schema` must declare
    ``supervisor_rounds: int = Field.reducer("sum")`` for that limit to apply.
    """
    agent_map = {a.name: a for a in agents}
    agent_names = list(agent_map)

    supervisor_agent = Agent(
        name=f"{name}_controller",
        model=model,
        instructions=instructions or (
            f"You are a supervisor managing these specialized agents: {', '.join(agent_names)}.\n"
            "Analyze the user request and conversation history.\n"
            'Respond ONLY with a JSON object: {"next": "<agent_name_or_FINISH>", "instruction": "<instructions_for_agent>"}\n'
            f"Valid choices for 'next' are: {', '.join(agent_names)} or '{FINISH}'."
        ),
    )

    def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
        rounds = state.get("supervisor_rounds") or 0
        if rounds >= max_rounds:
            logger.warning("Supervisor '%s' reached max_rounds=%d; finishing.", name, max_rounds)
            return {"next_agent": FINISH}

        resp = supervisor_agent.execute(state)
        chosen, instruction = _parse_decision(resp.get("response", ""), agent_names)
        if chosen not in agent_map:
            chosen = FINISH

        updates: dict[str, Any] = {"next_agent": chosen}
        if chosen != FINISH:
            updates["supervisor_rounds"] = 1
            note = f"{_DELEGATION_PREFIX}{chosen}: {instruction}" if instruction else f"{_DELEGATION_PREFIX}{chosen}"
            updates["messages"] = [{"role": "user", "content": note}]
        return updates

    def route_supervisor(state: dict[str, Any]) -> str:
        return state.get("next_agent") or FINISH

    def finish_node(state: dict[str, Any]) -> dict[str, Any]:
        last_msg = ""
        for m in reversed(state.get("messages") or []):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
            content = m.get("content", "") if isinstance(m, dict) else str(getattr(m, "content", m))
            if role in ("assistant", "ai") and content:
                last_msg = content
                break
        return {"final_response": last_msg, "output": last_msg}

    wf = Workflow(name=name, state_schema=state_schema or SupervisorState)
    wf.add_node("supervisor", supervisor_node)
    for agent_name, agent in agent_map.items():
        wf.add_node(agent_name, agent)
    wf.add_node(FINISH, finish_node)

    routes: dict[str, str] = {an: an for an in agent_names}
    routes[FINISH] = FINISH
    wf.branch(condition=route_supervisor, routes=routes, from_node="supervisor")

    # Every worker agent reports back to the supervisor.
    for agent_name in agent_names:
        wf.add_raw_edge(source=agent_name, target="supervisor")

    return wf
