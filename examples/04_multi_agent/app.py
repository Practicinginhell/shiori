"""Example 4: Multi-Agent System (Supervisor Pattern).

Demonstrates:
- Supervisor multi-agent orchestration pattern
- Specialized autonomous worker agents (Researcher, Coder, Reviewer)
- Dynamic task assignment and convergence to final completion
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from graphflow import Agent, Application, MockModel
from graphflow.multiagent import create_supervisor


def build_multi_agent_app() -> Application:
    # 1. Specialized Worker Agents
    researcher_model = MockModel(
        responses=[
            "Research finding: LangGraph checkpointing enables time-travel and conversational branching."
        ]
    )
    researcher = Agent(
        name="researcher",
        model=researcher_model,
        instructions="Research architectural patterns and state management requirements.",
    )

    coder_model = MockModel(
        responses=[
            "Code solution:\n```python\nfrom langgraph.checkpoint.memory import MemorySaver\ncheckpointer = MemorySaver()\n```"
        ]
    )
    coder = Agent(
        name="coder",
        model=coder_model,
        instructions="Implement robust, clean Python code according to requirements.",
    )

    reviewer_model = MockModel(
        responses=[
            "Code review: Implementation is clean, adheres to LangGraph best practices, and includes typing."
        ]
    )
    reviewer = Agent(
        name="reviewer",
        model=reviewer_model,
        instructions="Review code and architecture for bugs, security, and standards.",
    )

    # 2. Supervisor Model orchestrating the workers in order: researcher -> coder -> reviewer -> FINISH
    supervisor_model = MockModel(
        responses=[
            '{"next": "researcher", "instruction": "Find best checkpointing pattern."}',
            '{"next": "coder", "instruction": "Implement the checkpointer setup."}',
            '{"next": "reviewer", "instruction": "Verify the checkpointer implementation."}',
            '{"next": "FINISH", "instruction": "All tasks completed successfully."}',
        ]
    )

    # 3. Create Supervisor Workflow
    workflow = create_supervisor(
        name="software_team",
        model=supervisor_model,
        agents=[researcher, coder, reviewer],
    )

    app = Application("multi_agent_application")
    app.register(workflow)
    return app


if __name__ == "__main__":
    app = build_multi_agent_app()
    user_goal = "Build a stateful checkpointing module for LangGraph"
    print(f"Goal: {user_goal}\n")

    result = app.run({"query": user_goal})
    print("=== Execution Complete ===")
    print("Messages Log:")
    for msg in result.get("messages", []):
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        print(" -", content[:100] + ("..." if len(content) > 100 else ""))

    print("\nFinal Result:")
    print(result.get("output"))
