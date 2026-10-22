"""Example 3: Autonomous Research Agent with Reflection/Review Loop.

Demonstrates:
- Multi-step structured pipeline: Planner -> Search -> Analyze -> Writer -> Reviewer
- Looping reflection cycle: Reviewer sends draft back to Writer if revision needed
- Typed state with append reducers and iteration counters
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from graphflow import Application, Field, State, Workflow


class ResearchState(State):
    """State schema for research workflow."""
    topic: str = ""
    plan: list[str] = Field.reducer("append")
    findings: list[str] = Field.reducer("append")
    draft: str = ""
    feedback: str = ""
    iteration: int = 0
    is_approved: bool = False
    output: str = ""


def plan_node(state: dict) -> dict:
    topic = state.get("topic", "AI")
    return {
        "plan": [
            f"1. Investigate recent breakthroughs in {topic}",
            "2. Analyze industrial applications and bottlenecks",
            "3. Synthesize future outlook",
        ]
    }


def search_node(state: dict) -> dict:
    return {
        "findings": [
            "Finding A: Autonomous multi-agent coordination increases workflow throughput by 40%.",
            "Finding B: Deterministic checkpointing solves reproducibility challenges in distributed agent clusters.",
        ]
    }


def analyze_node(state: dict) -> dict:
    findings = state.get("findings", [])
    return {
        "findings": [f"Analysis: Identified {len(findings)} core pillars of resilient AI architectures."]
    }


def writer_node(state: dict) -> dict:
    iteration = state.get("iteration", 0) + 1
    topic = state.get("topic", "")
    feedback = state.get("feedback", "")
    rev_note = f" (Incorporating revision notes: {feedback})" if feedback else ""

    draft = (
        f"# Research Report on {topic} (v{iteration}){rev_note}\n\n"
        "Recent advances demonstrate that agent orchestration layered atop directed acyclic graphs "
        "enables fault-tolerant, verifiable automated execution at scale."
    )
    return {"draft": draft, "iteration": iteration}


def reviewer_node(state: dict) -> dict:
    iteration = state.get("iteration", 1)
    # Require 1 revision to demonstrate reflection cycle
    if iteration < 2:
        return {
            "feedback": "Needs stronger emphasis on fault-tolerance mechanisms.",
            "is_approved": False,
        }
    return {
        "feedback": "Excellent quality. Meets all publication criteria.",
        "is_approved": True,
        "output": state.get("draft", ""),
    }


def should_continue_review(state: dict) -> str:
    """Route back to writer if unapproved, or finish."""
    if state.get("is_approved", False) or state.get("iteration", 0) >= 3:
        return "FINISH"
    return "writer"


def build_research_app() -> Application:
    wf = Workflow("research_agent", state_schema=ResearchState)

    wf.then(plan_node)
    wf.then(search_node)
    wf.then(analyze_node)
    wf.then(writer_node)
    wf.then(reviewer_node)

    # Branch back to writer if unapproved
    wf.branch(
        condition=should_continue_review,
        routes={"writer": writer_node, "FINISH": "FINISH"},
        from_node="reviewer_node",
    )

    # Finish node
    def finish_node(state: dict) -> dict:
        return {"output": state.get("draft", "")}

    wf.add_node("FINISH", finish_node)

    app = Application("research_pipeline")
    app.register(wf)
    return app


if __name__ == "__main__":
    app = build_research_app()
    topic = "LangGraph-based Orchestration Frameworks"
    print(f"Starting research on: {topic}...\n")

    result = app.run({"topic": topic})

    print(f"Total Iterations: {result.get('iteration')}")
    print(f"Reviewer Feedback: {result.get('feedback')}")
    print("\nFinal Approved Report:")
    print(result.get("output"))
