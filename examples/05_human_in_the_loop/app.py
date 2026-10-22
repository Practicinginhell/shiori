"""Example 5: Human-In-The-Loop (HITL) Approval and Execution Workflow.

Demonstrates:
- Interrupting workflow execution before sensitive action
- Emitting HumanApprovalRequired events with execution snapshot
- Resuming execution with human decision and modifications
- Final execution of approved proposal
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from graphflow import (
    Application,
    ApprovalDecision,
    HumanApprovalRequired,
    InMemoryCheckpointStore,
    State,
    Workflow,
)


class ProposalState(State):
    """HITL proposal and execution state."""
    action_type: str = ""
    amount: float = 0.0
    recipient: str = ""
    proposal_summary: str = ""
    execution_status: str = "pending"
    output: str = ""


def generate_proposal_node(state: dict) -> dict:
    """Agent prepares a sensitive proposal."""
    action = state.get("action_type", "transfer")
    amount = state.get("amount", 1000.0)
    recipient = state.get("recipient", "Ops-Wallet")

    summary = f"Authorize {action} of ${amount:,.2f} to {recipient}."
    return {"proposal_summary": summary}


def execute_action_node(state: dict) -> dict:
    """Executes the approved proposal."""
    summary = state.get("proposal_summary", "")
    amount = state.get("amount", 0.0)
    status_msg = f"EXECUTED: {summary} (Confirmed amount: ${amount:,.2f})"
    return {
        "execution_status": "completed",
        "output": status_msg,
    }


def build_hitl_app() -> Application:
    wf = Workflow("hitl_approval_workflow", state_schema=ProposalState)

    wf.then(generate_proposal_node)

    # Require human approval before execution
    wf.require_approval(
        before=execute_action_node,
        prompt="Please verify and approve the proposed transaction.",
        description="High-value transfer requires authorized sign-off.",
    )

    wf.then(execute_action_node)

    app = Application(
        name="hitl_app",
        checkpointer=InMemoryCheckpointStore(),  # Checkpointer required for pause/resume
    )
    app.register(wf)
    return app


if __name__ == "__main__":
    app = build_hitl_app()
    thread_id = "tx-approval-7788"

    print("=== Step 1: Starting Workflow (Execution will pause for approval) ===")
    initial_input = {
        "action_type": "Wire Transfer",
        "amount": 25000.0,
        "recipient": "Vendor-Global-Corp",
    }

    # Stream events until pause
    interrupted = False
    for event in app.stream(initial_input, thread_id=thread_id):
        if isinstance(event, HumanApprovalRequired):
            interrupted = True
            print("\n[INTERRUPT DETECTED]")
            print(f"Prompt: {event.prompt}")
            print(f"Node: {event.node_name}")
            print(f"Thread ID: {event.thread_id}")
            print(f"Payload: {event.payload}")

    if interrupted:
        print("\n=== Step 2: Human Reviews and Resumes with Modification ===")
        # Human approves but reduces the authorized amount to $20,000
        decision = ApprovalDecision(
            approved=True,
            feedback="Approved with reduced risk ceiling of $20,000.",
            edits={"amount": 20000.0},
        )
        print(f"Submitting Human Decision: approved={decision.approved}, edits={decision.edits}")

        final_result = app.resume(thread_id=thread_id, decision=decision)
        print("\n=== Step 3: Resumed Execution Completed ===")
        print("Final Output:", final_result.get("output"))
        print("Execution Status:", final_result.get("execution_status"))
        print("Final Amount:", final_result.get("amount"))
