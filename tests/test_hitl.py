"""Unit tests for Human-In-The-Loop (HITL) approval, pause, and resume."""

import pytest

from graphflow import (
    Application,
    ApprovalDecision,
    HumanApprovalRequired,
    InMemoryCheckpointStore,
    State,
    Workflow,
)
from graphflow.core.errors import ExecutionError


class ApprovalState(State):
    request: str = ""
    status: str = "pending"
    amount: int = 100
    output: str = ""


def prepare_tx(state: dict) -> dict:
    return {"request": "Transfer funds", "amount": 500}


def execute_tx(state: dict) -> dict:
    return {"status": "executed", "output": f"Sent ${state.get('amount')}"}


def test_hitl_approval_and_resume_with_modifications():
    wf = Workflow("hitl_test", state_schema=ApprovalState)
    wf.then(prepare_tx)
    wf.require_approval(before=execute_tx, prompt="Authorize transfer?")
    wf.then(execute_tx)

    app = Application("hitl_app", checkpointer=InMemoryCheckpointStore())
    app.register(wf)

    thread_id = "test-thread-101"

    # Step 1: Run until interrupt
    events = list(app.stream({"request": "start"}, thread_id=thread_id))
    interrupt_events = [e for e in events if isinstance(e, HumanApprovalRequired)]
    assert len(interrupt_events) == 1
    assert interrupt_events[0].prompt == "Authorize transfer?"

    # Step 2: Resume with edit
    decision = ApprovalDecision(approved=True, edits={"amount": 250})
    final = app.resume(thread_id=thread_id, decision=decision)

    assert final["status"] == "executed"
    assert final["amount"] == 250
    assert final["output"] == "Sent $250"


def test_hitl_rejection():
    wf = Workflow("hitl_reject_test", state_schema=ApprovalState)
    wf.then(prepare_tx)
    wf.require_approval(before=execute_tx, prompt="Authorize transfer?")
    wf.then(execute_tx)

    app = Application("hitl_reject_app", checkpointer=InMemoryCheckpointStore())
    app.register(wf)

    thread_id = "test-thread-102"
    list(app.stream({"request": "start"}, thread_id=thread_id))

    decision = ApprovalDecision(approved=False, feedback="Unauthorized transaction")
    with pytest.raises(ExecutionError) as exc_info:
        app.resume(thread_id=thread_id, decision=decision)
    assert "rejected" in str(exc_info.value).lower()
