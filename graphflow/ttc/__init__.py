"""Test-time communication (TTC) for graphflow.

Implementation of "Scaling Discovery through Test-Time Communication"
(Park, Kontonis, Garg, Krishnamurthy, Papailiopoulos; arXiv:2609.21032, 2026):
k identical agents with no predefined roles share a workspace (atomic slot
claims, findings / disconfirmations / score logs, a self-authored coordination
file) under the paper's communication prompt, compared against the best of k
independent agents (best@k).

    from graphflow.ttc import Budget, PolyominoPackingTask, run_team, run_independent

    task = PolyominoPackingTask()
    team = run_team(task, model="openai:gpt-4.1", k=3, budget=Budget(seconds=3 * 3600), workdir="runs/team")
    solo = run_independent(task, model="openai:gpt-4.1", k=3, budget=Budget(seconds=3 * 3600), workdir="runs/solo")
    print(team.best.score, solo.best.score)
"""

from graphflow.ttc.harness import (
    Budget,
    RunResult,
    arun_independent,
    arun_team,
    build_workflow,
    run_independent,
    run_team,
)
from graphflow.ttc.ledger import LedgerEntry, SubmissionLedger
from graphflow.ttc.protocol import COMMUNICATION_PROMPT, SharedWorkspace, render_communication_prompt
from graphflow.ttc.tasks import CallableTask, Evaluation, PolyominoPackingTask, Task

__all__ = [
    "COMMUNICATION_PROMPT",
    "Budget",
    "CallableTask",
    "Evaluation",
    "LedgerEntry",
    "PolyominoPackingTask",
    "RunResult",
    "SharedWorkspace",
    "SubmissionLedger",
    "Task",
    "arun_independent",
    "arun_team",
    "build_workflow",
    "render_communication_prompt",
    "run_independent",
    "run_team",
]
