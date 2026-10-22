"""Example 6: Test-time communication (Park et al., 2026, arXiv:2609.21032) on graphflow.

    python examples/06_test_time_communication/app.py demo
        Offline, no API key: scripted agents in the paper's staged-improvement
        model (team@3 vs best@3), plus Proposition 1 exact values vs its bounds.

    python examples/06_test_time_communication/app.py packing --model openai:gpt-4.1 --k 3 --hours 3
        Real run on polyomino packing: team@k and k independent agents with the
        same per-agent budget. Needs the model provider's API key and a C++17 compiler.
        Runs in parallel on this machine; the paper isolates each trial in a container.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from graphflow import ModelResponse, ToolCall
from graphflow.models.mock import MockModel
from graphflow.ttc import Budget, CallableTask, Evaluation, PolyominoPackingTask, run_independent, run_team
from graphflow.ttc.theory import (
    best_success_upper_bound,
    exact_best_success,
    exact_team_success,
    team_success_lower_bound,
)

# --------------------------------------------------------------------------- #
# Offline demo
# --------------------------------------------------------------------------- #

def staged_task() -> CallableTask:
    def score(path: Path) -> Evaluation:
        try:
            return Evaluation(score=float(path.read_text().strip()), valid=True)
        except ValueError:
            return Evaluation(score=None, valid=False, status="rejected: not a number")

    return CallableTask("staged", "Submit a file with the highest stage you can verifiably reach.", score)


def staged_agent(agent: int, k: int):
    """Agent i can only find the improvement out of stage s when s % k == i (Figure 12)."""

    def call(name, **args):
        return ModelResponse(tool_calls=[ToolCall(id=name, name=name, args=args)], usage={"completion_tokens": 20})

    def decide(current: int):
        if current % k == agent:
            return call("write_file", path="c.txt", content=str(current + 1))
        return ModelResponse(content="no improvement found", usage={"completion_tokens": 20})

    def policy(messages):
        time.sleep(0.005)  # simulated model latency
        team = "SHARED WORKSPACE TOOLS" in messages[0]["content"]
        tools = [m for m in messages if m.get("role") == "tool"]
        last = messages[-1]
        own = [int(float(x)) for m in tools if m.get("name") == "submit" for x in re.findall(r"score=([0-9.]+)", m["content"])]
        if team and not any(m.get("name") == "claim_slot" for m in tools):
            return call("claim_slot", approach=f"specialist for stages = {agent} mod {k}")
        if last.get("role") == "user" or last.get("name") == "claim_slot":
            return call("read_shared", which="findings") if team else decide(max(own + [0]))
        if last.get("name") == "read_shared":
            shared = [int(float(x)) for x in re.findall(r"verified best=([0-9.]+)", last["content"])]
            return decide(max(shared + own + [0]))
        if last.get("name") == "write_file":
            return call("submit", path="c.txt")
        if last.get("name") == "submit" and team:
            return call("append_finding", text="verified best=" + re.findall(r"score=([0-9.]+)", last["content"])[-1])
        return ModelResponse(content="turn done", usage={"completion_tokens": 20})

    return policy


def demo() -> None:
    k, budget = 3, Budget(seconds=2.0)
    with tempfile.TemporaryDirectory() as d:
        team = run_team(staged_task(), lambda i: MockModel(staged_agent(i, k)), k, budget, Path(d) / "team")
        solo = run_independent(staged_task(), lambda i: MockModel(staged_agent(i, k)), k, budget, Path(d) / "solo")
    print(f"Staged task, {budget.seconds:g}s per agent:")
    print(f"  best@{k} (independent): stage {solo.best.score:g}")
    print(f"  team@{k} (communicating): stage {team.best.score:g}")
    print(f"  team trajectory (seconds, stage): {[(round(t, 2), s) for t, s in team.trajectory()][:6]} ...")

    print("\nProposition 1 (lambda = 1, alpha = 0.5): exact probabilities vs the paper's bounds")
    print(f"  {'m':>3} {'k':>2}  {'Pr(team)':>9} {'>= bound':>9}  {'Pr(best)':>9} {'<= bound':>9}")
    for m in (4, 8, 16, 32):
        for kk in (3, 5):
            print(
                f"  {m:>3} {kk:>2}  {exact_team_success(m, kk, 1, 0.5):9.4f} {team_success_lower_bound(m, kk, 0.5):9.4f}"
                f"  {exact_best_success(m, kk, 1, 0.5):9.4f} {best_success_upper_bound(m, kk, 0.5):9.4f}"
            )


# --------------------------------------------------------------------------- #
# Real run
# --------------------------------------------------------------------------- #

def packing(model: str, k: int, hours: float, cases: int, out: Path) -> None:
    task = PolyominoPackingTask(num_cases=cases)
    budget = Budget(seconds=hours * 3600)
    team = run_team(task, model, k, budget, out / "team")
    solo = run_independent(task, model, k, budget, out / "independent")
    summary = {
        "task": task.name,
        "k": k,
        "hours": hours,
        "team_best": team.best.score if team.best else None,
        "best_at_k": solo.best.score if solo.best else None,
        "team_trajectory_time": team.trajectory("time"),
        "team_trajectory_tokens": team.trajectory("tokens"),
        "best_at_k_trajectory_time": solo.trajectory("time"),
        "best_at_k_trajectory_tokens": solo.trajectory("tokens"),
        "team_agents": team.agent_reports,
        "independent_agents": solo.agent_reports,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"team@{k}: {summary['team_best']}   best@{k}: {summary['best_at_k']}   (details in {out / 'summary.json'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    p = sub.add_parser("packing")
    p.add_argument("--model", required=True, help='e.g. "openai:gpt-4.1" or "anthropic:claude-sonnet-4-6"')
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--cases", type=int, default=70)
    p.add_argument("--out", type=Path, default=Path("runs/ttc-packing"))
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    else:
        packing(args.model, args.k, args.hours, args.cases, args.out)
