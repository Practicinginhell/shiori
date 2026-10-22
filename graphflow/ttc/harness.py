"""team@k vs best@k harness, built on graphflow.

Both settings compile to the same graphflow Workflow::

    prepare --parallel--> agent_0 ... agent_{k-1} --> collect

- team@k:        one trial container; every agent receives the task instruction
                 plus the paper's communication prompt and shares the workspace.
- independent:   k trial containers, one agent each, task instruction only;
                 best@k is the best outcome among them.

Each ``agent_i`` node runs a graphflow ``Agent`` turn after turn in its own
private context until the budget (time, turns, output tokens) runs out, like
the paper's CLI agents that keep working until time runs out. The parallel
branches execute concurrently; a crashed worker is isolated into ``errors``
while its teammates continue (the paper terminates an agent that exhausts its
budget and lets the rest continue).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphflow.core.agent import Agent
from graphflow.core.application import Application
from graphflow.core.state import Field, State
from graphflow.core.workflow import Workflow
from graphflow.models.base import BaseModelAdapter
from graphflow.models.registry import resolve_model
from graphflow.ttc.ledger import LedgerEntry, SubmissionLedger
from graphflow.ttc.metering import MeteredModel, TokenMeter
from graphflow.ttc.metrics import best_so_far, independent_best_trajectory, on_token_axis
from graphflow.ttc.protocol import SharedWorkspace, render_communication_prompt
from graphflow.ttc.tasks.base import Task
from graphflow.ttc.tools import AgentContext, build_tools, tool_guide

ModelSpec = Any  # a model spec string, a BaseModelAdapter / LangChain chat model, or index -> either


@dataclass
class Budget:
    """Per-agent resources. At least one limit is required (agents otherwise never stop)."""
    seconds: float | None = None
    max_turns: int | None = None
    output_tokens_per_agent: int | None = None
    max_tool_calls_per_turn: int = 40

    def __post_init__(self) -> None:
        if self.seconds is None and self.max_turns is None and self.output_tokens_per_agent is None:
            raise ValueError("Budget needs at least one of seconds, max_turns or output_tokens_per_agent")


class TTCState(State):
    started_at: float = 0.0
    deadline: float = 0.0  # 0 = no wall-clock limit
    agent_reports: list = Field.reducer("append")
    errors: list = Field.reducer("append")
    output: str = ""


@dataclass
class RunResult:
    """Outcome of a team@k or independent (best@k) run."""
    mode: str  # "team" | "independent"
    k: int
    task_name: str
    higher_is_better: bool
    trial_dirs: list[Path]
    per_trial: list[list[LedgerEntry]]
    agent_reports: list[dict[str, Any]]
    token_timelines: list[list[tuple[float, int]]]
    errors: list[str] = field(default_factory=list)

    @property
    def ledger_entries(self) -> list[LedgerEntry]:
        return sorted((e for trial in self.per_trial for e in trial), key=lambda e: e.elapsed)

    @property
    def best(self) -> LedgerEntry | None:
        valid = [e for e in self.ledger_entries if e.valid and e.score is not None]
        if not valid:
            return None
        pick = max if self.higher_is_better else min
        return pick(valid, key=lambda e: e.score)

    def trajectory(self, axis: str = "time") -> list[tuple[float, float]]:
        """Best-so-far score against elapsed seconds or total output tokens of all agents."""
        if self.mode == "team":
            curve = best_so_far(((e.elapsed, e.score) for e in self.ledger_entries), self.higher_is_better)
        else:
            runs = [[(e.elapsed, e.score) for e in trial] for trial in self.per_trial]
            curve = independent_best_trajectory(runs, self.higher_is_better)
        if axis == "time":
            return curve
        if axis == "tokens":
            return on_token_axis(curve, self.token_timelines)
        raise ValueError("axis must be 'time' or 'tokens'")


def _model_for(model: ModelSpec, index: int) -> BaseModelAdapter:
    if isinstance(model, (str, BaseModelAdapter)) or hasattr(model, "invoke"):
        spec = model.clone() if isinstance(model, BaseModelAdapter) else model
    elif callable(model):
        spec = model(index)
    else:
        spec = model
    return resolve_model(spec, owner=f"agent_{index}")


def _instructions(task: Task, team: bool, k: int, workspace: SharedWorkspace | None, allow_shell: bool) -> str:
    parts = [task.instruction().strip()]
    if team:
        assert workspace is not None
        parts.append(render_communication_prompt(k, workspace).strip())
    parts.append(tool_guide(team, allow_shell))
    return "\n\n".join(parts)


def _turn_message(ctx: AgentContext, turn: int) -> str:
    remaining = "no time limit" if ctx.deadline is None else f"{max(0.0, ctx.deadline - time.time()):.0f}s remaining"
    turns = "" if ctx.turns_left is None else f", {ctx.turns_left} turn(s) remaining"
    if turn == 0:
        return f"[harness] Start working on the task now ({remaining}{turns})."
    return f"[harness] Continue working ({remaining}{turns})."


class _AgentWorker:
    """A graphflow node that runs one agent until its budget is spent."""

    def __init__(self, ctx: AgentContext, agent: Agent, budget: Budget, trial: int):
        self.ctx = ctx
        self.agent = agent
        self.budget = budget
        self.trial = trial
        self.__name__ = f"agent_{ctx.agent}"

    def _start(self, state: dict[str, Any]) -> None:
        self.ctx.started_at = state.get("started_at") or time.time()
        self.ctx.meter.started_at = self.ctx.started_at
        self.ctx.deadline = state.get("deadline") or None

    def _stop_reason(self, turn: int) -> str | None:
        if self.ctx.time_is_up():
            return "time"
        if self.ctx.meter.exhausted:
            return "token budget"
        if self.budget.max_turns is not None and turn >= self.budget.max_turns:
            return "turns"
        return None

    def _report(self, turns: int, stopped: str) -> dict[str, Any]:
        mine = [e for e in self.ctx.ledger.entries() if e.agent == self.ctx.agent]
        valid = [e.score for e in mine if e.valid and e.score is not None]
        best = (max(valid) if self.ctx.task.higher_is_better else min(valid)) if valid else None
        return {
            "agent": self.ctx.agent,
            "trial": self.trial,
            "slot": self.ctx.slot,
            "turns": turns,
            "output_tokens": self.ctx.meter.output_tokens,
            "submissions": len(mine),
            "best": best,
            "stopped": stopped,
        }

    def _prepare_turn(self, history: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        if self.budget.max_turns is not None:
            self.ctx.turns_left = self.budget.max_turns - turn
        message = _turn_message(self.ctx, turn)
        history.append({"role": "user", "content": message})
        return {"messages": list(history), "input": message}

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        self._start(state)
        history: list[dict[str, Any]] = []
        turn = 0
        while (stopped := self._stop_reason(turn)) is None:
            update = self.agent.execute(self._prepare_turn(history, turn))
            history.extend(update.get("messages", []))
            turn += 1
        return {"agent_reports": [self._report(turn, stopped)]}

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        self._start(state)
        history: list[dict[str, Any]] = []
        turn = 0
        while (stopped := self._stop_reason(turn)) is None:
            update = await self.agent.aexecute(self._prepare_turn(history, turn))
            history.extend(update.get("messages", []))
            turn += 1
        return {"agent_reports": [self._report(turn, stopped)]}


@dataclass
class _Trial:
    index: int
    container: Path
    harness_dir: Path
    ledger: SubmissionLedger
    workspace: SharedWorkspace | None


def build_workflow(
    task: Task,
    model: ModelSpec,
    k: int,
    budget: Budget,
    workdir: Path,
    team: bool,
    allow_shell: bool = False,
) -> tuple[Workflow, list[_Trial], list[TokenMeter]]:
    """Builds the graphflow Workflow for a team@k (team=True) or independent (team=False) run."""
    if k < 1:
        raise ValueError("k must be >= 1")
    workdir = Path(workdir)
    n_trials = 1 if team else k
    trials = []
    for t in range(n_trials):
        root = workdir / f"trial-{t}"
        container = root / "container"
        harness_dir = root / "harness"  # outside the agent container: ledger + submission snapshots
        ledger = SubmissionLedger(harness_dir / "ledger.jsonl", task.higher_is_better)
        trials.append(_Trial(t, container, harness_dir, ledger, SharedWorkspace(container) if team else None))

    workers: list[_AgentWorker] = []
    meters: list[TokenMeter] = []
    for i in range(k):
        trial = trials[0] if team else trials[i]
        meter = TokenMeter(budget.output_tokens_per_agent)
        ctx = AgentContext(
            agent=i,
            team_size=k if team else 1,
            task=task,
            container=trial.container,
            harness_dir=trial.harness_dir,
            ledger=trial.ledger,
            meter=meter,
            workspace=trial.workspace,
        )
        agent = Agent(
            name=f"agent_{i}",
            model=MeteredModel(_model_for(model, i), meter),
            instructions=_instructions(task, team, k, trial.workspace, allow_shell),
            tools=build_tools(ctx, allow_shell=allow_shell),
            max_tool_iterations=budget.max_tool_calls_per_turn,
        )
        workers.append(_AgentWorker(ctx, agent, budget, trial.index))
        meters.append(meter)

    def prepare(state: dict[str, Any]) -> dict[str, Any]:
        for trial in trials:
            task.setup(trial.container)
            (trial.container / "work").mkdir(parents=True, exist_ok=True)
            trial.harness_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        return {"started_at": started, "deadline": started + budget.seconds if budget.seconds else 0.0}

    def collect(state: dict[str, Any]) -> dict[str, Any]:
        best = [trial.ledger.best() for trial in trials]
        scores = [b.score for b in best if b is not None]
        if not scores:
            return {"output": "no valid submission"}
        top = max(scores) if task.higher_is_better else min(scores)
        return {"output": f"best score {top}"}

    wf = Workflow(f"{'team' if team else 'independent'}@{k}", state_schema=TTCState)
    wf.then(prepare)
    for worker in workers:
        wf.add_node(worker.__name__, worker)
    wf.parallel([w.__name__ for w in workers], fan_in=collect, from_node="prepare")
    return wf, trials, meters


def _result(team: bool, task: Task, k: int, trials: list[_Trial], meters: list[TokenMeter], state: dict[str, Any]) -> RunResult:
    return RunResult(
        mode="team" if team else "independent",
        k=k,
        task_name=task.name,
        higher_is_better=task.higher_is_better,
        trial_dirs=[t.container for t in trials],
        per_trial=[t.ledger.entries() for t in trials],
        agent_reports=sorted(state.get("agent_reports", []), key=lambda r: r["agent"]),
        token_timelines=[list(m.timeline) for m in meters],
        errors=list(state.get("errors", [])),
    )


def _app(wf: Workflow, listeners: Sequence[Callable[..., Any]] | None) -> Application:
    return Application(wf.name, listeners=list(listeners or [])).register(wf)


def run_team(
    task: Task,
    model: ModelSpec,
    k: int,
    budget: Budget,
    workdir: str | Path,
    allow_shell: bool = False,
    listeners: Sequence[Callable[..., Any]] | None = None,
) -> RunResult:
    """Runs team@k: k communicating agents in one shared container."""
    wf, trials, meters = build_workflow(task, model, k, budget, Path(workdir), team=True, allow_shell=allow_shell)
    return _result(True, task, k, trials, meters, _app(wf, listeners).run({}))


def run_independent(
    task: Task,
    model: ModelSpec,
    k: int,
    budget: Budget,
    workdir: str | Path,
    allow_shell: bool = False,
    listeners: Sequence[Callable[..., Any]] | None = None,
) -> RunResult:
    """Runs k independent agents in separate containers (the best@k baseline)."""
    wf, trials, meters = build_workflow(task, model, k, budget, Path(workdir), team=False, allow_shell=allow_shell)
    return _result(False, task, k, trials, meters, _app(wf, listeners).run({}))


async def arun_team(
    task: Task,
    model: ModelSpec,
    k: int,
    budget: Budget,
    workdir: str | Path,
    allow_shell: bool = False,
    listeners: Sequence[Callable[..., Any]] | None = None,
) -> RunResult:
    """Async team@k."""
    wf, trials, meters = build_workflow(task, model, k, budget, Path(workdir), team=True, allow_shell=allow_shell)
    return _result(True, task, k, trials, meters, await _app(wf, listeners).arun({}))


async def arun_independent(
    task: Task,
    model: ModelSpec,
    k: int,
    budget: Budget,
    workdir: str | Path,
    allow_shell: bool = False,
    listeners: Sequence[Callable[..., Any]] | None = None,
) -> RunResult:
    """Async independent (best@k) run."""
    wf, trials, meters = build_workflow(task, model, k, budget, Path(workdir), team=False, allow_shell=allow_shell)
    return _result(False, task, k, trials, meters, await _app(wf, listeners).arun({}))
