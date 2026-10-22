"""Graphflow tools for one agent in a test-time communication trial.

The paper's agents are CLI agents with a shell in the task container. Here each
agent is a graphflow Agent whose tools are the protocol's shell steps as
structured calls (claim_slot, log_score, ...), the task's own verifier
(submit), and file access confined to the trial's container. With
``allow_shell=True`` a ``run_shell`` tool is added so the verbatim prompt's
commands can be run literally; only enable it inside an isolated container.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from graphflow.tools.base import Tool
from graphflow.ttc.ledger import SubmissionLedger
from graphflow.ttc.metering import TokenMeter
from graphflow.ttc.protocol import SharedWorkspace
from graphflow.ttc.tasks.base import Evaluation, Task

TIME_UP = "TIME IS UP: the run has ended. Stop now."
_MAX_OUTPUT = 8_000


@dataclass
class AgentContext:
    """Everything one agent's tools act on."""
    agent: int
    team_size: int
    task: Task
    container: Path
    harness_dir: Path
    ledger: SubmissionLedger
    meter: TokenMeter
    workspace: SharedWorkspace | None  # None for independent agents
    started_at: float = field(default_factory=time.time)
    deadline: float | None = None  # wall-clock time.time() deadline
    slot: int | None = None
    turns_left: int | None = None

    @property
    def cwd(self) -> Path:
        """Relative paths resolve here: the private scratch once it exists, else the container."""
        if self.workspace is not None:
            return self.workspace.private_dir(self.slot) if self.slot is not None else self.container
        return self.container / "work"

    def time_is_up(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def resolve(self, path: str) -> Path:
        p = Path(path)
        p = (p if p.is_absolute() else self.cwd / p).resolve()
        root = self.container.resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"'{path}' is outside the task container ({root})")
        return p


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT else text[:_MAX_OUTPUT] + f"\n... [{len(text) - _MAX_OUTPUT} chars truncated]"


def build_tools(ctx: AgentContext, allow_shell: bool = False) -> list[Tool]:
    """The tools for one agent (shared-workspace tools only in team trials)."""
    tools: list[Tool] = []

    def add(fn, name: str, description: str) -> None:
        tools.append(Tool(fn, name=name, description=description))

    # -------------------------------------------------------------- #
    # Files
    # -------------------------------------------------------------- #

    def write_file(path: str, content: str) -> str:
        target = ctx.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {target}"

    def read_file(path: str) -> str:
        return _truncate(ctx.resolve(path).read_text(encoding="utf-8"))

    def list_files(path: str = ".") -> str:
        root = ctx.resolve(path)
        if not root.is_dir():
            return f"{root} is not a directory"
        entries = sorted(str(p.relative_to(root)) + ("/" if p.is_dir() else "") for p in root.iterdir())
        return "\n".join(entries) or "(empty)"

    add(write_file, "write_file", "Write a text file. Relative paths are inside your private scratch.")
    add(read_file, "read_file", "Read a text file (relative to your private scratch, or an absolute container path).")
    add(list_files, "list_files", "List a directory (relative to your private scratch, or an absolute container path).")

    # -------------------------------------------------------------- #
    # The task's own verifier
    # -------------------------------------------------------------- #

    def submit(path: str) -> str:
        if ctx.time_is_up():
            return TIME_UP
        source = ctx.resolve(path)
        if not source.is_file():
            return f"Error: {source} does not exist"
        snapshots = ctx.harness_dir / "submissions"
        snapshots.mkdir(parents=True, exist_ok=True)
        snapshot = snapshots / f"{time.time_ns()}-agent{ctx.agent}-{uuid.uuid4().hex[:8]}{source.suffix}"
        shutil.copyfile(source, snapshot)  # score a snapshot: later edits can't change a recorded result
        try:
            evaluation = ctx.task.evaluate(snapshot)
        except Exception as e:  # noqa: BLE001 - a broken scorer run is a rejected submission, not a crashed agent
            evaluation = Evaluation(None, False, f"rejected: scorer error: {e}")
        entry = ctx.ledger.record(
            agent=ctx.agent,
            slot=ctx.slot,
            candidate=str(snapshot),
            evaluation=evaluation,
            elapsed=ctx.elapsed(),
            tokens=ctx.meter.output_tokens,
        )
        details = " ".join(f"{k}={v}" for k, v in evaluation.details.items())
        return (
            f"submission #{entry.index}: score={evaluation.score} valid={evaluation.valid} "
            f"status={evaluation.status} {details}".rstrip()
        )

    add(submit, "submit", "Score a candidate file with the task's own verifier. Every submission is recorded; "
                          "the best valid one counts.")

    def time_remaining() -> str:
        remaining = "unlimited" if ctx.deadline is None else f"{max(0.0, ctx.deadline - time.time()):.0f}s"
        turns = "unlimited" if ctx.turns_left is None else str(ctx.turns_left)
        return f"elapsed={ctx.elapsed():.0f}s remaining={remaining} turns_remaining={turns}"

    add(time_remaining, "time_remaining", "Time and turns left in this run.")

    # -------------------------------------------------------------- #
    # Shared workspace (team trials only)
    # -------------------------------------------------------------- #

    ws = ctx.workspace
    if ws is not None:
        def require_slot() -> int:
            if ctx.slot is None:
                raise ValueError("claim a slot first (claim_slot)")
            return ctx.slot

        def claim_slot(approach: str) -> str:
            if ctx.slot is None:
                ctx.slot = ws.claim_slot(ctx.team_size)
            ws.set_approach(ctx.slot, approach)
            listing = ws.read("approaches")
            return (
                f"You own slot {ctx.slot}. Private scratch: {ws.private_dir(ctx.slot)}\n"
                f"Claimed approaches:\n{listing}"
            )

        def read_shared(which: str) -> str:
            return _truncate(ws.read(which)) or "(empty)"

        def log_score(score: str, family: str) -> str:
            ws.log_score(require_slot(), score, family)
            plateau = ws.is_plateaued(ctx.slot, higher_is_better=ctx.task.higher_is_better)  # type: ignore[arg-type]
            return "logged" + (" -- you are PLATEAUED (rule 8): switch to a structurally different family" if plateau else "")

        def append_finding(text: str) -> str:
            ws.append_finding(require_slot(), text)
            return "appended to findings"

        def append_disconfirmation(text: str) -> str:
            ws.append_disconfirmation(require_slot(), text)
            return "appended to disconfirmations"

        def write_coordination(text: str) -> str:
            ws.append_coordination(require_slot(), text)
            return "appended to coordination"

        add(claim_slot, "claim_slot", "Step 1: atomically claim the lowest free slot (once), record your approach "
                                      "(or update it), and list every claimed approach.")
        add(read_shared, "read_shared", "Read a shared record: findings | disconfirmations | score_log | "
                                        "coordination | approaches.")
        add(log_score, "log_score", "Step 3: append '[slot S HH:MM:SS] score=<score> family=<approach>' to the score log.")
        add(append_finding, "append_finding", "Step 4: append a concise finding with evidence and cost.")
        add(append_disconfirmation, "append_disconfirmation", "Step 4: append a negative result.")
        add(write_coordination, "write_coordination", "Step 5: append a coordination convention.")

    # -------------------------------------------------------------- #
    # Shell (opt-in: run only inside an isolated container)
    # -------------------------------------------------------------- #

    if allow_shell:
        def run_shell(command: str) -> str:
            if ctx.time_is_up():
                return TIME_UP
            timeout = 600.0 if ctx.deadline is None else max(1.0, min(600.0, ctx.deadline - time.time()))
            env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(ctx.container)}
            if ctx.slot is not None:
                env["S"] = str(ctx.slot)
            try:
                proc = subprocess.run(  # noqa: S603 - opt-in agent shell (allow_shell=True), for isolated containers
                    ["bash", "-c", command], cwd=ctx.cwd if ctx.cwd.exists() else ctx.container,  # noqa: S607
                    capture_output=True, text=True, timeout=timeout, env=env, check=False,
                )
            except subprocess.TimeoutExpired:
                return f"command timed out after {timeout:.0f}s"
            return _truncate(f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}")

        add(run_shell, "run_shell", "Run a bash command in the task container.")

    return tools


def tool_guide(team: bool, allow_shell: bool) -> str:
    """Harness note appended after the paper's prompt, mapping its shell steps to the available tools."""
    if not team:
        return (
            "TOOLS (harness note): write_file / read_file / list_files (relative paths are in your private "
            "scratch), submit(path) runs the task's own scorer (every submission is recorded; the best valid "
            "one counts), time_remaining()."
            + (" run_shell(command) runs bash in the task container." if allow_shell else "")
        )
    shell_line = (
        "You also have run_shell(command), a bash shell in the task container: you may run the commands "
        "above literally.\n" if allow_shell else
        "There is no shell in this run; each tool below performs the corresponding shell step above.\n"
    )
    return (
        "SHARED WORKSPACE TOOLS (harness note):\n"
        + shell_line
        + "- claim_slot(approach): the step-1 mkdir loop + writing your approach + listing all approaches.\n"
        "- read_shared(which): findings | disconfirmations | score_log | coordination | approaches.\n"
        "- log_score(score, family): the step-3 score-log line for your slot.\n"
        "- append_finding(text) / append_disconfirmation(text) / write_coordination(text).\n"
        "- write_file / read_file / list_files: relative paths are in your private scratch.\n"
        "- submit(path): the task's own scorer (see AGENT.md). Every submission is recorded; the best valid "
        "one counts.\n"
        "- time_remaining(): time and turns left."
    )
