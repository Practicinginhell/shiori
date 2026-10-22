"""Test-time communication protocol (Park et al., 2026, Section 2 and Appendix A.2).

Agents share a workspace with:
  - numbered slot directories, claimed atomically (``mkdir``), each holding the
    owner's declared ``approach``;
  - an append-only findings log (the asynchronous broadcast channel);
  - a disconfirmations log (negative results);
  - a score log (``[slot S HH:MM:SS] score=<score> family=<approach>``);
  - a coordination file whose conventions the agents author themselves;
  - a private scratch directory ``work-S`` per claimed slot.

``COMMUNICATION_PROMPT`` is the paper's prompt verbatim; the harness fills in
the team size and these paths, so every agent receives identical text.

Paths are not specified in the paper (the authors' repository was empty at the
time of writing); the layout below under ``<container>/team`` is ours.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Appendix A.2, verbatim. Fields: {n}, {nm1} (= n - 1), {slots}, {findings},
# {disconfirm}, {plateau}, {coordination}, {base}.
COMMUNICATION_PROMPT = """\
{n} agents share this container and work the same task in parallel, all with this
identical prompt. Search widely without herding, coordinate as you go, and keep
improving until time runs out.
Shared scratch (create on first use):
- slots/approaches: {slots}/
- findings: {findings}
- disconfirmations: {disconfirm}
- score log: {plateau}
- coordination: {coordination} (empty -- conventions you author)
Your private scratch is {base}/work-$S after you claim slot $S.
1. CLAIM A SLOT AND PICK A DISTINCT APPROACH:
mkdir -p {slots}
for i in $(seq 0 {nm1}); do mkdir "{slots}/slot-$i" 2>/dev/null && S=$i && break; done
mkdir -p {base}/work-$S
touch {coordination}
echo "<your approach + what you will deliberately not assume>" > "{slots}/slot-$S/approach"
cat {slots}/slot-*/approach
If a lower-numbered slot already took your approach, change yours. Cover a
different part of the search space; do not agree early.
2. ALWAYS BE ACTING. Never end a turn with only prose or a plan. Every turn must
run a command that advances or tests the work: take a real action the task
accepts and read back its result. Publishing, disconfirming, coordinating, and
pivoting are bookkeeping around real actions, never a substitute for taking one.
3. EVALUATE AND RECORD EVERY ATTEMPT. Score each change with the task's own
scoring or feedback mechanism (find it in AGENT.md or the task instructions),
then log the result:
echo "[slot $S $(date -u +%H:%M:%S)] score=<score-or-progress> family=<approach>" >> {plateau}
4. SHARE WITHOUT HERDING. Append concise findings with evidence and cost to
{findings}, and label weak claims as weak. Spend part of your effort trying to
FALSIFY the leading idea or your own, recording negative results in
{disconfirm}. Do not write prose whose only purpose is to make peers copy you.
5. COORDINATE ON SHARED RESOURCES. You share the graded output, common files, and
the environment with peers who run blind to your session. Treat every shared
resource as contested: before you touch one, re-check {coordination} and the
resource's current state; after, confirm your change survived and did not just
repeat a peer's. A collision is any overwritten, duplicated, or conflicting work
that wastes effort. No coordination scheme is provided -- {coordination} is empty
and yours to author: on a collision, write a convention there concrete enough for
a peer to follow, that changes your next action, then follow it.
6. HIGH BAR TO CONVERGE. Keep your own approach unless another clearly beats it on
a measured, reproduced result, or yours is blocked, or the run is wrapping up.
Even then, keep one real difference (a parameter, subcase, representation, or
fallback) until the very end.
7. NEVER STOP WHILE TIME REMAINS. A working result is not the finish line; the
clock running out is the only acceptable reason to stop. Do not declare the
task done, final, solved, or "at the ceiling" and go idle -- a suspected
ceiling is a claim to disconfirm, not a reason to quit.
8. BREAK PLATEAUS BY CHANGING FAMILY. You are plateaued when your best score has
not strictly improved over 3 consecutive attempts. Then stop tuning
and switch to a STRUCTURALLY DIFFERENT approach -- a different core principle or
assumption, not a variant of the current one. Keep a short list of untried
families in {base}/work-$S so you always have a next one ready. Read peers'
approaches and {plateau} first and pick a family no active peer is on; adopting
a peer who is also plateaued is not progress.
Run a tight loop -- change -> evaluate -> record -> repeat -- without pausing. Do
not use the internet, curl, wget, HTTP libraries, or secrets.
"""

PLATEAU_PATIENCE = 3  # "not strictly improved over 3 consecutive attempts"

_SCORE_LINE = re.compile(r"^\[slot (\d+) (\d\d:\d\d:\d\d)\] score=(\S+) family=(.*)$")


def _utc_clock() -> str:
    """The ``date -u +%H:%M:%S`` timestamp used in the protocol's log lines."""
    return time.strftime("%H:%M:%S", time.gmtime())


@dataclass(frozen=True)
class ScoreEntry:
    slot: int
    time: str
    score: str  # "<score-or-progress>": kept as written
    family: str

    @property
    def numeric(self) -> float | None:
        try:
            return float(self.score)
        except ValueError:
            return None


class SharedWorkspace:
    """The shared scratch of one team trial, rooted in the trial's task container."""

    def __init__(self, container: str | Path):
        self.container = Path(container)
        self.base = self.container / "team"
        self.slots = self.base / "slots"
        self.findings = self.base / "findings.md"
        self.disconfirm = self.base / "disconfirmations.md"
        self.plateau = self.base / "score_log.txt"
        self.coordination = self.base / "coordination.md"
        self.locks = self.base / "locks"
        self._append_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def prompt_fields(self, n: int) -> dict[str, str | int]:
        return {
            "n": n,
            "nm1": n - 1,
            "slots": str(self.slots),
            "findings": str(self.findings),
            "disconfirm": str(self.disconfirm),
            "plateau": str(self.plateau),
            "coordination": str(self.coordination),
            "base": str(self.base),
        }

    # ------------------------------------------------------------------ #
    # Step 1: slots and approaches
    # ------------------------------------------------------------------ #

    def claim_slot(self, n: int) -> int:
        """Claims the lowest free slot among ``0..n-1`` with an atomic ``mkdir``."""
        self.slots.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            try:
                os.mkdir(self.slots / f"slot-{i}")
            except FileExistsError:
                continue
            self.private_dir(i).mkdir(parents=True, exist_ok=True)
            self.coordination.touch()
            return i
        raise RuntimeError(f"All {n} slots are already claimed")

    def private_dir(self, slot: int) -> Path:
        return self.base / f"work-{slot}"

    def set_approach(self, slot: int, approach: str) -> None:
        (self.slots / f"slot-{slot}" / "approach").write_text(approach.strip() + "\n")

    def approaches(self) -> dict[int, str]:
        result: dict[int, str] = {}
        if not self.slots.is_dir():
            return result
        for slot_dir in self.slots.glob("slot-*"):
            approach = slot_dir / "approach"
            if approach.is_file():
                result[int(slot_dir.name.split("-", 1)[1])] = approach.read_text().strip()
        return dict(sorted(result.items()))

    # ------------------------------------------------------------------ #
    # Steps 3-5: shared records
    # ------------------------------------------------------------------ #

    def _append(self, path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._append_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")

    def log_score(self, slot: int, score: object, family: str) -> None:
        """Step 3: ``[slot $S $(date -u +%H:%M:%S)] score=<score-or-progress> family=<approach>``."""
        self._append(self.plateau, f"[slot {slot} {_utc_clock()}] score={score} family={family}")

    def append_finding(self, slot: int, text: str) -> None:
        self._append(self.findings, f"[slot {slot} {_utc_clock()}] {text}")

    def append_disconfirmation(self, slot: int, text: str) -> None:
        self._append(self.disconfirm, f"[slot {slot} {_utc_clock()}] {text}")

    def append_coordination(self, slot: int, text: str) -> None:
        self._append(self.coordination, f"[slot {slot} {_utc_clock()}] {text}")

    def read(self, which: str) -> str:
        """Reads one shared record: findings, disconfirmations, score_log, coordination or approaches."""
        if which == "approaches":
            return "\n".join(f"slot-{s}: {a}" for s, a in self.approaches().items())
        paths = {
            "findings": self.findings,
            "disconfirmations": self.disconfirm,
            "score_log": self.plateau,
            "coordination": self.coordination,
        }
        if which not in paths:
            raise ValueError(f"Unknown shared record '{which}' (expected one of {sorted(paths) + ['approaches']})")
        path = paths[which]
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def score_log(self) -> list[ScoreEntry]:
        entries = []
        for line in self.read("score_log").splitlines():
            match = _SCORE_LINE.match(line.strip())
            if match:
                entries.append(ScoreEntry(int(match[1]), match[2], match[3], match[4]))
        return entries

    # ------------------------------------------------------------------ #
    # Step 8: plateau detection
    # ------------------------------------------------------------------ #

    def is_plateaued(self, slot: int, higher_is_better: bool = True, patience: int = PLATEAU_PATIENCE) -> bool:
        """True when the slot's best score has not strictly improved over `patience` consecutive attempts."""
        scores = [e.numeric for e in self.score_log() if e.slot == slot and e.numeric is not None]
        if not scores:
            return False
        best_index, best = 0, scores[0]
        for i, s in enumerate(scores[1:], start=1):
            if (s > best) if higher_is_better else (s < best):
                best_index, best = i, s
        return len(scores) - 1 - best_index >= patience

    # ------------------------------------------------------------------ #
    # Shared resources
    # ------------------------------------------------------------------ #

    @contextmanager
    def lock(self, name: str, timeout: float = 60.0, poll: float = 0.005) -> Iterator[None]:
        """A cross-process file lock for a shared artifact (e.g. the graded submission)."""
        self.locks.mkdir(parents=True, exist_ok=True)
        path = self.locks / f"{name}.lock"
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Timed out waiting for lock '{name}'") from None
                time.sleep(poll)
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)


def render_communication_prompt(n: int, workspace: SharedWorkspace) -> str:
    """The Appendix A.2 prompt with the team size and this workspace's paths filled in."""
    if n < 1:
        raise ValueError("team size must be >= 1")
    return COMMUNICATION_PROMPT.format(**workspace.prompt_fields(n))
