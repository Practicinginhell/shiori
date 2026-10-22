"""Task interface: an instruction, container setup, and the task's own verifier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Evaluation:
    """The verifier's verdict on one submission."""
    score: float | None
    valid: bool
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """A task with an agent-accessible verifier.

    The paper's central condition is *verified progress sharing*: agents must be
    able to score candidates themselves (``evaluate``), repeatedly, without
    seeing hidden evaluation data.
    """

    name: str = "task"
    higher_is_better: bool = True

    @abstractmethod
    def instruction(self) -> str:
        """The task statement given to every agent."""

    def setup(self, container: Path) -> None:
        """Writes the task files into a fresh container (at least AGENT.md)."""
        container.mkdir(parents=True, exist_ok=True)
        (container / "AGENT.md").write_text(self.instruction() + "\n", encoding="utf-8")

    @abstractmethod
    def evaluate(self, candidate: Path) -> Evaluation:
        """Scores a submitted candidate file. Never exposes hidden data."""


class CallableTask(Task):
    """A task defined by an instruction and a scoring function over the submitted file."""

    def __init__(
        self,
        name: str,
        instruction: str,
        scorer: Callable[[Path], float | Evaluation],
        higher_is_better: bool = True,
        files: dict[str, str] | None = None,
    ):
        self.name = name
        self._instruction = instruction
        self._scorer = scorer
        self.higher_is_better = higher_is_better
        self._files = dict(files or {})

    def instruction(self) -> str:
        return self._instruction

    def setup(self, container: Path) -> None:
        super().setup(container)
        for rel, content in self._files.items():
            path = container / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def evaluate(self, candidate: Path) -> Evaluation:
        result = self._scorer(candidate)
        if isinstance(result, Evaluation):
            return result
        return Evaluation(score=float(result), valid=True)
