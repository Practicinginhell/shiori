"""Submission ledger: every attempt, timestamp, status and score, kept outside the agent container.

As in the paper's packing verifier (Appendix A.4), the run's result is the
highest valid scored submission, protecting an earlier champion from a broken
final edit.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from graphflow.ttc.tasks.base import Evaluation


@dataclass
class LedgerEntry:
    index: int
    agent: int
    slot: int | None
    candidate: str
    score: float | None
    valid: bool
    status: str
    elapsed: float  # seconds since the trial started
    tokens: int     # the submitting agent's cumulative output tokens
    details: dict[str, Any] = field(default_factory=dict)


class SubmissionLedger:
    """Append-only JSONL ledger, safe to share between concurrent agents."""

    def __init__(self, path: str | Path, higher_is_better: bool = True):
        self.path = Path(path)
        self.higher_is_better = higher_is_better
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        agent: int,
        slot: int | None,
        candidate: str,
        evaluation: Evaluation,
        elapsed: float,
        tokens: int,
    ) -> LedgerEntry:
        with self._lock:
            entry = LedgerEntry(
                index=len(self._read()),
                agent=agent,
                slot=slot,
                candidate=candidate,
                score=evaluation.score if evaluation.valid else None,
                valid=evaluation.valid,
                status=evaluation.status,
                elapsed=elapsed,
                tokens=tokens,
                details=dict(evaluation.details),
            )
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), default=str) + "\n")
            return entry

    def _read(self) -> list[LedgerEntry]:
        if not self.path.is_file():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [LedgerEntry(**json.loads(line)) for line in f if line.strip()]

    def entries(self) -> list[LedgerEntry]:
        with self._lock:
            return self._read()

    def best(self, higher_is_better: bool | None = None) -> LedgerEntry | None:
        higher = self.higher_is_better if higher_is_better is None else higher_is_better
        valid = [e for e in self.entries() if e.valid and e.score is not None]
        if not valid:
            return None
        return max(valid, key=lambda e: e.score) if higher else min(valid, key=lambda e: e.score)
