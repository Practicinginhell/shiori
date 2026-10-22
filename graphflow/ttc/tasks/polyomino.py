"""Polyomino packing (Park et al., 2026, Appendix A.4; task from Frontier-CS).

Each case holds n edge-connected polyominoes of one to ten cells. A program may
reflect, rotate by multiples of 90 degrees, and translate each piece; all pieces
must lie without overlap in one integer-grid rectangle. With C occupied cells
and returned area A = W * H, case quality is C / A and the score is the mean over
the fixed cases. Invalid output rejects the submission. Candidates are GNU C++17
programs with 2 seconds and 256 MiB per case.

The Frontier-CS hidden instances are not public: ``PolyominoPackingTask``
generates its own seeded instances with the paper's parameters
(70 cases, n in [100, 10^4] pieces, 1-10 cells). Instances live only in the
scorer (in memory) and are never written into the agent's container.

I/O format (ours; Frontier-CS's exact format is not given in the paper):
  input   N, then N lines "c x1 y1 ... xc yc" (cells of each piece)
  output  "W H", then N lines "r f x y": flip (f=1 mirrors x -> -x), then rotate
          r quarter-turns counter-clockwise ((x, y) -> (-y, x)), then shift so the
          piece's minimum x and y are 0, then translate by (x, y).
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from graphflow.ttc.tasks.base import Evaluation, Task

Cell = tuple[int, int]


class InvalidPacking(ValueError):
    """The program's output is not a valid packing (the submission is rejected)."""


def _normalize(cells: list[Cell]) -> list[Cell]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return [(x - min_x, y - min_y) for x, y in cells]


def transform_cells(cells: list[Cell], rotation: int, flip: int) -> list[Cell]:
    """Flip (x -> -x) if requested, rotate `rotation` quarter-turns CCW, then normalize to min 0."""
    out = [(-x, y) if flip else (x, y) for x, y in cells]
    for _ in range(rotation % 4):
        out = [(-y, x) for x, y in out]
    return _normalize(out)


@dataclass(frozen=True)
class Piece:
    cells: list[Cell]

    def __hash__(self) -> int:
        return hash(tuple(self.cells))

    def is_edge_connected(self) -> bool:
        cells = set(self.cells)
        if not cells:
            return False
        seen, stack = set(), [next(iter(cells))]
        while stack:
            x, y = stack.pop()
            if (x, y) in seen:
                continue
            seen.add((x, y))
            stack.extend(n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if n in cells)
        return seen == cells


def random_polyomino(rng: random.Random, size: int) -> Piece:
    """Grows an edge-connected polyomino of `size` cells from the origin."""
    cells = {(0, 0)}
    while len(cells) < size:
        x, y = rng.choice(sorted(cells))
        dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((x + dx, y + dy))
    return Piece(sorted(_normalize(sorted(cells))))


def format_case(pieces: list[Piece]) -> str:
    lines = [str(len(pieces))]
    for p in pieces:
        coords = " ".join(f"{x} {y}" for x, y in p.cells)
        lines.append(f"{len(p.cells)} {coords}")
    return "\n".join(lines) + "\n"


def packing_quality(pieces: list[Piece], output: str) -> float:
    """Validates a program's output for one case and returns C / (W * H)."""
    try:
        tokens = [int(t) for t in output.split()]
    except ValueError:
        raise InvalidPacking("could not parse output as integers") from None
    if len(tokens) < 2:
        raise InvalidPacking("could not parse the rectangle size 'W H'")
    width, height = tokens[0], tokens[1]
    if width <= 0 or height <= 0:
        raise InvalidPacking(f"could not parse a positive rectangle (got {width} x {height})")
    placements = tokens[2:]
    if len(placements) != 4 * len(pieces):
        raise InvalidPacking(f"expected {len(pieces)} piece placements, got {len(placements) / 4:g}")

    occupied: set[Cell] = set()
    total = 0
    for i, piece in enumerate(pieces):
        rotation, flip, dx, dy = placements[4 * i: 4 * i + 4]
        if rotation not in (0, 1, 2, 3) or flip not in (0, 1):
            raise InvalidPacking(f"piece {i}: invalid rotation/flip ({rotation}, {flip})")
        for x, y in transform_cells(piece.cells, rotation, flip):
            cx, cy = x + dx, y + dy
            if not (0 <= cx < width and 0 <= cy < height):
                raise InvalidPacking(f"piece {i}: cell ({cx}, {cy}) lies outside the {width} x {height} rectangle")
            if (cx, cy) in occupied:
                raise InvalidPacking(f"piece {i}: cell ({cx}, {cy}) causes an overlap")
            occupied.add((cx, cy))
        total += len(piece.cells)
    return total / (width * height)


def _command(binary: Path, memory_mb: int) -> list[str]:
    """The command running `binary` under a best-effort address-space limit.

    The limit is set by the shell (``ulimit -v``) rather than ``preexec_fn``,
    which is unsafe when the process has threads (agents submit concurrently).
    Where ``ulimit -v`` is not supported (e.g. macOS) the program runs unlimited.
    """
    if os.name != "posix":
        return [str(binary)]
    return ["/bin/sh", "-c", f'ulimit -v {memory_mb * 1024} 2>/dev/null; exec "$0"', str(binary)]


class PolyominoPackingTask(Task):
    """Frontier-CS-style polyomino packing with a hidden-case scorer."""

    name = "polyomino-packing"
    higher_is_better = True

    def __init__(
        self,
        num_cases: int = 70,
        min_pieces: int = 100,
        max_pieces: int = 10_000,
        max_cells: int = 10,
        seed: int = 0,
        time_limit: float = 2.0,
        memory_mb: int = 256,
        compiler: str = "c++",
    ):
        self.num_cases = num_cases
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces
        self.max_cells = max_cells
        self.seed = seed
        self.time_limit = time_limit
        self.memory_mb = memory_mb
        self.compiler = compiler
        self._cases: list[list[Piece]] | None = None

    def cases(self) -> list[list[Piece]]:
        if self._cases is None:
            rng = random.Random(self.seed)
            self._cases = [
                [random_polyomino(rng, rng.randint(1, self.max_cells))
                 for _ in range(rng.randint(self.min_pieces, self.max_pieces))]
                for _ in range(self.num_cases)
            ]
        return self._cases

    def instruction(self) -> str:
        return f"""# Polyomino packing

Write a GNU C++17 program that packs polyominoes into a rectangle of minimum area.

Input (stdin): N, then N lines "c x1 y1 ... xc yc" giving the cells of each
edge-connected piece (1-{self.max_cells} cells; {self.min_pieces}-{self.max_pieces} pieces per case).

Output (stdout): "W H", then one line "r f x y" per piece, in input order:
flip the piece if f=1 (x -> -x), rotate it r quarter-turns counter-clockwise
((x, y) -> (-y, x)), shift it so its minimum x and y are 0, then translate it by
(x, y). Every cell must lie in [0, W) x [0, H), and no two cells may overlap.

Score: for each case, occupied cells / (W * H); the score is the mean over
{self.num_cases} hidden cases, in [0, 1], higher is better. Any invalid case
(bad output, overlap, crash, over {self.time_limit:g}s or {self.memory_mb} MiB)
rejects the whole submission.

Scoring: call submit(path) with your .cpp file. It compiles and runs it on the
hidden cases and returns the score; you never see the cases or program outputs.
You may submit as often as you like; the best valid submission counts.
"""

    def evaluate(self, candidate: Path) -> Evaluation:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="gf-poly-") as tmp:
            binary = Path(tmp) / ("solution.exe" if sys.platform == "win32" else "solution")
            try:
                compiled = subprocess.run(  # noqa: S603 - compiling the agent's program is the task
                    [self.compiler, "-std=gnu++17", "-O2", "-o", str(binary), str(candidate)],
                    capture_output=True, text=True, timeout=120, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                return Evaluation(None, False, f"rejected: compile error: {e}")
            if compiled.returncode != 0:
                return Evaluation(None, False, f"rejected: compile error: {compiled.stderr[-800:]}")

            qualities: list[float] = []
            for index, pieces in enumerate(self.cases()):
                try:
                    run = subprocess.run(  # noqa: S603 - running the agent's program is the task
                        _command(binary, self.memory_mb),
                        input=format_case(pieces),
                        capture_output=True,
                        text=True,
                        timeout=self.time_limit,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return Evaluation(None, False, f"rejected: case {index}: time limit {self.time_limit:g}s exceeded")
                if run.returncode != 0:
                    return Evaluation(None, False, f"rejected: case {index}: exit code {run.returncode}")
                try:
                    qualities.append(packing_quality(pieces, run.stdout))
                except InvalidPacking as e:
                    return Evaluation(None, False, f"rejected: case {index}: {e}")

        score = sum(qualities) / len(qualities)
        return Evaluation(
            score=score,
            valid=True,
            details={
                "cases": len(qualities),
                "min_case": min(qualities),
                "seconds": round(time.perf_counter() - started, 3),
            },
        )
