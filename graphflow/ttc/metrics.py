"""Evaluation metrics from Park et al. (2026).

- best@k (Section 2.1): the best outcome among k independent agents. For a finite
  pool with s successes among n trials, computed exactly without replacement:
  best@k = 1 - C(n - s, k) / C(n, k).
- RHAE (Eq. 1-2): relative human action efficiency for level-based games.
- Best-so-far trajectories against time or tokens (Appendix A.5).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Point = tuple[float, float]


def best_at_k_exact(n: int, s: int, k: int) -> float:
    """Probability that k trials drawn without replacement from n (s successes) include a success."""
    if not (0 <= s <= n) or not (1 <= k <= n):
        raise ValueError(f"need 0 <= s <= n and 1 <= k <= n (got n={n}, s={s}, k={k})")
    return 1.0 - math.comb(n - s, k) / math.comb(n, k)


def mean_best_at_k(games: Sequence[tuple[int, int]], k: int) -> float:
    """best@k averaged over games, each weighted equally. `games` holds (n, s) per game."""
    return sum(best_at_k_exact(n, s, k) for n, s in games) / len(games)


def matching_pool_size(games: Sequence[tuple[int, int]], target: float) -> int | None:
    """Smallest k whose mean best@k reaches `target` (e.g. a team's solve rate), or None."""
    max_k = min(n for n, _ in games)
    for k in range(1, max_k + 1):
        if mean_best_at_k(games, k) >= target - 1e-12:
            return k
    return None


def rhae_level_score(human: float, agent: float | None) -> float:
    """Eq. 1: S = min(1.15, (h / a)^2); 0 for a level the agent never completes."""
    if agent is None:
        return 0.0
    if agent <= 0:
        raise ValueError("agent action count must be positive")
    return min(1.15, (human / agent) ** 2)


def rhae_game(levels: Sequence[tuple[float, float | None]]) -> float:
    """Eq. 2: level-weighted score (w_l = l), capped by the weighted fraction of completed levels.

    `levels` lists (human baseline actions, agent actions or None if not completed) in level order.
    """
    weights = [float(level) for level in range(1, len(levels) + 1)]
    total = sum(weights)
    completed = sum(w for w, (_, agent) in zip(weights, levels, strict=True) if agent is not None)
    weighted = sum(w * rhae_level_score(h, a) for w, (h, a) in zip(weights, levels, strict=True))
    return min(completed / total, weighted / total)


def rhae(games: Iterable[Sequence[tuple[float, float | None]]]) -> float:
    """Reported RHAE: the mean of per-game scores."""
    scores = [rhae_game(g) for g in games]
    return sum(scores) / len(scores)


def best_so_far(events: Iterable[tuple[float, float | None]], higher_is_better: bool = True) -> list[Point]:
    """Best-so-far curve from (x, score) events in x order; None scores (invalid) are skipped."""
    curve: list[Point] = []
    best: float | None = None
    for x, score in events:
        if score is None:
            continue
        if best is None or (score > best if higher_is_better else score < best):
            best = score
            curve.append((x, score))
    return curve


def independent_best_trajectory(
    runs: Sequence[Sequence[tuple[float, float | None]]], higher_is_better: bool = True
) -> list[Point]:
    """best@k over time: at time t, the best result any run found within its own first t (Appendix A.5)."""
    merged = sorted((e for run in runs for e in run), key=lambda e: e[0])
    return best_so_far(merged, higher_is_better)


def tokens_at(timeline: Sequence[tuple[float, int]], t: float) -> int:
    """Cumulative tokens of one agent by elapsed time t, from its (elapsed, cumulative) timeline."""
    total = 0
    for elapsed, cumulative in timeline:
        if elapsed > t:
            break
        total = cumulative
    return total


def on_token_axis(
    curve: Sequence[Point], timelines: Sequence[Sequence[tuple[float, int]]]
) -> list[Point]:
    """Moves a best-so-far curve from elapsed time to total output tokens of *all* agents
    by that time (Appendix A.5: neither curve counts only the improving agent's tokens)."""
    return [(float(sum(tokens_at(tl, t) for tl in timelines)), score) for t, score in curve]
