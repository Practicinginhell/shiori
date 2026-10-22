"""The staged-improvement model and Proposition 1 (Park et al., 2026, Section 3.4, Appendix B).

A task needs m successive improvements; X_ij is agent i's search time at stage j.
Independent agents must each finish every stage: T_best = min_i sum_j X_ij.
A communicating team continues from the first verified discovery at each stage:
T_team = sum_j min_i X_ij  (a "sum of minima" instead of a "minimum of sums").

With X_ij ~ Exp(lambda) i.i.d. and per-agent runtime tau = alpha * m / lambda,
1/k < alpha < 1, Proposition 1 states, with I(a) = a - 1 - log a:
    Pr(T_team <= tau) >= 1 - exp(-m I(k alpha)),
    Pr(T_best <= tau) <= k exp(-m I(alpha)).

Herding: if the k agents act as g <= k independent groups whose members
duplicate the same search, the team's discovery rate is g * lambda.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def rate_function(a: float) -> float:
    """I(a) = a - 1 - log a (the Erlang/exponential Chernoff exponent)."""
    if a <= 0:
        raise ValueError("I(a) is defined for a > 0")
    return a - 1.0 - math.log(a)


def _check_regime(k: int, alpha: float) -> None:
    if k < 1 or not (1.0 / k < alpha < 1.0):
        raise ValueError(f"Proposition 1 requires 1/k < alpha < 1 (got k={k}, alpha={alpha})")


def team_success_lower_bound(m: int, k: int, alpha: float) -> float:
    """Pr(T_team <= tau) >= 1 - exp(-m I(k alpha))."""
    _check_regime(k, alpha)
    return 1.0 - math.exp(-m * rate_function(k * alpha))


def best_success_upper_bound(m: int, k: int, alpha: float) -> float:
    """Pr(T_best <= tau) <= k exp(-m I(alpha)) (the paper's union bound; may exceed 1 for small m)."""
    _check_regime(k, alpha)
    return k * math.exp(-m * rate_function(alpha))


def erlang_cdf(t: float, m: int, rate: float) -> float:
    """Pr(Erlang(m, rate) <= t) = 1 - sum_{n<m} e^{-rate t} (rate t)^n / n!."""
    if t <= 0:
        return 0.0
    x = rate * t
    term, tail = math.exp(-x), 0.0
    for n in range(m):
        if n > 0:
            term *= x / n
        tail += term
    return 1.0 - tail


def exact_team_success(m: int, k: int, lam: float, alpha: float, groups: int | None = None) -> float:
    """Pr(T_team <= tau) exactly: T_team ~ Erlang(m, g * lambda) with g = groups or k."""
    g = groups or k
    return erlang_cdf(alpha * m / lam, m, g * lam)


def exact_best_success(m: int, k: int, lam: float, alpha: float) -> float:
    """Pr(T_best <= tau) exactly: 1 - (1 - Pr(Erlang(m, lambda) <= tau))^k."""
    return 1.0 - (1.0 - erlang_cdf(alpha * m / lam, m, lam)) ** k


@dataclass
class SimulationResult:
    tau: float
    team_success: float
    best_success: float
    mean_team_time: float
    mean_best_time: float
    pointwise_team_le_best: bool


def simulate_completion(
    m: int,
    k: int,
    lam: float = 1.0,
    alpha: float = 0.5,
    trials: int = 10_000,
    seed: int | None = None,
    groups: int | None = None,
) -> SimulationResult:
    """Monte Carlo of the staged model under a shared draw of search times X_ij.

    With `groups` = g < k, the team herds: only g of its agents search independently
    at each stage. Independent agents are unaffected.
    """
    if m < 1 or k < 1 or lam <= 0 or trials < 1:
        raise ValueError("need m >= 1, k >= 1, lam > 0, trials >= 1")
    g = k if groups is None else groups
    if not 1 <= g <= k:
        raise ValueError("groups must be between 1 and k")
    rng = random.Random(seed)
    tau = alpha * m / lam
    team_hits = best_hits = 0
    team_sum = best_sum = 0.0
    pointwise = True
    for _ in range(trials):
        x = [[rng.expovariate(lam) for _ in range(m)] for _ in range(k)]
        t_best = min(sum(row) for row in x)
        t_team = sum(min(x[i][j] for i in range(g)) for j in range(m))
        if g == k and t_team > t_best + 1e-12:
            pointwise = False
        team_hits += t_team <= tau
        best_hits += t_best <= tau
        team_sum += t_team
        best_sum += t_best
    return SimulationResult(
        tau=tau,
        team_success=team_hits / trials,
        best_success=best_hits / trials,
        mean_team_time=team_sum / trials,
        mean_best_time=best_sum / trials,
        pointwise_team_le_best=pointwise,
    )
