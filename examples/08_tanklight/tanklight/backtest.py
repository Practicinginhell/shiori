"""Backtest on the simulated community: did the device warn in time, and do smarter trucks help?

Reports, for one season and one truck policy:
  runouts / runouts_warned_24h     tanks that ran dry, and how many were warned 24 h ahead
  dry_home_hours                   hours homes spent with no water (the harm a truck policy can reduce)
  losses / losses_warned_6h        chlorine fell below 0.2 mg/L, and how many were warned 6 h ahead
  share_of_time_not_protected      how often the light is not green (the alarm burden)
  visits, litres_per_visit         truck stops, and how full each top-up was

SIMULATED data (see simulate.py). The numbers show the method, not Inukjuak's reality.
"""

from __future__ import annotations

from .simulate import Community


def run(season: str = "january", days: int = 30, seed: int = 11, policy: str = "priority", n_homes: int = 12,
        visits_per_day: int = 5, warmup_days: int = 14) -> dict:
    town = Community(season, n_homes=n_homes, seed=seed, policy=policy, visits_per_day=visits_per_day,
                     warmup_days=warmup_days, featured_engine="direct")
    town.advance(days * 24.0)
    s = town.stats
    return {
        "season": season,
        "policy": policy,
        "days": days,
        "homes": n_homes,
        "runouts": s["runouts"],
        "runouts_warned_24h": s["runouts_warned_24h"],
        "dry_home_hours": s["dry_home_hours"],
        "losses": s["losses"],
        "losses_warned_6h": s["losses_warned_6h"],
        "share_of_time_not_protected": round(s["not_protected_steps"] / max(1, s["home_steps"]), 3),
        "visits": s["visits"],
        "litres_per_visit": round(s["litres_delivered"] / max(1, s["visits"]), 1),
    }


def compare(season: str = "january", days: int = 30, seed: int = 11) -> list[dict]:
    """Fixed loop vs delivery priority, same town, same storms."""
    return [run(season, days, seed, policy) for policy in ("fixed", "priority")]
