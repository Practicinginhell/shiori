"""Delivery priority for the water trucks: who needs water first, and who can wait.

A fixed loop visits every house in turn, full or not. Ranking by each tank's own forecast
sends trucks to the homes about to run out and skips homes that are still full, so the
same trucks prevent more run-outs with fewer wasted stops.
"""

from __future__ import annotations

import math

SKIP_LEVEL_PCT = 50.0  # used only when the device doesn't know the household's use yet
# A tank is "full enough" by days of water, not by percent: 70% lasts 2 days for 8 people.
SKIP_DAYS = 3.0
# Run-out risk comes from past delivery gaps, which run long for homes the plan skipped before;
# it only makes a home urgent when the water is also short.
RISK_MATTERS_BELOW_DAYS = 3.0
# A top-up only restores chlorine when most of the tank becomes new water.
TOP_UP_HELPS_BELOW_PCT = 60.0


def _tier(home: dict) -> tuple[int, str, bool]:
    """(tier, why, can_skip) for one home."""
    reasons, risk = home.get("reasons") or [], home.get("runout_risk") or 0.0
    days_left, level = home.get("days_left"), home.get("level_pct")
    if "water_delivery" in (home.get("requests") or []):
        return 1, "The family asked for water", False  # they can see the tank; trust them over the sensor
    short = days_left is None or days_left < RISK_MATTERS_BELOW_DAYS
    if "tank_nearly_empty" in reasons or (risk >= 0.9 and short) or (days_left is not None and days_left < 0.5):
        return 0, "Tank nearly empty or will run out before the next truck", False
    if risk >= 0.5 and short:
        return 1, f"May run out before the next truck (risk {risk:.0%})", False
    if "water_running_low" in reasons:
        return 1, "Less than a day and a half of water left", False
    chlorine_low = "no_chlorine_protection" in reasons or "protection_fading" in reasons
    if chlorine_low and (level is None or level < TOP_UP_HELPS_BELOW_PCT):
        return 2, "Chlorine protection low: a delivery replaces most of the water", False
    if "no_chlorine_protection" in reasons:
        return 3, "No chlorine protection but tank full: a top-up won't fix it; ask the water office to check the tank", True
    if "water_cloudy" in reasons:
        return 3, "Cloudy water: a top-up won't clear it; ask the water office to check the tank", True
    full_enough = days_left >= SKIP_DAYS if days_left is not None else level is not None and level >= SKIP_LEVEL_PCT
    if full_enough:
        return 3, "Full enough: skip today", True
    return 3, "Routine top-up", False


def delivery_priority(homes: list[dict]) -> list[dict]:
    """Homes in delivery order, most urgent first, each with `why` and `can_skip`.

    Within a tier: fewest days of water first, then the larger household.
    """
    ranked = []
    for home in homes:
        tier, why, can_skip = _tier(home)
        ranked.append({
            "home_id": home["home_id"],
            "tier": tier,
            "why": why,
            "can_skip": can_skip,
            "level_pct": home.get("level_pct"),
            "days_left": home.get("days_left"),
            "people": home.get("people"),
        })
    days = {h["home_id"]: h.get("days_left") for h in homes}
    ranked.sort(key=lambda r: (r["tier"], math.inf if days[r["home_id"]] is None else days[r["home_id"]],
                               -(r["people"] or 0)))
    return ranked


SEWAGE_SKIP_PCT = 50.0


def _pump_out_tier(sewage: dict, requests: list[str]) -> tuple[int, str, bool]:
    reasons, level, days = sewage.get("reasons") or [], sewage.get("level_pct"), sewage.get("days_to_full")
    if "pump_out" in requests:
        return 0, "The family reported sewage backing up or a full tank", False
    if "sewage_full" in reasons:
        return 0, "Sewage tank full: it can back up into the house", False
    if "sewage_freezing" in reasons:
        return 0, "Sewage tank near freezing: pump out before it freezes and check the heater", False
    if "sewage_filling" in reasons:
        return 1, "Sewage tank full within a day and a half", False
    if "sewage_sensor_fault" in reasons:
        return 2, "Sewage level unknown: check the sensor on this visit", False
    if level is not None and level < SEWAGE_SKIP_PCT and (days is None or days >= 3):
        return 3, "Plenty of room: skip today", True
    return 3, "Routine pump-out", False


def pump_out_priority(homes: list[dict]) -> list[dict]:
    """Homes in pump-out order for the sewage truck, most urgent first (fewest days to full)."""
    ranked = []
    for home in homes:
        sewage = home.get("sewage") or {}
        tier, why, can_skip = _pump_out_tier(sewage, home.get("requests") or [])
        ranked.append({"home_id": home["home_id"], "tier": tier, "why": why, "can_skip": can_skip,
                       "level_pct": sewage.get("level_pct"), "days_to_full": sewage.get("days_to_full"),
                       "people": home.get("people")})
    ranked.sort(key=lambda r: (r["tier"], math.inf if r["days_to_full"] is None else r["days_to_full"],
                               -(r["people"] or 0)))
    return ranked
