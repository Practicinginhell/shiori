"""TankLight rules core: every decision about a household tank, as plain functions.

This file is the one source of truth for the device and the simulator. It imports only
`math` and avoids annotations, dataclasses and f-string tricks so the same file can be
copied onto an ESP32 running MicroPython. (Written to that subset; not yet run on a device.)

Units: free chlorine in mg/L, ORP in mV, temperature in deg C, volume in litres, time in
hours. Numbers marked "placeholder" must be calibrated in the field before any real use.
"""

import math

# Regulatory minimum free chlorine for delivered water in Nunavut, and the household
# protection threshold used by SWOT (York U / MSF). Below it, water is unprotected.
PROTECTED_MG_L = 0.2
# Warn this many hours before the forecast says protection will run out.
PROTECTION_WARNING_HOURS = 24.0
# Minimum for drinking, cooking and handwashing (ANTHC Mini-PASS evaluation, 20 L/p/d).
MIN_LITRES_PER_PERSON_DAY = 20.0
# Used until the device has seen a day of real use. Placeholder.
ASSUMED_LITRES_PER_PERSON_DAY = 60.0

# Chlorine decay: first-order, k(T) = k20 * THETA ** (T - 20). Placeholders.
DECAY_THETA = 1.07
DEFAULT_K20_PER_HOUR = 0.03
MIN_K20, MAX_K20 = 0.002, 0.3

# ORP -> hypochlorous acid: ln(HOCl mg/L) = (ORP - orp_at_1mg_hocl) / slope_mv.
# Placeholder until fitted against DPD test-kit readings in the home (fit_calibration).
DEFAULT_CALIBRATION = {"slope_mv": 40.0, "orp_at_1mg_hocl": 720.0}

SEVERITY = {"protected": 0, "check": 1, "service": 2, "boil": 3}
REASON_STATE = {
    "no_chlorine_protection": "boil",
    "water_cloudy": "boil",
    "tank_nearly_empty": "boil",
    "sensor_fault": "service",
    "internal_error": "service",
    "protection_fading": "check",
    "water_cloudier": "check",
    "water_running_low": "check",
}

TURBIDITY_CHECK_JUMP_NTU = 1.0
TURBIDITY_BOIL_JUMP_NTU = 5.0
LEVEL_BOIL_PCT = 5.0
LEVEL_CHECK_PCT = 15.0
RUNOUT_CHECK_RISK = 0.5
# Warn with a day and a half of water left, whatever the delivery history says: a storm
# can outlast every gap the device has seen, and a household needs time to ask for a truck.
RUNOUT_WARNING_HOURS = 36.0
ORP_FAULTS = ("orp_missing", "orp_range", "orp_stuck")
LEVEL_FAULTS = ("level_missing", "level_range")


# --------------------------------------------------------------------------- #
# Chemistry
# --------------------------------------------------------------------------- #

def hocl_fraction(ph, temp_c):
    """Share of free chlorine present as HOCl (what ORP responds to) at this pH and temperature."""
    kelvin = temp_c + 273.15
    pka = 3000.0 / kelvin - 10.0686 + 0.0253 * kelvin  # Morris (1966)
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def free_chlorine_from_orp(orp_mv, ph, temp_c, calibration=None):
    """Estimated free chlorine (mg/L) from an ORP reading."""
    cal = calibration or DEFAULT_CALIBRATION
    hocl = math.exp((orp_mv - cal["orp_at_1mg_hocl"]) / cal["slope_mv"])
    return hocl / hocl_fraction(ph, temp_c)


def orp_from_free_chlorine(fc_mg_l, ph, temp_c, calibration=None):
    """The ORP (mV) the calibration expects for this free chlorine; the inverse of the above."""
    cal = calibration or DEFAULT_CALIBRATION
    hocl = fc_mg_l * hocl_fraction(ph, temp_c)
    return cal["orp_at_1mg_hocl"] + cal["slope_mv"] * math.log(hocl)


def fit_calibration(samples):
    """Fit the ORP curve from (orp_mv, ph, temp_c, dpd_mg_l) pairs taken with a DPD test kit.

    Raises ValueError with fewer than two usable pairs or when all ORP values are equal.
    """
    points = [(orp, math.log(dpd * hocl_fraction(ph, temp))) for orp, ph, temp, dpd in samples if dpd > 0]
    if len(points) < 2:
        raise ValueError("calibration needs at least two DPD readings above zero")
    slope, intercept = _line_fit(points)
    if slope is None or slope <= 0:
        raise ValueError("calibration readings do not rise with chlorine; check the probe")
    return {"slope_mv": 1.0 / slope, "orp_at_1mg_hocl": -intercept / slope}


def decay_rate(k20, temp_c):
    """Chlorine decay rate per hour at this water temperature."""
    return k20 * DECAY_THETA ** (temp_c - 20.0)


def fit_decay(samples, default_k20=DEFAULT_K20_PER_HOUR):
    """Fit the decay since the last delivery from (hours_since_fill, fc_mg_l, temp_c) samples.

    Returns {"c0", "k20", "fitted"}. Falls back to `default_k20` (fitted False) with fewer
    than three samples, under three hours of data, or a rate outside the plausible range.
    """
    usable = [(t, fc, temp) for t, fc, temp in samples if fc > 0]
    if len(usable) >= 3 and usable[-1][0] - usable[0][0] >= 3.0:
        # Temperature-normalised time: an hour at 24 C counts as 1.07^4 hours at 20 C.
        points = [(t * DECAY_THETA ** (temp - 20.0), math.log(fc)) for t, fc, temp in usable]
        slope, intercept = _line_fit(points)
        if slope is not None and MIN_K20 <= -slope <= MAX_K20:
            return {"c0": math.exp(intercept), "k20": -slope, "fitted": True}
    c0 = usable[0][1] if usable else None
    return {"c0": c0, "k20": default_k20, "fitted": False}


def hours_protected(fc_mg_l, k20, temp_c, threshold=PROTECTED_MG_L):
    """Hours until free chlorine decays below `threshold` (0 when it already has)."""
    if fc_mg_l <= threshold:
        return 0.0
    return math.log(fc_mg_l / threshold) / decay_rate(k20, temp_c)


# --------------------------------------------------------------------------- #
# Supply
# --------------------------------------------------------------------------- #

def daily_usage_litres(levels):
    """Litres used per day from chronological (hour, litres) readings; fills don't count.

    Returns None with under six hours of readings.
    """
    if len(levels) < 2 or levels[-1][0] - levels[0][0] < 6.0:
        return None
    used = 0.0
    for i in range(1, len(levels)):  # no itertools.pairwise on MicroPython
        drop = levels[i - 1][1] - levels[i][1]
        if drop > 0:
            used += drop
    return used * 24.0 / (levels[-1][0] - levels[0][0])


def percentile(values, q):
    """Nearest-rank percentile (q in 0..1) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def runout_risk(hours_of_water, hours_since_delivery, gaps_hours):
    """Chance the next delivery comes after the water runs out, from past delivery gaps.

    Uses only past gaps longer than the time already waited (the truck hasn't come yet).
    Returns None without history, and 1.0 when the wait is already longer than every gap seen.
    """
    if not gaps_hours:
        return None
    still_possible = [g for g in gaps_hours if g > hours_since_delivery]
    if not still_possible:
        return 1.0
    late = [g for g in still_possible if g > hours_since_delivery + hours_of_water]
    return len(late) / len(still_possible)


# --------------------------------------------------------------------------- #
# Sewage (the outdoor tank)
# --------------------------------------------------------------------------- #

# The sewage tank is outside or under the house. It has its own status, separate from the
# drinking-water light: a full sewage tank backs up into the house, a frozen one can't be
# pumped, but neither makes the drinking water unsafe. Placeholders.
SEWAGE_SEVERITY = {"ok": 0, "soon": 1, "service": 2, "urgent": 3}
SEWAGE_REASON_STATE = {"sewage_full": "urgent", "sewage_freezing": "urgent", "sewage_filling": "soon",
                       "sewage_sensor_fault": "service"}
SEWAGE_SOON_PCT = 75.0
SEWAGE_FULL_PCT = 95.0
SEWAGE_SOON_DAYS = 1.5
SEWAGE_FREEZE_C = 2.0


def assess_sewage(level_l, capacity_l, temp_c, inflow_l_per_day):
    """Status of the sewage tank: how full, days until full, and freezing risk."""
    if level_l is None or not -0.02 * capacity_l <= level_l <= 1.05 * capacity_l:
        return {"state": "service", "reasons": ["sewage_sensor_fault"], "level_pct": None, "days_to_full": None,
                "temp_c": temp_c}
    level_l = max(0.0, level_l)
    level_pct = 100.0 * level_l / capacity_l
    days_to_full = max(0.0, capacity_l - level_l) / inflow_l_per_day if inflow_l_per_day else None
    reasons = []
    if level_pct >= SEWAGE_FULL_PCT:
        reasons.append("sewage_full")
    if temp_c is not None and temp_c <= SEWAGE_FREEZE_C:
        reasons.append("sewage_freezing")
    if level_pct >= SEWAGE_SOON_PCT or (days_to_full is not None and days_to_full < SEWAGE_SOON_DAYS):
        reasons.append("sewage_filling")
    state = "ok"
    for code in reasons:
        if SEWAGE_SEVERITY[SEWAGE_REASON_STATE[code]] > SEWAGE_SEVERITY[state]:
            state = SEWAGE_REASON_STATE[code]
    return {"state": state, "reasons": reasons, "level_pct": round(level_pct, 1),
            "days_to_full": None if days_to_full is None else round(days_to_full, 2),
            "temp_c": None if temp_c is None else round(temp_c, 1)}


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

def sanity_faults(reading, context):
    """Sensor faults in this reading. Any fault means the device must not show Protected."""
    faults = []
    orp = reading.get("orp_mv")
    if orp is None:
        faults.append("orp_missing")
    else:
        if not -200.0 <= orp <= 1200.0:
            faults.append("orp_range")
        recent = context.get("orp_recent") or []
        if len(recent) >= 5 and all(v == orp for v in recent[-5:]):
            faults.append("orp_stuck")
    temp = reading.get("temp_c")
    if temp is None or not 0.0 <= temp <= 45.0:
        faults.append("temp_range")
    ph = reading.get("ph")
    if ph is not None and not 5.0 <= ph <= 10.0:
        faults.append("ph_range")
    level = reading.get("level_l")
    if level is None:
        faults.append("level_missing")
    elif not -0.02 * context["capacity_l"] <= level <= 1.05 * context["capacity_l"]:
        faults.append("level_range")
    return faults


def assess_quality(reading, context, faults):
    """Chlorine protection and clarity. Unknown (state None) when the ORP or temperature is faulty."""
    if any(f in faults for f in ORP_FAULTS) or "temp_range" in faults:
        return {"state": None, "reasons": [], "fc_mg_l": None, "hours_protected": None}
    ph = reading.get("ph")
    if ph is None or "ph_range" in faults:
        ph = context["default_ph"]
    temp = reading["temp_c"]
    fc = free_chlorine_from_orp(reading["orp_mv"], ph, temp, context.get("calibration"))
    samples = list(context.get("decay_samples") or [])
    if context.get("hours_since_delivery") is not None:
        samples.append((context["hours_since_delivery"], fc, temp))
    hours = hours_protected(fc, fit_decay(samples)["k20"], temp)

    reasons = []
    if fc < PROTECTED_MG_L:
        reasons.append("no_chlorine_protection")
    elif hours < PROTECTION_WARNING_HOURS:
        reasons.append("protection_fading")
    turbidity, baseline = reading.get("turbidity_ntu"), context.get("turbidity_baseline")
    if turbidity is not None and baseline is not None:
        if turbidity - baseline >= TURBIDITY_BOIL_JUMP_NTU:
            reasons.append("water_cloudy")
        elif turbidity - baseline >= TURBIDITY_CHECK_JUMP_NTU:
            reasons.append("water_cloudier")
    return {"state": _worst(reasons), "reasons": reasons, "fc_mg_l": round(fc, 3), "hours_protected": round(hours, 1)}


def assess_supply(reading, context, faults):
    """Water left and the chance it runs out before the truck. Unknown when the level is faulty."""
    if any(f in faults for f in LEVEL_FAULTS):
        return {"state": None, "reasons": [], "level_pct": None, "days_left": None, "essential_days": None,
                "runout_risk": None}
    litres = max(0.0, reading["level_l"])
    level_pct = 100.0 * litres / context["capacity_l"]
    people = context["people"]
    usage = context.get("usage_l_per_day") or people * ASSUMED_LITRES_PER_PERSON_DAY
    days_left = litres / usage
    since = context.get("hours_since_delivery")
    risk = None if since is None else runout_risk(days_left * 24.0, since, context.get("gaps_hours") or [])

    reasons = []
    if level_pct < LEVEL_BOIL_PCT:
        reasons.append("tank_nearly_empty")  # the pump draws sediment from the bottom
    if (level_pct < LEVEL_CHECK_PCT or days_left * 24.0 < RUNOUT_WARNING_HOURS
            or (risk is not None and risk >= RUNOUT_CHECK_RISK)):
        reasons.append("water_running_low")
    return {"state": _worst(reasons), "reasons": reasons, "level_pct": round(level_pct, 1),
            "days_left": round(days_left, 2), "essential_days": round(litres / (people * MIN_LITRES_PER_PERSON_DAY), 2),
            "runout_risk": None if risk is None else round(risk, 2)}


def combine(faults, quality, supply, errors):
    """One status: the worst state of every reason; reasons ordered most severe first."""
    reasons = []
    if errors:
        reasons.append("internal_error")
    if faults:
        reasons.append("sensor_fault")
    for part in (quality, supply):
        if part:
            reasons.extend(part["reasons"])
    if (quality is None or supply is None) and not errors:
        reasons.append("internal_error")
    # Stable sort: most severe first, original order within a severity.
    reasons = sorted(reasons, key=lambda code: -SEVERITY[REASON_STATE[code]])
    quality, supply = quality or {}, supply or {}
    return {
        "state": _worst(reasons),
        "reasons": reasons,
        "faults": list(faults),
        "fc_mg_l": quality.get("fc_mg_l"),
        "hours_protected": quality.get("hours_protected"),
        "level_pct": supply.get("level_pct"),
        "days_left": supply.get("days_left"),
        "essential_days": supply.get("essential_days"),
        "runout_risk": supply.get("runout_risk"),
    }


def settle(shown, recent_raw):
    """The state to display, given the one shown now and recent raw states (newest last).

    A sensor fault shows at once (unless Boil is showing). Getting worse needs the last two
    readings to agree; getting better needs three, so one noisy reading neither alarms nor
    reassures.
    """
    if not recent_raw:
        return shown
    if shown is None:
        return recent_raw[-1]
    rank = SEVERITY[shown]
    if recent_raw[-1] == "service" and rank < SEVERITY["service"]:
        return "service"
    last_two, last_three = recent_raw[-2:], recent_raw[-3:]
    if len(last_two) == 2 and all(SEVERITY[s] > rank for s in last_two):
        return min(last_two, key=lambda s: SEVERITY[s])
    if len(last_three) == 3 and all(SEVERITY[s] < rank for s in last_three):
        return max(last_three, key=lambda s: SEVERITY[s])
    return shown


def _worst(reasons):
    state = "protected"
    for code in reasons:
        if SEVERITY[REASON_STATE[code]] > SEVERITY[state]:
            state = REASON_STATE[code]
    return state


def _line_fit(points):
    """Least-squares line through (x, y) points: (slope, intercept), or (None, None) if x is constant."""
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx == 0:
        return None, None
    slope = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / sxx
    return slope, mean_y - slope * mean_x
