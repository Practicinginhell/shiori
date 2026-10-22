"""A simulated community of truck-served homes, for the demo and the backtest.

Everything here is SIMULATED. The physics (chlorine decay, water use, storms) uses
plausible placeholder numbers, not measurements from Inukjuak; replace them with real
delivery logs and DPD readings before drawing conclusions.

Each home has a true state (litres, chlorine, cloudiness) that the sensors read with noise.
A TankMonitor per home sees only those readings, as the device would. Trucks make a
limited number of visits per day, chosen by a fixed loop or by delivery priority.
"""

from __future__ import annotations

import math
import random
from typing import Any

from . import fleet, pipeline, rules

SEASONS = {
    # water_temp_c: water in an indoor tank; storm_day_prob: days trucks can't run.
    # outdoor_temp_c: around the sewage tank outside or under the house.
    "january": {"label": "January", "month": 1, "water_temp_c": 16.0, "storm_day_prob": 0.15, "outdoor_temp_c": -25.0},
    "july": {"label": "July", "month": 7, "water_temp_c": 23.0, "storm_day_prob": 0.02, "outdoor_temp_c": 8.0},
}
TRUE_K20_PER_HOUR = 0.025
TRUCK_FC_RANGE = (0.3, 0.8)  # Cambridge Bay trucks averaged 0.48 mg/L (IJCH 2025)
BASE_TURBIDITY_NTU = 0.3
STEP_HOURS = 0.5
PLAN_HOUR, FIRST_VISIT_HOUR = 8.0, 9.0
TIMELINE_POINTS = 7 * 48
POLICIES = ("fixed", "priority")
# Sewage tank: a heater keeps it above freezing; without it the tank cools toward the
# outdoor air (time constant in hours). Placeholders.
HEATED_TANK_C = 8.0
TANK_COOLING_HOURS = 6.0
PUMP_OUTS_PER_DAY = 4


class Home:
    """A home's tank as it really is (the device never sees this directly)."""

    def __init__(self, home_id: str, people: int, capacity_l: float, litres_per_person_day: float,
                 litres: float, fc_mg_l: float):
        self.home_id = home_id
        self.people = people
        self.capacity_l = capacity_l
        self.litres_per_person_day = litres_per_person_day
        self.litres = litres
        self.fc_mg_l = fc_mg_l
        self.turbidity_ntu = BASE_TURBIDITY_NTU
        self.probe_attached = True
        # The sewage tank: every litre used ends up here.
        self.sewage_capacity_l = capacity_l
        self.sewage_l = 0.0
        self.sewage_temp_c = HEATED_TANK_C
        self.heater_on = True

    def advance(self, hours: float, temp_c: float, outdoor_c: float = HEATED_TANK_C) -> None:
        used = min(self.litres, self.people * self.litres_per_person_day * hours / 24.0)
        self.litres -= used
        self.sewage_l = min(self.sewage_capacity_l, self.sewage_l + used)
        self.fc_mg_l *= math.exp(-rules.decay_rate(TRUE_K20_PER_HOUR, temp_c) * hours)
        target = HEATED_TANK_C if self.heater_on else outdoor_c
        self.sewage_temp_c += (target - self.sewage_temp_c) * (1.0 - math.exp(-hours / TANK_COOLING_HOURS))

    def deliver(self, fc_fill: float) -> float:
        """Top the tank up; the new water mixes with what's left. Returns litres delivered."""
        added = self.capacity_l - self.litres
        self.fc_mg_l = (self.litres * self.fc_mg_l + added * fc_fill) / self.capacity_l
        self.turbidity_ntu = (self.litres * self.turbidity_ntu + added * BASE_TURBIDITY_NTU) / self.capacity_l
        self.litres = self.capacity_l
        return added

    def read(self, rng: random.Random, hour: float, month: int, temp_c: float) -> dict:
        """One sensor reading, with noise."""
        ph = 7.4 + rng.gauss(0.0, 0.03)
        orp = None
        if self.probe_attached:
            orp = rules.orp_from_free_chlorine(max(self.fc_mg_l, 1e-4), ph, temp_c) + rng.gauss(0.0, 3.0)
        return {
            "hour": hour, "month": month, "orp_mv": orp, "ph": ph, "temp_c": temp_c + rng.gauss(0.0, 0.2),
            "level_l": max(0.0, self.litres + rng.gauss(0.0, 4.0)),
            "turbidity_ntu": self.turbidity_ntu + abs(rng.gauss(0.0, 0.05)),
        }

    def read_sewage(self, rng: random.Random) -> tuple[float, float]:
        """(litres, deg C) from the sewage tank's level and temperature sensors, with noise."""
        return max(0.0, self.sewage_l + rng.gauss(0.0, 4.0)), self.sewage_temp_c + rng.gauss(0.0, 0.2)

    def config(self) -> dict:
        return {"home_id": self.home_id, "capacity_l": self.capacity_l, "people": self.people, "default_ph": 7.4}


class Community:
    """Homes, their monitors, and the trucks. Home 0 is the featured home in the demo.

    After `warmup_days` (so each device has learned its delivery gaps) the featured home's
    monitor switches to the graphflow pipeline, unless `featured_engine="direct"`.
    """

    def __init__(self, season: str = "january", n_homes: int = 12, seed: int = 7, policy: str = "priority",
                 visits_per_day: int = 5, warmup_days: int = 21, featured_engine: str = "graphflow"):
        if season not in SEASONS:
            raise ValueError(f"unknown season {season!r}; choose one of {sorted(SEASONS)}")
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choose one of {POLICIES}")
        self.season = season
        self.policy = policy
        self.visits_per_day = visits_per_day
        self.seed = seed
        self.rng = random.Random(seed)
        self.hour = 0.0
        self.homes = [self._new_home(i) for i in range(n_homes)]
        # Sewage draws come from their own streams, so adding the sewage tank left every
        # drinking-water result (and the backtest numbers) unchanged.
        self.sewage_rng = random.Random(f"{seed}-sewage")
        for i, home in enumerate(self.homes):
            home.sewage_l = 300.0 if i == 0 else home.sewage_capacity_l * random.Random(f"{seed}-sewage-{i}").uniform(0.0, 0.7)
        self.sewage = [rules.assess_sewage(h.sewage_l, h.sewage_capacity_l, h.sewage_temp_c, None) for h in self.homes]
        self._sewage_since_pump: list[list[tuple[float, float]]] = [[] for _ in self.homes]
        self._pump_plan: list[int] = []
        self.daily: list[dict] = []  # one row per day at planning time: the forecaster's history
        # What families asked for in messages (inbox.py), until a truck answers it.
        self.requests: dict[str, set[str]] = {h.home_id: set() for h in self.homes}
        self.inbox: list[dict] = []  # newest first
        self._next_in_pump_loop = 0
        self.monitors = [pipeline.TankMonitor(h.config()) for h in self.homes]
        self.timeline: list[dict] = []
        self._storm: dict[int, bool] = {}
        self._plan: list[int] = []
        self._next_in_loop = 0
        self._tracking = [self._fresh_tracking() for _ in self.homes]
        self.counting = False
        self.stats = self._fresh_stats()

        self.advance(warmup_days * 24.0)
        self.counting = True
        if featured_engine == "graphflow":
            self.monitors[0].use_app(pipeline.build_app())

    # ------------------------------------------------------------------ setup
    def _new_home(self, i: int) -> Home:
        if i == 0:  # the demo home: five people, a 1,100 L tank (about four days of water)
            return Home("home-01", 5, 1100.0, 55.0, 700.0, 0.3)
        capacity = self.rng.choice([900.0, 1100.0, 1350.0])
        return Home(f"home-{i + 1:02d}", self.rng.choice([2, 3, 4, 5, 6, 7, 8]), capacity,
                    self.rng.uniform(45.0, 65.0), capacity * self.rng.uniform(0.3, 1.0), self.rng.uniform(0.05, 0.5))

    @staticmethod
    def _fresh_tracking() -> dict:
        return {"supply_warned_at": None, "quality_warned_at": None, "out": False, "unprotected": False}

    @staticmethod
    def _fresh_stats() -> dict:
        return {"runouts": 0, "runouts_warned_24h": 0, "dry_home_hours": 0.0, "losses": 0, "losses_warned_6h": 0,
                "visits": 0, "litres_delivered": 0.0, "home_steps": 0, "not_protected_steps": 0,
                "pump_outs": 0, "sewage_backup_hours": 0.0}

    # ------------------------------------------------------------------ time
    @property
    def day(self) -> int:
        return int(self.hour // 24)

    @property
    def month(self) -> int:
        return SEASONS[self.season]["month"]

    def water_temp(self) -> float:
        # A small daily swing: the utility room is warmer in the evening.
        return SEASONS[self.season]["water_temp_c"] + 0.5 * math.sin(2 * math.pi * (self.hour % 24 - 9) / 24)

    def is_storm_day(self, day: int) -> bool:
        # Seeded per day, so looking (e.g. a UI refresh) never changes what happens next.
        if day not in self._storm:
            draw = random.Random(f"{self.seed}-{self.season}-{day}").random()
            self._storm[day] = draw < SEASONS[self.season]["storm_day_prob"]
        return self._storm[day]

    def advance(self, hours: float) -> None:
        for _ in range(round(hours / STEP_HOURS)):
            self._step()

    def _step(self) -> None:
        hod = self.hour % 24
        if hod == PLAN_HOUR:
            storm = self.is_storm_day(self.day)
            self._record_day(storm)
            self._plan = [] if storm else self._plan_day()
            self._pump_plan = [] if storm else self._plan_pump_outs()
        visit = hod - FIRST_VISIT_HOUR
        if visit >= 0 and visit == int(visit) and int(visit) < len(self._plan):
            self._deliver(self._plan[int(visit)])
        if visit >= 0 and visit == int(visit) and int(visit) < len(self._pump_plan):
            self._pump_out(self._pump_plan[int(visit)])

        temp, outdoor = self.water_temp(), SEASONS[self.season]["outdoor_temp_c"]
        for home in self.homes:
            home.advance(STEP_HOURS, temp, outdoor)
        self.hour += STEP_HOURS
        for i, (home, monitor) in enumerate(zip(self.homes, self.monitors, strict=True)):
            status = monitor.observe(home.read(self.rng, self.hour, self.month, temp))
            self._track_events(i, home, status)
            self._observe_sewage(i, home)
        self._record_timeline()

    # ------------------------------------------------------------------ trucks
    def _plan_day(self) -> list[int]:
        n = len(self.homes)
        if self.policy == "fixed":
            plan = [(self._next_in_loop + k) % n for k in range(self.visits_per_day)]
            self._next_in_loop = (self._next_in_loop + self.visits_per_day) % n
            return plan
        index = {h.home_id: i for i, h in enumerate(self.homes)}
        ranked = fleet.delivery_priority(self.home_statuses())
        return [index[r["home_id"]] for r in ranked if not r["can_skip"]][: self.visits_per_day]

    def _deliver(self, i: int, fc_fill: float | None = None) -> None:
        fc = self.rng.uniform(*TRUCK_FC_RANGE) if fc_fill is None else fc_fill
        added = self.homes[i].deliver(fc)
        self.requests[self.homes[i].home_id].discard("water_delivery")
        self._tracking[i] = self._fresh_tracking()
        if self.counting:
            self.stats["visits"] += 1
            self.stats["litres_delivered"] += added

    def _plan_pump_outs(self) -> list[int]:
        n = len(self.homes)
        if self.policy == "fixed":
            plan = [(self._next_in_pump_loop + k) % n for k in range(PUMP_OUTS_PER_DAY)]
            self._next_in_pump_loop = (self._next_in_pump_loop + PUMP_OUTS_PER_DAY) % n
            return plan
        index = {h.home_id: i for i, h in enumerate(self.homes)}
        ranked = fleet.pump_out_priority(self.home_statuses())
        return [index[r["home_id"]] for r in ranked if not r["can_skip"]][:PUMP_OUTS_PER_DAY]

    def _pump_out(self, i: int) -> None:
        self.homes[i].sewage_l = 0.0
        self._sewage_since_pump[i] = []
        self.requests[self.homes[i].home_id].discard("pump_out")
        if self.counting:
            self.stats["pump_outs"] += 1

    def _observe_sewage(self, i: int, home: Home) -> None:
        litres, temp = home.read_sewage(self.sewage_rng)
        history = self._sewage_since_pump[i]
        history.append((self.hour, litres))
        del history[:-96]  # the last two days
        # Inflow from the rise since the last pump-out; until six hours are seen, assume the usual use.
        inflow = home.people * rules.ASSUMED_LITRES_PER_PERSON_DAY
        if history[-1][0] - history[0][0] >= 6.0:
            inflow = max(1.0, (history[-1][1] - history[0][1]) * 24.0 / (history[-1][0] - history[0][0]))
        self.sewage[i] = rules.assess_sewage(litres, home.sewage_capacity_l, temp, inflow)
        if self.counting and home.sewage_l >= home.sewage_capacity_l:
            self.stats["sewage_backup_hours"] += STEP_HOURS

    def home_statuses(self) -> list[dict]:
        rows = []
        for home, monitor in zip(self.homes, self.monitors, strict=True):
            status = monitor.last_status or {}
            rows.append({"home_id": home.home_id, "people": home.people, "capacity_l": home.capacity_l,
                         **{k: status.get(k) for k in ("state", "reasons", "level_pct", "days_left", "runout_risk",
                                                        "fc_mg_l", "hours_protected")},
                         "sewage": self.sewage[len(rows)], "requests": sorted(self.requests[home.home_id])})
        return rows

    # ------------------------------------------------------------------ bookkeeping
    def _track_events(self, i: int, home: Home, status: dict) -> None:
        track, shown, reasons = self._tracking[i], status["state"], status["reasons"]
        warning = shown in ("check", "boil")
        if warning and track["supply_warned_at"] is None and {"water_running_low", "tank_nearly_empty"} & set(reasons):
            track["supply_warned_at"] = self.hour
        if warning and track["quality_warned_at"] is None and {"protection_fading", "no_chlorine_protection"} & set(reasons):
            track["quality_warned_at"] = self.hour
        if not self.counting:
            return
        self.stats["home_steps"] += 1
        self.stats["not_protected_steps"] += shown != "protected"
        if home.litres <= 0:
            self.stats["dry_home_hours"] += STEP_HOURS
        if home.litres <= 0 and not track["out"]:
            track["out"] = True
            self.stats["runouts"] += 1
            warned = track["supply_warned_at"]
            self.stats["runouts_warned_24h"] += warned is not None and self.hour - warned >= 24.0
        if home.litres > 0 and home.fc_mg_l < rules.PROTECTED_MG_L and not track["unprotected"]:
            track["unprotected"] = True
            self.stats["losses"] += 1
            warned = track["quality_warned_at"]
            self.stats["losses_warned_6h"] += warned is not None and self.hour - warned >= 6.0

    def _record_day(self, storm: bool) -> None:
        homes = self.home_statuses()
        self.daily.append({
            "day": self.day, "storm": storm,
            "urgent_water": sum(r["tier"] <= 1 for r in fleet.delivery_priority(homes)),
            "urgent_sewage": sum(r["tier"] <= 1 for r in fleet.pump_out_priority(homes)),
        })

    def _record_timeline(self) -> None:
        if not self.counting:
            return
        home, status = self.homes[0], self.monitors[0].last_status
        self.timeline.append({"h": self.hour, "fc_true": round(home.fc_mg_l, 3), "fc_est": status["fc_mg_l"],
                              "level_pct": round(100 * home.litres / home.capacity_l, 1), "state": status["state"]})
        del self.timeline[:-TIMELINE_POINTS]

    # ------------------------------------------------------------------ demo actions (featured home)
    def featured_home(self) -> Home:
        return self.homes[0]

    def featured_monitor(self) -> pipeline.TankMonitor:
        return self.monitors[0]

    def featured_status(self) -> dict:
        return self.monitors[0].last_status

    def deliver_now(self, fc_fill: float | None = None) -> None:
        self._deliver(0, fc_fill)

    def drain(self, pct: float) -> None:
        """Demo setup: set the featured tank to `pct` percent full, as if the water had been used."""
        home = self.homes[0]
        home.litres = home.capacity_l * max(0.0, min(100.0, pct)) / 100.0

    def vitamin_c(self) -> None:
        """A vitamin C tablet neutralises chlorine (safe to do in a demo jar)."""
        self.homes[0].fc_mg_l = 0.0

    def make_murky(self) -> None:
        self.homes[0].turbidity_ntu += 8.0

    def pull_probe(self) -> None:
        self.homes[0].probe_attached = False

    def attach_probe(self) -> None:
        self.homes[0].probe_attached = True

    def pump_out_now(self) -> None:
        self._pump_out(0)

    def fresh_sewage_tank(self) -> None:
        """Demo setup: the featured sewage tank just pumped out, warm, heater on (repeatable demos)."""
        self._pump_out(0)
        home = self.homes[0]
        home.heater_on, home.sewage_temp_c = True, HEATED_TANK_C

    def heater_off(self) -> None:
        """Demo: the sewage tank's heater fails (in January the tank starts to freeze)."""
        self.homes[0].heater_on = False

    def heater_on(self) -> None:
        self.homes[0].heater_on = True

    def storm(self, days: int) -> None:
        """No trucks from today for `days` days."""
        for d in range(self.day, self.day + days):
            self._storm[d] = True
        self._plan = []
        self._pump_plan = []

    def record_request(self, home_id: str, needs: list[str]) -> None:
        """A family asked for something (from the inbox); the truck plans take it into account."""
        self.requests[home_id].update(n for n in needs if n in ("water_delivery", "pump_out", "water_quality", "device"))

    def add_message(self, decision: dict) -> None:
        self.inbox.insert(0, decision)
        del self.inbox[20:]

    def storm_on(self, first_day: int, days: int) -> None:
        """A blizzard forecast: no trucks on `days` days from `first_day` (demo and tests)."""
        for d in range(first_day, first_day + days):
            self._storm[d] = True

    def set_policy(self, policy: str) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy

    def snapshot(self) -> dict[str, Any]:
        """Everything the UI shows, as JSON-ready data."""
        home = self.homes[0]
        homes = self.home_statuses()
        return {
            "simulated": True,
            "hour": self.hour,
            "day": self.day,
            "clock": f"Day {self.day + 1}, {int(self.hour % 24):02d}:{int(self.hour % 1 * 60):02d}",
            "season": self.season,
            "season_label": SEASONS[self.season]["label"],
            "storm_today": self.is_storm_day(self.day),
            "policy": self.policy,
            "visits_per_day": self.visits_per_day,
            "featured": {
                "home_id": home.home_id, "people": home.people, "capacity_l": home.capacity_l,
                "probe_attached": home.probe_attached, "engine": self.monitors[0].engine_name,
                "status": self.monitors[0].last_status, "timeline": self.timeline,
                "sewage": self.sewage[0], "heater_on": home.heater_on,
            },
            "homes": homes,
            "fleet": fleet.delivery_priority(homes),
            "pump_outs": fleet.pump_out_priority(homes),
            "inbox": list(self.inbox),
            "pump_outs_per_day": PUMP_OUTS_PER_DAY,
            "stats": dict(self.stats),
        }
