"""The household pipeline: validate a reading, then assess quality and supply in parallel.

    reading ──▶ validate ──┬──▶ quality ──┐
                           └──▶ supply ───┴──▶ combine ──▶ status

`build_app()` compiles it as a graphflow workflow. `evaluate()` runs the same node
functions in plain Python, for bulk simulation (a test proves both give the same status).
A branch that raises is isolated by graphflow into `errors`, and `combine` turns that into
Service: a crash never leaves the light green.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from graphflow import Application, Field, State, Workflow

from . import rules


class HouseholdState(State):
    reading: dict | None = None
    context: dict | None = None
    faults: list | None = None
    quality: dict | None = None
    supply: dict | None = None
    status: dict | None = None
    errors: list = Field.reducer("append")


def validate(state: dict) -> dict:
    return {"faults": rules.sanity_faults(state["reading"], state["context"])}


def quality(state: dict) -> dict:
    return {"quality": rules.assess_quality(state["reading"], state["context"], state["faults"])}


def supply(state: dict) -> dict:
    return {"supply": rules.assess_supply(state["reading"], state["context"], state["faults"])}


def combine(state: dict) -> dict:
    status = rules.combine(state["faults"] or [], state.get("quality"), state.get("supply"), state.get("errors") or [])
    return {"status": status}


def build_app(supply_node: Callable[[dict], dict] = supply) -> Application:
    """The household workflow as a graphflow application. `supply_node` is swappable for tests."""
    wf = Workflow("household", state_schema=HouseholdState)
    wf.then(validate)
    wf.parallel([quality, supply_node], fan_in=combine, from_node="validate")
    return Application("tanklight").register(wf)


def run_app(app: Application, reading: dict, context: dict) -> dict:
    return app.run({"reading": reading, "context": context})["status"]


async def arun_app(app: Application, reading: dict, context: dict) -> dict:
    return (await app.arun({"reading": reading, "context": context}))["status"]


def evaluate(reading: dict, context: dict) -> dict:
    """The same nodes as the workflow, composed directly."""
    state: dict[str, Any] = {"reading": reading, "context": context}
    state.update(validate(state))
    state.update(quality(state))
    state.update(supply(state))
    return combine(state)["status"]


class TankMonitor:
    """What the device remembers between readings: fills, delivery gaps, usage, decay samples.

    `observe(reading)` returns the status with `state` settled for display and the raw
    evaluation in `raw_state`. Readings are dicts with hour, month, orp_mv, ph, temp_c,
    level_l and (optionally) turbidity_ntu.
    """

    FILL_JUMP_SHARE = 0.10  # a level rise over 10% of the tank between readings is a delivery
    DRAIN_DROP_SHARE = 0.20  # a drop over 20% in one reading is a drain (cleaning, a leak), not household use
    MAX_GAPS_PER_MONTH = 30

    def __init__(self, config: dict, app: Application | None = None):
        self.config = config
        self.gaps_by_month: dict[int, list[float]] = {}
        self._last_level: float | None = None
        self._last_fill_hour: float | None = None
        self._decay_samples: list[tuple[float, float, float]] = []
        self._levels: deque[tuple[float, float]] = deque()
        self._orp_recent: deque[float] = deque(maxlen=5)
        self._recent_raw: deque[str] = deque(maxlen=3)
        self._turbidity_baseline: float | None = None
        self.shown: str | None = None
        self.last_status: dict | None = None
        self.use_app(app)

    @property
    def engine_name(self) -> str:
        return "graphflow" if self._app is not None else "direct"

    def use_app(self, app: Application | None) -> None:
        self._app = app

    def hours_since_fill(self, hour: float) -> float | None:
        return None if self._last_fill_hour is None else hour - self._last_fill_hour

    def observe(self, reading: dict) -> dict:
        hour, level = reading["hour"], reading.get("level_l")
        if level is not None:
            self._track_level(hour, level, reading)
        context = self._context(hour, reading["month"])
        raw = run_app(self._app, reading, context) if self._app is not None else evaluate(reading, context)

        since = context["hours_since_delivery"]
        if raw["fc_mg_l"] is not None and since is not None:
            self._decay_samples.append((since, raw["fc_mg_l"], reading["temp_c"]))
            del self._decay_samples[:-48]
        if reading.get("orp_mv") is not None:
            self._orp_recent.append(reading["orp_mv"])
        self._recent_raw.append(raw["state"])
        self.shown = rules.settle(self.shown, list(self._recent_raw))
        self.last_status = {**raw, "raw_state": raw["state"], "state": self.shown}
        return self.last_status

    def _track_level(self, hour: float, level: float, reading: dict) -> None:
        capacity = self.config["capacity_l"]
        if self._last_level is not None and self._last_level - level > self.DRAIN_DROP_SHARE * capacity:
            self._levels.clear()  # restart the usage estimate after a drain
        if self._last_level is not None and level - self._last_level > self.FILL_JUMP_SHARE * capacity:
            if self._last_fill_hour is not None:
                gaps = self.gaps_by_month.setdefault(reading["month"], [])
                gaps.append(hour - self._last_fill_hour)
                del gaps[: -self.MAX_GAPS_PER_MONTH]
            self._last_fill_hour = hour
            self._decay_samples = []
            self._turbidity_baseline = reading.get("turbidity_ntu")
        self._last_level = level
        self._levels.append((hour, level))
        while self._levels and hour - self._levels[0][0] > 24.0:
            self._levels.popleft()

    def _context(self, hour: float, month: int) -> dict:
        return {
            **self.config,
            "calibration": self.config.get("calibration"),
            "decay_samples": list(self._decay_samples),
            "usage_l_per_day": rules.daily_usage_litres(list(self._levels)),
            "hours_since_delivery": self.hours_since_fill(hour),
            "gaps_hours": list(self.gaps_by_month.get(month, [])),  # seasons differ, so same month only
            "turbidity_baseline": self._turbidity_baseline,
            "orp_recent": list(self._orp_recent),
        }
