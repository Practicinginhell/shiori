"""The water office plan: a forecast and the truck lists, as a note staff can act on.

    facts ──▶ draft (local LLM) ──▶ check ──▶ note

The code decides every fact: which days have no trucks, which homes to top up or pump out
before a blizzard, which tanks need a check. The LLM (a small local model through Ollama)
only rewrites that into a friendlier note. `check` keeps the LLM's note only when it carries
exactly the same home names and numbers as the facts; otherwise the plain template is used.
A small model can reword, but it can't drop a home or invent a number.

This is a graphflow workflow, like the household pipeline. Everything here is SIMULATED data.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable

from graphflow import Application, State, Workflow

from . import fleet, simulate

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # local only: the plan never leaves the office
DEFAULT_LLM = "gemma3:270m"
# Water must last until trucks are back after the blizzard, plus the usual warning margin.
MARGIN_DAYS = 1.5

PROMPT = ("Rewrite this plan as a short, clear note for the water office staff. Keep every home name and every "
          "number exactly as written. Do not add anything.\n\n")

Writer = Callable[[str], str]


def plan_facts(town: simulate.Community, predicted: list[float]) -> dict:
    """What the office should do over the forecast window, worked out by code."""
    today, horizon = town.day, len(predicted)
    storm_days: list[int] = []
    for d in range(today + 1, today + 1 + horizon):
        if town.is_storm_day(d):
            storm_days.append(d)
        elif storm_days:
            break  # the first blizzard only
    rows = town.home_statuses()
    water_short: list[str] = []
    sewage_short: list[str] = []
    stops_before = pumps_before = 0
    if storm_days:
        need_days = storm_days[-1] + 1 - today + MARGIN_DAYS
        truck_days = max(0, storm_days[0] - today - 1 + (town.hour % 24 < simulate.PLAN_HOUR))
        stops_before = town.visits_per_day * truck_days
        pumps_before = simulate.PUMP_OUTS_PER_DAY * truck_days
        # Most urgent first (an unknown level counts as most urgent), so the stops go where they matter.
        water_short = [r["home_id"] for r in sorted(rows, key=lambda r: -1 if r["days_left"] is None else r["days_left"])
                       if r["days_left"] is None or r["days_left"] < need_days]
        sewage_short = [r["home_id"] for r in sorted(rows, key=lambda r: -1 if r["sewage"].get("days_to_full") is None
                                                     else r["sewage"]["days_to_full"])
                        if r["sewage"].get("days_to_full") is None or r["sewage"]["days_to_full"] < need_days]
    checks = [r["home_id"] for r in fleet.delivery_priority(rows) if "check the tank" in r["why"]]
    peak = max(range(horizon), key=lambda k: predicted[k]) if horizon else 0
    return {
        "first_day": today + 2, "last_day": today + 1 + horizon,  # shown as "Day N", like the UI clock
        "storm_days": storm_days,
        "top_up_before_storm": water_short[:stops_before], "stops_before_storm": stops_before,
        "short_of_water": len(water_short[stops_before:]),
        "pump_out_before_storm": sewage_short[:pumps_before], "pump_outs_before_storm": pumps_before,
        "sewage_may_fill": len(sewage_short[pumps_before:]),
        "tank_checks": checks,
        "peak_day": today + 2 + peak, "peak_homes": min(len(rows), round(predicted[peak])) if horizon else 0,
    }


def template_note(facts: dict) -> str:
    lines = [f"Water office plan, Day {facts['first_day']} to Day {facts['last_day']} (simulated)."]
    storm = facts["storm_days"]
    if storm:
        when = f"on Day {storm[0] + 1}" if len(storm) == 1 else f"from Day {storm[0] + 1} to Day {storm[-1] + 1}"
        lines.append(f"Blizzard: no trucks {when}.")
        if facts["top_up_before_storm"]:
            lines.append(f"Before the blizzard, top up first: {', '.join(facts['top_up_before_storm'])}.")
        if facts["short_of_water"]:
            lines.append(f"{facts['short_of_water']} more homes may run short during the blizzard: "
                         "ask them to save water.")
        if facts["pump_out_before_storm"]:
            lines.append(f"Before the blizzard, pump out first: {', '.join(facts['pump_out_before_storm'])}.")
        if facts["sewage_may_fill"]:
            lines.append(f"{facts['sewage_may_fill']} more sewage tanks may fill up during the blizzard.")
    else:
        lines.append("No blizzard in the forecast.")
    lines.append(f"Busiest day: Day {facts['peak_day']}, about {facts['peak_homes']} homes needing urgent water.")
    if facts["tank_checks"]:
        lines.append(f"Tank checks (a truck won't fix these): {', '.join(facts['tank_checks'])}.")
    return "\n".join(lines)


def _tokens(text: str) -> tuple[set[str], set[str]]:
    homes = set(re.findall(r"home-\d+", text))
    numbers = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"home-\d+", "", text)))
    return homes, numbers


# The action a line asks for. Lines that name homes must keep their action, so "top up
# home-01" can't become "pump out home-01". A note that rewords the action is rejected (safe).
ACTIONS = ("top up", "pump out", "tank check", "run short", "fill up")


def _line_key(line: str) -> tuple:
    text = line.lower()
    return (next((a for a in ACTIONS if a in text), None), *_tokens(line))


def faithful(note: str, template: str) -> bool:
    """True when `note` names exactly the same homes and numbers as `template`, line by line.

    Each template line's homes, numbers and action must appear together on one line of the note,
    so a note that moves a home from the top-up list to the pump-out list is rejected. Merged or split
    lines are rejected too; the template goes out instead, which is always safe.
    """
    if _tokens(note) != _tokens(template):
        return False
    note_lines = [_line_key(line) for line in note.splitlines()]
    return all(_line_key(line) in note_lines for line in template.splitlines() if any(_tokens(line)))


def ollama_writer(model: str = DEFAULT_LLM, timeout: float = 60.0) -> Writer:
    """A writer that asks a local Ollama model to reword the plan."""
    def write(text: str) -> str:
        body = json.dumps({"model": model, "prompt": PROMPT + text, "stream": False,
                           "options": {"temperature": 0}}).encode()
        request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read())["response"].strip()
    return write


class OfficeState(State):
    facts: dict | None = None
    template: str | None = None
    draft: str | None = None
    writer_error: str | None = None
    note: str | None = None
    source: str | None = None


def build_office_app(writer: Writer | None) -> Application:
    def compose(state: dict) -> dict:
        return {"template": template_note(state["facts"])}

    def draft(state: dict) -> dict:
        if writer is None:
            return {}
        try:
            return {"draft": writer(state["template"])}
        except Exception as exc:  # no Ollama, a timeout, a bad reply: the template still goes out
            return {"writer_error": f"{type(exc).__name__}: {exc}"}

    def check(state: dict) -> dict:
        text = state.get("draft")
        if text and faithful(text, state["template"]):
            return {"note": text, "source": "llm"}
        return {"note": state["template"], "source": "template"}

    wf = Workflow("office_plan", state_schema=OfficeState)
    wf.then(compose).then(draft).then(check)
    return Application("tanklight-office").register(wf)


def run_office_plan(facts: dict, writer: Writer | None = None) -> dict:
    """Run the office workflow: {"note", "source" ("llm" or "template"), "draft", "writer_error"}."""
    out = build_office_app(writer).run({"facts": facts})
    return {k: out.get(k) for k in ("note", "source", "draft", "writer_error")}
