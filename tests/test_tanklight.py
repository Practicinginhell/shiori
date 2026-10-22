"""TankLight example (examples/08_tanklight): the household water protection tracker.

Each end condition in tasks/todo.md is a test here:
  rules      chlorine from ORP, decay forecast, run-out risk, fail-safe checks, worst state wins
  pipeline   the graphflow household workflow agrees with the direct path; a failing branch is Service
  simulate   the demo scenarios (vitamin C, murky water, probe pulled, storm, season)
  backtest   warnings before run-outs and chlorine loss, deterministic
  fleet      delivery priority for the trucks
  server     the JSON API and the static-file whitelist
  ui         the TypeScript UI builds with Node's type stripping
"""

import asyncio
import json
import math
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "08_tanklight"
sys.path.insert(0, str(EXAMPLE))

from tanklight import backtest, fleet, forecast, inbox, office, pipeline, rules, simulate  # noqa: E402
from tanklight.server import make_server  # noqa: E402

CONFIG = {"home_id": "h1", "capacity_l": 1000.0, "people": 4, "default_ph": 7.4}


def reading(**overrides):
    base = {"hour": 10.0, "month": 1, "orp_mv": 700.0, "ph": 7.4, "temp_c": 18.0, "level_l": 800.0,
            "turbidity_ntu": 0.3}
    base.update(overrides)
    return base


def context(**overrides):
    base = {**CONFIG, "calibration": None, "decay_samples": [], "usage_l_per_day": 240.0,
            "hours_since_delivery": 10.0, "gaps_hours": [48.0, 50.0, 52.0, 60.0], "turbidity_baseline": 0.3,
            "orp_recent": []}
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# rules: chemistry and forecasts
# --------------------------------------------------------------------------- #

def test_hocl_fraction_is_half_at_pka_and_falls_with_ph():
    assert rules.hocl_fraction(7.54, 25.0) == pytest.approx(0.5, abs=0.01)
    assert rules.hocl_fraction(8.0, 20.0) < rules.hocl_fraction(7.0, 20.0)


def test_orp_and_free_chlorine_are_inverse_and_monotonic():
    fc = rules.free_chlorine_from_orp(680.0, 7.4, 18.0)
    assert rules.orp_from_free_chlorine(fc, 7.4, 18.0) == pytest.approx(680.0)
    assert rules.free_chlorine_from_orp(620.0, 7.4, 18.0) < fc


def test_field_calibration_recovers_the_orp_curve():
    truth = {"slope_mv": 35.0, "orp_at_1mg_hocl": 690.0}
    samples = [(rules.orp_from_free_chlorine(c, 7.4, 18.0, truth), 7.4, 18.0, c) for c in (0.1, 0.3, 0.8)]
    fitted = rules.fit_calibration(samples)
    assert fitted["slope_mv"] == pytest.approx(35.0, rel=1e-6)
    assert fitted["orp_at_1mg_hocl"] == pytest.approx(690.0, rel=1e-6)
    with pytest.raises(ValueError):
        rules.fit_calibration(samples[:1])


def test_decay_is_faster_in_warm_water():
    assert rules.decay_rate(0.02, 24.0) > rules.decay_rate(0.02, 20.0) > rules.decay_rate(0.02, 16.0)
    assert rules.decay_rate(0.02, 20.0) == pytest.approx(0.02)


def test_decay_fit_recovers_rate_and_falls_back_without_data():
    k20 = 0.03
    samples = [(t, 0.6 * math.exp(-rules.decay_rate(k20, 18.0) * t), 18.0) for t in (1.0, 6.0, 12.0, 20.0)]
    fit = rules.fit_decay(samples)
    assert fit["fitted"] is True
    assert fit["k20"] == pytest.approx(k20, rel=1e-6)
    assert rules.fit_decay([(1.0, 0.5, 18.0)])["k20"] == rules.DEFAULT_K20_PER_HOUR


def test_hours_protected_counts_down_to_the_regulatory_minimum():
    k = rules.decay_rate(0.03, 20.0)
    assert rules.hours_protected(0.4, 0.03, 20.0) == pytest.approx(math.log(0.4 / 0.2) / k)
    assert rules.hours_protected(0.15, 0.03, 20.0) == 0.0
    assert rules.PROTECTED_MG_L == 0.2


def test_daily_usage_ignores_fills():
    levels = [(0.0, 900.0), (6.0, 840.0), (12.0, 780.0), (13.0, 1000.0), (24.0, 890.0)]
    # 60 + 60 + 110 litres used in 24 hours; the refill at hour 13 is not "negative use"
    assert rules.daily_usage_litres(levels) == pytest.approx(230.0)
    assert rules.daily_usage_litres([(0.0, 900.0), (2.0, 880.0)]) is None


def test_runout_risk_uses_the_same_months_delivery_gaps():
    gaps = [40.0, 48.0, 50.0, 72.0, 120.0]
    # 10 h since the last delivery, 30 h of water left: late only if the gap exceeds 40 h
    assert rules.runout_risk(30.0, 10.0, gaps) == pytest.approx(4 / 5)
    assert rules.runout_risk(200.0, 10.0, gaps) == 0.0
    assert rules.runout_risk(10.0, 130.0, gaps) == 1.0  # overdue past every gap seen
    assert rules.runout_risk(10.0, 10.0, []) is None


def test_percentile_is_nearest_rank():
    assert rules.percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 20.0
    assert rules.percentile([10.0, 20.0, 30.0, 40.0], 0.9) == 40.0


# --------------------------------------------------------------------------- #
# rules: status
# --------------------------------------------------------------------------- #

def test_sanity_faults_never_let_bad_sensors_look_protected():
    assert rules.sanity_faults(reading(orp_mv=None), context()) == ["orp_missing"]
    assert "orp_range" in rules.sanity_faults(reading(orp_mv=2500.0), context())
    assert "temp_range" in rules.sanity_faults(reading(temp_c=80.0), context())
    stuck = context(orp_recent=[700.0] * 5)
    assert "orp_stuck" in rules.sanity_faults(reading(orp_mv=700.0), stuck)
    status = pipeline.evaluate(reading(orp_mv=None), context())
    assert status["state"] == "service"


def test_no_chlorine_means_boil_first():
    status = pipeline.evaluate(reading(orp_mv=560.0), context())
    assert status["state"] == "boil"
    assert "no_chlorine_protection" in status["reasons"]
    assert status["fc_mg_l"] < rules.PROTECTED_MG_L


def test_fresh_well_chlorinated_full_tank_is_protected():
    status = pipeline.evaluate(reading(orp_mv=720.0), context(hours_since_delivery=2.0))
    assert status["state"] == "protected", status
    assert status["hours_protected"] > rules.PROTECTION_WARNING_HOURS


def test_fading_protection_and_low_water_are_check():
    fading = pipeline.evaluate(reading(orp_mv=650.0), context())
    assert fading["state"] == "check" and "protection_fading" in fading["reasons"]
    low = pipeline.evaluate(reading(orp_mv=720.0, level_l=120.0), context())
    assert low["state"] == "check" and "water_running_low" in low["reasons"]


def test_murky_water_and_an_empty_tank_are_boil():
    murky = pipeline.evaluate(reading(orp_mv=720.0, turbidity_ntu=6.0), context())
    assert murky["state"] == "boil" and "water_cloudy" in murky["reasons"]
    empty = pipeline.evaluate(reading(orp_mv=720.0, level_l=30.0), context())
    assert empty["state"] == "boil" and "tank_nearly_empty" in empty["reasons"]


def test_worst_state_wins_and_reasons_are_ordered_by_severity():
    status = pipeline.evaluate(reading(orp_mv=560.0, level_l=120.0), context())
    assert status["state"] == "boil"
    assert status["reasons"][0] == "no_chlorine_protection"
    assert "water_running_low" in status["reasons"]


def test_settle_escalates_after_two_readings_and_recovers_after_three():
    assert rules.settle("protected", ["protected", "boil"]) == "protected"  # one noisy reading
    assert rules.settle("protected", ["boil", "boil"]) == "boil"
    assert rules.settle("boil", ["protected", "protected"]) == "boil"
    assert rules.settle("boil", ["protected", "protected", "protected"]) == "protected"
    assert rules.settle("protected", ["service"]) == "service"  # a fault shows at once


# --------------------------------------------------------------------------- #
# pipeline: the graphflow household workflow
# --------------------------------------------------------------------------- #

def test_graphflow_pipeline_agrees_with_the_direct_path():
    app = pipeline.build_app()
    for r, c in [(reading(), context()), (reading(orp_mv=560.0), context()), (reading(orp_mv=None), context()),
                 (reading(level_l=30.0), context())]:
        assert pipeline.run_app(app, r, c) == pipeline.evaluate(r, c)


def test_graphflow_pipeline_async_path():
    app = pipeline.build_app()
    status = asyncio.run(pipeline.arun_app(app, reading(orp_mv=560.0), context()))
    assert status["state"] == "boil"


def test_a_failing_branch_shows_service_not_protected():
    def broken_supply(state):
        raise RuntimeError("level sensor driver crashed")

    app = pipeline.build_app(supply_node=broken_supply)
    status = pipeline.run_app(app, reading(orp_mv=720.0), context(hours_since_delivery=2.0))
    assert status["state"] == "service"
    assert "internal_error" in status["reasons"]


def test_monitor_detects_fills_and_learns_delivery_gaps():
    monitor = pipeline.TankMonitor(CONFIG)
    hour = 0.0
    for level in (900.0, 700.0, 1000.0):
        monitor.observe(reading(hour=hour, level_l=level))
        hour += 30.0
    for level in (800.0, 1000.0):
        monitor.observe(reading(hour=hour, level_l=level))
        hour += 30.0
    assert monitor.gaps_by_month[1] == [60.0]
    assert monitor.hours_since_fill(hour - 30.0) == 0.0


def test_monitor_does_not_count_a_drain_as_household_use():
    monitor = pipeline.TankMonitor(CONFIG)
    for hour, level in [(0.0, 1000.0), (6.0, 950.0), (12.0, 900.0), (12.5, 100.0), (18.5, 50.0), (24.5, 0.0)]:
        status = monitor.observe(reading(hour=hour, level_l=level))
    # 12 hours after the drain at 100 L/12 h, not 1,000 L in a day
    assert monitor._context(24.5, 1)["usage_l_per_day"] == pytest.approx(200.0, rel=0.05)
    assert status["days_left"] == 0.0


# --------------------------------------------------------------------------- #
# simulate: the demo scenarios
# --------------------------------------------------------------------------- #

@pytest.fixture
def town():
    return simulate.Community("january", seed=3, warmup_days=14)


def test_simulation_is_deterministic():
    a = simulate.Community("january", seed=5, warmup_days=5)
    b = simulate.Community("january", seed=5, warmup_days=5)
    assert a.snapshot() == b.snapshot()


def test_vitamin_c_turns_the_light_red(town):
    town.deliver_now()
    town.advance(1.0)
    town.vitamin_c()
    town.advance(1.5)
    assert town.featured_status()["state"] == "boil"


def test_murky_water_turns_the_light_red(town):
    town.deliver_now()
    town.advance(1.0)
    town.make_murky()
    town.advance(1.5)
    assert "water_cloudy" in town.featured_status()["reasons"]


def test_pulled_probe_is_service_and_reattaching_recovers(town):
    town.deliver_now(fc_fill=0.8)
    town.advance(2.0)
    assert town.featured_status()["state"] != "boil"  # Boil is never downgraded to Service
    town.pull_probe()
    town.advance(0.5)
    assert town.featured_status()["state"] == "service"
    town.attach_probe()
    town.advance(3.0)
    assert town.featured_status()["state"] != "service"


def test_storm_warns_before_the_tank_runs_out(town):
    town.deliver_now()
    town.storm(days=6)
    warned_at = ran_out_at = None
    for _ in range(6 * 48):
        town.advance(0.5)
        status = town.featured_status()
        if warned_at is None and "water_running_low" in status["reasons"]:
            warned_at = town.hour
        if town.featured_home().litres <= 0:
            ran_out_at = town.hour
            break
    assert ran_out_at is not None, "the storm should outlast one tank"
    assert warned_at is not None and ran_out_at - warned_at >= 24.0


def test_protection_lasts_longer_in_january_than_july():
    hours = {}
    for season in ("january", "july"):
        town = simulate.Community(season, seed=3, warmup_days=14)
        town.deliver_now(fc_fill=0.6)
        town.advance(1.0)
        hours[season] = town.featured_status()["hours_protected"]
    assert hours["january"] > hours["july"] > 0


def test_graphflow_runs_the_featured_home(town):
    assert town.featured_monitor().engine_name == "graphflow"


# --------------------------------------------------------------------------- #
# sewage: the outdoor tank (fills with used water, can freeze)
# --------------------------------------------------------------------------- #

def test_sewage_tank_warns_before_it_is_full_and_when_it_may_freeze():
    ok = rules.assess_sewage(300.0, 1000.0, 6.0, 250.0)
    assert ok["state"] == "ok" and ok["days_to_full"] == pytest.approx(2.8)
    assert rules.assess_sewage(800.0, 1000.0, 6.0, 250.0)["reasons"] == ["sewage_filling"]
    assert rules.assess_sewage(650.0, 1000.0, 6.0, 250.0)["reasons"] == ["sewage_filling"]  # under 1.5 days to full
    full = rules.assess_sewage(960.0, 1000.0, 6.0, 250.0)
    assert full["state"] == "urgent" and full["reasons"][0] == "sewage_full"
    cold = rules.assess_sewage(300.0, 1000.0, 1.0, 250.0)
    assert cold["state"] == "urgent" and cold["reasons"] == ["sewage_freezing"]
    assert rules.assess_sewage(None, 1000.0, 6.0, 250.0)["state"] == "service"


def test_the_sewage_tank_fills_with_used_water_and_a_pump_out_empties_it(town):
    town.pump_out_now()
    before = town.featured_home().sewage_l
    town.advance(24.0)
    used = town.featured_home().people * town.featured_home().litres_per_person_day
    assert town.featured_home().sewage_l - before == pytest.approx(used, rel=0.05)
    town.pump_out_now()
    assert town.featured_home().sewage_l == 0.0
    assert town.snapshot()["featured"]["sewage"]["state"] == "ok"


def test_a_failed_tank_heater_in_january_warns_of_freezing(town):
    town.pump_out_now()
    town.heater_off()
    town.advance(12.0)
    sewage = town.snapshot()["featured"]["sewage"]
    assert sewage["state"] == "urgent" and "sewage_freezing" in sewage["reasons"]
    town.heater_on()
    town.advance(12.0)
    assert "sewage_freezing" not in town.snapshot()["featured"]["sewage"]["reasons"]


def test_a_fresh_sewage_tank_is_empty_warm_and_heated_for_a_repeatable_demo(town):
    town.heater_off()
    town.advance(12.0)
    town.fresh_sewage_tank()
    home = town.featured_home()
    assert home.sewage_l == 0.0 and home.heater_on and home.sewage_temp_c == simulate.HEATED_TANK_C
    town.heater_off()
    town.advance(6.0)
    assert "sewage_freezing" in town.snapshot()["featured"]["sewage"]["reasons"]


def test_the_sewage_label_is_translated():
    i18n = (EXAMPLE / "ui" / "src" / "i18n.ts").read_text()
    page = (EXAMPLE / "ui" / "index.html").read_text()
    assert "Réservoir d'eaux usées (extérieur)" in i18n and 'id="sewage-label"' in page


def test_pump_out_priority_sends_the_sewage_truck_to_full_and_freezing_tanks_first():
    homes = [
        {"home_id": "a", "people": 3, "sewage": rules.assess_sewage(200.0, 1000.0, 6.0, 200.0)},
        {"home_id": "b", "people": 3, "sewage": rules.assess_sewage(970.0, 1000.0, 6.0, 200.0)},
        {"home_id": "c", "people": 3, "sewage": rules.assess_sewage(780.0, 1000.0, 6.0, 200.0)},
        {"home_id": "d", "people": 3, "sewage": rules.assess_sewage(300.0, 1000.0, 0.5, 200.0)},
    ]
    ranked = fleet.pump_out_priority(homes)
    assert [r["home_id"] for r in ranked[:2]] == ["b", "d"]
    assert ranked[2]["home_id"] == "c" and ranked[3]["can_skip"]
    assert "heater" in ranked[1]["why"]


def test_priority_pump_outs_cut_sewage_backups():
    fixed = simulate.Community("january", seed=4, policy="fixed", warmup_days=10)
    priority = simulate.Community("january", seed=4, policy="priority", warmup_days=10)
    for town in (fixed, priority):
        town.advance(20 * 24.0)
    assert priority.stats["sewage_backup_hours"] < fixed.stats["sewage_backup_hours"]


# --------------------------------------------------------------------------- #
# forecast and office plan (the local AI; tests use fakes, so no model is needed)
# --------------------------------------------------------------------------- #

def test_the_town_keeps_a_daily_history_for_forecasting():
    town = simulate.Community("january", seed=3, warmup_days=10)
    town.advance(5 * 24.0)
    days = [row["day"] for row in town.daily]
    assert days == list(range(len(days))) and len(days) >= 14
    assert {"urgent_water", "urgent_sewage", "storm"} <= set(town.daily[0])


def test_baseline_forecast_is_the_mean_of_the_last_week():
    assert forecast.baseline([0, 0, 0, 7, 7, 7, 7, 7, 7, 7], 3) == [7.0, 7.0, 7.0]


def test_backtest_scores_forecasters_on_the_same_past_days():
    series = [float(d % 7) for d in range(70)]
    perfect = lambda history, horizon, storms: [float((len(history) + k) % 7) for k in range(horizon)]  # noqa: E731
    scores = forecast.backtest(series, [False] * 70, {"perfect": perfect, "baseline": forecast.baseline_model},
                               horizon=7, origins=4)
    assert scores["perfect"] == 0.0 and scores["baseline"] > 0.5


def test_the_storm_schedule_is_passed_as_a_known_future_input():
    seen = {}

    def model(history, horizon, storms):
        seen["storms"] = storms
        return [0.0] * horizon

    forecast.backtest([1.0] * 30, [d == 27 for d in range(30)], {"m": model}, horizon=3, origins=1)
    assert len(seen["storms"]) == 30 and seen["storms"][27]


def test_a_failing_model_falls_back_to_the_baseline(monkeypatch):
    def broken(history, horizon, storms):
        raise OSError("no internet to download the weights")

    monkeypatch.setattr(forecast, "timesfm_available", lambda: True)
    monkeypatch.setattr(forecast, "timesfm_model", broken)
    week = forecast.forecast_week([3.0] * 60, [False] * 67, 12)
    assert week["model"].startswith("baseline") and "no internet" in week["model_error"]


@pytest.fixture
def office_town():
    town = simulate.Community("january", seed=3, warmup_days=21)
    town.storm_on(town.day + 2, days=3)
    return town


def test_office_facts_top_up_homes_that_would_run_dry_in_the_blizzard(office_town):
    facts = office.plan_facts(office_town, [2.0] * 7)
    assert facts["storm_days"] == [office_town.day + 2, office_town.day + 3, office_town.day + 4]
    rows = {r["home_id"]: r for r in office_town.home_statuses()}
    for home_id in facts["top_up_before_storm"]:
        assert rows[home_id]["days_left"] < 2 + 3 + 1.5
    assert office_town.featured_home().home_id is not None and "Blizzard" in office.template_note(facts)


def test_office_plan_fits_the_truck_stops_and_names_who_will_run_short(office_town):
    facts = office.plan_facts(office_town, [20.0] * 7)
    assert len(facts["top_up_before_storm"]) <= facts["stops_before_storm"]
    assert len(facts["pump_out_before_storm"]) <= facts["pump_outs_before_storm"]
    rows = {r["home_id"]: r for r in office_town.home_statuses()}
    chosen = [rows[h]["days_left"] for h in facts["top_up_before_storm"]]
    assert chosen == sorted(chosen)  # the most urgent homes get the stops
    assert facts["peak_homes"] <= len(office_town.homes)  # a forecast can't exceed the town


def test_the_llm_note_is_used_only_when_every_fact_survives(office_town):
    facts = office.plan_facts(office_town, [2.0] * 7)
    template = office.template_note(facts)
    faithful = office.run_office_plan(facts, writer=lambda text: "Hello team.\n" + template)
    assert faithful["source"] == "llm" and faithful["note"].startswith("Hello team.")
    dropped = office.run_office_plan(facts, writer=lambda text: "Hello team. A storm is coming.")
    assert dropped["source"] == "template" and dropped["note"] == template
    invented = office.run_office_plan(facts, writer=lambda text: template + " Also send 99 trucks.")
    assert invented["source"] == "template"


def test_the_check_rejects_a_note_that_moves_homes_between_lists(office_town):
    facts = office.plan_facts(office_town, [2.0] * 7)
    facts["top_up_before_storm"], facts["pump_out_before_storm"] = ["home-01", "home-02"], ["home-03"]
    template = office.template_note(facts)
    swapped = template.replace("top up first: home-01, home-02", "top up first: home-03").replace(
        "pump out first: home-03", "pump out first: home-01, home-02")
    assert swapped != template and not office.faithful(swapped, template)
    assert office.faithful("Hi all.\n\n" + template.replace("Blizzard:", "**Blizzard:**"), template)


def test_a_writer_that_fails_falls_back_to_the_template(office_town):
    facts = office.plan_facts(office_town, [2.0] * 7)

    def broken(text):
        raise ConnectionError("ollama is not running")

    plan = office.run_office_plan(facts, writer=broken)
    assert plan["source"] == "template" and "ollama is not running" in plan["writer_error"]


# --------------------------------------------------------------------------- #
# inbox: residents' messages triaged by a local classifier (Laya); tests use fakes
# --------------------------------------------------------------------------- #

def fake_classifier(probabilities):
    return lambda text: {"probabilities": probabilities, "model": "fake"}


def test_a_health_report_comes_first_even_when_another_need_scores_higher():
    # Laya scored "my kids have diarrhea after drinking the tap water" 0.95 for water delivery
    # and 0.93 for bad water: every need above the threshold counts, and health goes first.
    app = inbox.build_inbox_app(fake_classifier({"water_delivery": 0.95, "water_quality": 0.93, "pump_out": 0.2,
                                                 "device": 0.1, "emergency": 0.3}))
    decision = inbox.triage(app, "My kids have diarrhea after drinking the tap water.", "home-01")
    assert decision["route"] == "water_quality" and decision["urgent"]
    assert decision["needs"] == ["water_quality", "water_delivery"]


def test_the_strongest_need_routes_and_a_weak_health_signal_still_raises_an_alert():
    # Laya on "the toilet is backing up": sewage 0.80, bad water 0.56. Send the sewage truck, but
    # don't drop the possible health report: mark it urgent and tell the water office too.
    app = inbox.build_inbox_app(fake_classifier({"pump_out": 0.80, "water_quality": 0.56, "water_delivery": 0.1,
                                                 "device": 0.4, "emergency": 0.2}))
    decision = inbox.triage(app, "The toilet is backing up!", "home-01")
    assert decision["route"] == "pump_out" and decision["urgent"]
    assert "water_quality" in decision["needs"] and "health" in decision["action"]
    device = inbox.build_inbox_app(fake_classifier({"device": 0.81, "pump_out": 0.64, "water_delivery": 0.63,
                                                    "water_quality": 0.37, "emergency": 0.1}))
    assert inbox.triage(device, "The screen says needs service.", "home-01")["route"] == "device"


def test_an_unclear_message_goes_to_a_person():
    app = inbox.build_inbox_app(fake_classifier({k: 0.2 for k in inbox.NEEDS}))
    decision = inbox.triage(app, "The light on the box in the kitchen is grey.", "home-01")
    assert decision["route"] == "person" and decision["needs"] == [] and "person" in decision["action"]


def test_a_classifier_failure_sends_the_message_to_a_person():
    def broken(text):
        raise OSError("model not downloaded")

    decision = inbox.triage(inbox.build_inbox_app(broken), "Our tank is empty", "home-01")
    assert decision["route"] == "person" and "model not downloaded" in decision["error"]


def test_a_family_asking_for_water_gets_a_truck_even_when_the_sensor_says_full(town):
    town.deliver_now()
    town.advance(1.0)  # the device sees the full tank on its next readings
    home_id = town.featured_home().home_id
    plan = {r["home_id"]: r for r in town.snapshot()["fleet"]}
    assert plan[home_id]["can_skip"]
    town.record_request(home_id, ["water_delivery"])
    plan = {r["home_id"]: r for r in town.snapshot()["fleet"]}
    assert not plan[home_id]["can_skip"] and plan[home_id]["tier"] <= 1 and "asked" in plan[home_id]["why"]
    town.deliver_now()  # the delivery answers the request
    assert town.home_statuses()[0]["requests"] == []


def test_a_reported_sewage_backup_goes_to_the_top_of_the_pump_out_plan(town):
    town.fresh_sewage_tank()
    home_id = town.featured_home().home_id
    town.record_request(home_id, ["pump_out"])
    first = town.snapshot()["pump_outs"][0]
    assert first["home_id"] == home_id and first["tier"] == 0 and "reported" in first["why"]


def test_the_inbox_is_part_of_the_snapshot(town):
    decision = {"text": "Our tank is empty", "home_id": "home-01", "route": "water_delivery", "needs": ["water_delivery"],
                "urgent": False, "action": "Added to the water truck plan.", "probabilities": {}, "model": "fake",
                "error": None}
    town.add_message(decision)
    assert town.snapshot()["inbox"][0]["text"] == "Our tank is empty"


# --------------------------------------------------------------------------- #
# backtest and fleet
# --------------------------------------------------------------------------- #

def test_backtest_reports_warning_lead_times_deterministically():
    first = backtest.run(season="january", days=20, seed=11, policy="fixed")
    second = backtest.run(season="january", days=20, seed=11, policy="fixed")
    assert first == second
    assert first["runouts"] > 0
    assert first["runouts_warned_24h"] / first["runouts"] >= 0.8
    assert 0.0 <= first["share_of_time_not_protected"] <= 1.0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_priority_dispatch_leaves_homes_dry_for_fewer_hours_than_a_fixed_loop(seed):
    # Same trucks, same storms. Counting run-out *events* hides the harm when trucks are
    # short (every home runs dry sometimes); hours without water is what dispatch changes.
    fixed = backtest.run(season="january", days=20, seed=seed, policy="fixed")
    priority = backtest.run(season="january", days=20, seed=seed, policy="priority")
    assert priority["visits"] <= fixed["visits"]
    assert priority["dry_home_hours"] < fixed["dry_home_hours"]
    assert priority["litres_per_visit"] >= fixed["litres_per_visit"]


def test_a_top_up_cannot_restore_chlorine_in_a_mostly_full_tank(town):
    town.drain(pct=90)
    town.vitamin_c()
    town.deliver_now(fc_fill=0.6)  # only a tenth of the tank is new water
    town.advance(1.5)
    assert town.featured_status()["state"] == "boil"
    town.drain(pct=10)
    town.deliver_now(fc_fill=0.6)  # now nine tenths is new water
    town.advance(1.5)
    assert town.featured_status()["fc_mg_l"] >= rules.PROTECTED_MG_L


def test_low_protection_in_a_full_tank_is_a_tank_check_not_a_truck_stop():
    full = {"home_id": "f", "state": "boil", "level_pct": 90.0, "days_left": 3.0, "runout_risk": 0.0, "people": 4,
            "reasons": ["no_chlorine_protection"]}
    low = {**full, "home_id": "l", "level_pct": 30.0, "days_left": 1.8}
    fading = {**full, "home_id": "g", "state": "check", "reasons": ["protection_fading"]}
    plan = {p["home_id"]: p for p in fleet.delivery_priority([full, low, fading])}
    assert plan["l"]["tier"] == 2 and not plan["l"]["can_skip"]
    assert plan["f"]["can_skip"] and "check the tank" in plan["f"]["why"]
    # normal fading in a full tank is not a tank problem, just not a stop today
    assert plan["g"]["can_skip"] and "check the tank" not in plan["g"]["why"]


def test_cloudy_water_in_a_full_tank_asks_for_a_tank_check():
    home = {"home_id": "m", "state": "boil", "level_pct": 95.0, "days_left": 3.5, "runout_risk": 0.0, "people": 5,
            "reasons": ["water_cloudy"]}
    plan = fleet.delivery_priority([home])[0]
    assert plan["can_skip"] and "check the tank" in plan["why"]


def test_truck_plan_explains_a_low_water_warning_without_a_misleading_risk():
    home = {"home_id": "w", "state": "check", "level_pct": 30.0, "days_left": 1.2, "runout_risk": 0.0, "people": 5,
            "reasons": ["water_running_low"]}
    why = fleet.delivery_priority([home])[0]["why"]
    assert "0%" not in why and "day and a half" in why


def test_trucks_go_by_days_of_water_not_by_tank_percent_or_a_stale_risk():
    # From a real plan: homes with 3.6-4.2 days got trucks (risk from long past gaps) while a
    # home at 70% full with 8 people, 2.1 days of water, was skipped as "full enough".
    def home(home_id, level, days, risk, people, state="check"):
        return {"home_id": home_id, "state": state, "level_pct": level, "days_left": days, "runout_risk": risk,
                "people": people, "reasons": []}
    homes = [home("h12", 76.0, 3.6, 0.5, 3), home("h06", 54.0, 4.2, 0.75, 2), home("h04", 55.0, 1.6, 0.0, 5),
             home("h03", 62.0, 2.1, 0.0, 4), home("h05", 70.0, 2.1, 0.0, 8, "protected")]
    plan = {p["home_id"]: p for p in fleet.delivery_priority(homes)}
    assert plan["h12"]["can_skip"] and plan["h06"]["can_skip"]
    assert not plan["h05"]["can_skip"] and not plan["h03"]["can_skip"]
    order = [p["home_id"] for p in fleet.delivery_priority(homes) if not p["can_skip"]]
    assert order == ["h04", "h05", "h03"]  # fewest days first, then the larger household


def test_delivery_priority_puts_the_most_urgent_home_first():
    homes = [
        {"home_id": "a", "state": "protected", "level_pct": 80.0, "days_left": 3.0, "runout_risk": 0.0,
         "people": 4, "reasons": []},
        {"home_id": "b", "state": "boil", "level_pct": 3.0, "days_left": 0.1, "runout_risk": 1.0, "people": 6,
         "reasons": ["tank_nearly_empty"]},
        {"home_id": "c", "state": "check", "level_pct": 30.0, "days_left": 0.8, "runout_risk": 0.6, "people": 5,
         "reasons": ["water_running_low"]},
    ]
    plan = fleet.delivery_priority(homes)
    assert [p["home_id"] for p in plan] == ["b", "c", "a"]
    assert plan[-1]["can_skip"] is True
    assert plan[0]["why"]


# --------------------------------------------------------------------------- #
# server and UI
# --------------------------------------------------------------------------- #

@pytest.fixture
def server():
    srv = make_server(port=0, community=simulate.Community("january", seed=3, warmup_days=5), use_ai=False)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read()


def test_api_state_and_actions(server):
    status, body = _get(server + "/api/state")
    state = json.loads(body)
    assert status == 200 and state["featured"]["status"]["state"] in rules.SEVERITY
    assert len(state["fleet"]) == len(state["homes"])
    req = urllib.request.Request(server + "/api/action", data=json.dumps({"action": "vitamin_c"}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
    bad = urllib.request.Request(server + "/api/action", data=b'{"action": "rm -rf"}', method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(bad, timeout=10)
    assert err.value.code == 400


def test_api_office_plan_without_ai_uses_the_baseline_and_template(server):
    req = urllib.request.Request(server + "/api/plan", data=b"{}", method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        plan = json.loads(resp.read())
    assert plan["model"].startswith("baseline") and plan["source"] == "template"
    assert len(plan["predicted"]) == 7 and plan["note"].startswith("Water office plan")


def test_api_message_without_ai_goes_to_a_person(server):
    req = urllib.request.Request(server + "/api/message", data=json.dumps({"text": "Our tank is empty"}).encode(),
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read())
    assert out["decision"]["route"] == "person" and out["snapshot"]["inbox"][0]["text"] == "Our tank is empty"
    bad = urllib.request.Request(server + "/api/message", data=json.dumps({"text": ""}).encode(), method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(bad, timeout=10)
    assert err.value.code == 400


@pytest.mark.parametrize("length", ["-5", "abc"])
def test_bad_content_length_is_rejected_not_hung(server, length):
    import http.client

    host, port = server.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    conn.putrequest("POST", "/api/action")
    conn.putheader("Content-Length", length)
    conn.endheaders()
    assert conn.getresponse().status == 400
    conn.close()


def test_static_files_are_whitelisted(server):
    status, body = _get(server + "/")
    assert status == 200 and b"TankLight" in body
    for path in ("/../app.py", "/tanklight/rules.py", "/%2e%2e/app.py"):
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(server + path, timeout=10)
        assert err.value.code == 404


def test_every_ui_module_is_served():
    from tanklight.server import STATIC

    modules = {p.stem for p in (EXAMPLE / "ui" / "src").glob("*.ts")}
    assert {f"/dist/{m}.js" for m in modules} <= set(STATIC)


def test_every_tour_step_points_at_an_element_on_the_page():
    import re

    tour = (EXAMPLE / "ui" / "src" / "tour.ts").read_text()
    page = (EXAMPLE / "ui" / "index.html").read_text()
    targets = re.findall(r'target: "([\w-]+)"', tour)
    assert len(targets) >= 10
    missing = [t for t in targets if f'id="{t}"' not in page]
    assert not missing, missing
    assert "sewage" in targets and "kitchen-toggle" in targets and "inbox-card" in targets
    for control in ("tour", "tour-start", "tour-title", "tour-body", "tour-count", "tour-progress", "tour-back",
                    "tour-next", "tour-show", "tour-close"):
        assert f'id="{control}"' in page, control


def test_kitchen_mode_hides_the_demo_and_keeps_the_light():
    page = (EXAMPLE / "ui" / "index.html").read_text()
    css = (EXAMPLE / "ui" / "styles.css").read_text()
    assert 'id="kitchen-toggle"' in page and 'aria-pressed="false"' in page
    assert "body.kitchen #controls" in css and "body.kitchen #device" in css


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed to build the TypeScript UI")
def test_typescript_ui_builds(tmp_path):
    result = subprocess.run(["node", str(EXAMPLE / "ui" / "build.mjs"), str(tmp_path)],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    main = (tmp_path / "main.js").read_text()
    assert "import" in main and ": HTMLElement" not in main
