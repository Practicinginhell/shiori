"""Example 8: TankLight, a household water protection tracker for truck-served tanks.

    python examples/08_tanklight/app.py demo
        Offline walk-through in the terminal: delivery, vitamin C, murky water, a pulled
        probe and a blizzard, with the light the device would show after each.

    python examples/08_tanklight/app.py backtest [--season january] [--days 30] [--seed 11]
        Simulated community: warnings before run-outs and chlorine loss, and a fixed truck
        loop against TankLight's delivery priority.

    python examples/08_tanklight/app.py plan [--season january] [--no-ai]
        The water office's 7-day plan: TimesFM (local, MLX) forecasts how many homes will need
        urgent water, checked against a simple baseline; a small local LLM (Ollama) rewords the
        plan, and a check keeps its note only if every home and number survived.

    python examples/08_tanklight/app.py inbox [--no-ai]
        Residents' messages triaged by Laya, a small local classifier: water truck, sewage truck,
        bad water (urgent), device, or a person reads it. Scores it on labelled examples.

    python examples/08_tanklight/app.py serve [--port 8008]
        Builds the TypeScript UI (needs Node >= 22.13) and serves it at http://127.0.0.1:8008.

All data is SIMULATED; see tanklight/simulate.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))
sys.path.insert(0, HERE)

from tanklight import backtest, forecast, inbox, office  # noqa: E402
from tanklight.server import make_server  # noqa: E402
from tanklight.simulate import Community  # noqa: E402

LIGHTS = {"protected": "🟢 ✓ Protected", "check": "🟡 ! Check", "boil": "🔴 ✕ Boil first", "service": "⚪ 🔧 Service"}


def show(town: Community, label: str) -> None:
    s = town.featured_status()
    detail = []
    if s["fc_mg_l"] is not None:
        detail.append(f"chlorine ~{s['fc_mg_l']:.2f} mg/L")
    if s["hours_protected"] is not None:
        detail.append(f"protected {s['hours_protected']:.0f} h")
    if s["days_left"] is not None:
        detail.append(f"water {s['days_left']:.1f} days")
    reasons = ", ".join(s["reasons"]) or "all clear"
    print(f"{town.snapshot()['clock']:<16} {label:<34} {LIGHTS[s['state']]:<16} {'; '.join(detail)}  [{reasons}]")


def fresh_tank(season: str = "january") -> Community:
    """A simulated town whose featured tank has just had a delivery into a nearly empty tank."""
    town = Community(season, seed=3, warmup_days=14)
    town.drain(pct=10)
    town.deliver_now(fc_fill=0.6)
    town.advance(2.0)
    return town


def demo() -> None:
    print("TankLight demo (SIMULATED data). Each scenario starts from its own fresh tank, like separate jars.\n")
    town = fresh_tank()
    print(f"The featured home runs on the {town.featured_monitor().engine_name} household pipeline.\n")
    show(town, "Truck filled a nearly empty tank")

    town.vitamin_c()
    town.advance(1.5)
    show(town, "Vitamin C tablet (no chlorine)")
    town.deliver_now(fc_fill=0.6)
    town.advance(1.5)
    show(town, "Top-up of that mostly-full tank")

    town = fresh_tank()
    town.make_murky()
    town.advance(1.5)
    show(town, "Clay stirred in (murky)")

    town = fresh_tank()
    town.pull_probe()
    town.advance(0.5)
    show(town, "Probe pulled out")
    town.attach_probe()
    town.advance(2.0)
    show(town, "Probe back in")

    town = fresh_tank()
    town.storm(days=6)
    for day in range(1, 5):
        town.advance(24.0)
        show(town, f"Blizzard day {day}, no trucks")

    for season in ("january", "july"):
        town = fresh_tank(season)
        show(town, f"Same delivery in {season.capitalize()}")

    print("\nNever green while a sensor is unsure; warns before the tank runs dry; and a top-up cannot")
    print("fix a tank that is mostly old water (that needs a tank check, so the truck plan says so).")


def run_backtest(season: str, days: int, seed: int) -> None:
    print(f"Backtest, SIMULATED community of 12 homes, {season}, {days} days, seed {seed}\n")
    for r in backtest.compare(season, days, seed):
        warned = f"{r['runouts_warned_24h']}/{r['runouts']}" if r["runouts"] else "n/a"
        lost = f"{r['losses_warned_6h']}/{r['losses']}" if r["losses"] else "n/a"
        print(f"  {r['policy']:<9} dry home-hours {r['dry_home_hours']:>6.0f} | run-outs warned 24 h ahead {warned:<7} "
              f"| chlorine losses warned 6 h ahead {lost:<7} | light not green {r['share_of_time_not_protected']:.0%} "
              f"| {r['visits']} stops, {r['litres_per_visit']:.0f} L per stop")


def plan_town(season: str, days: int, seed: int, storm_in: int) -> Community:
    """A town with `days` days of history, at 09:00, with a 3-day blizzard forecast `storm_in` days out."""
    town = Community(season, seed=seed)
    town.advance(days * 24.0 + 9.0)
    if storm_in:
        town.storm_on(town.day + storm_in, days=3)
    return town


def run_plan(season: str, days: int, seed: int, storm_in: int, use_ai: bool, llm: str) -> None:
    town = plan_town(season, days, seed, storm_in)
    series, storms = forecast.town_inputs(town)
    print(f"Water office plan, SIMULATED town of 12 homes, {season}, {len(series)} days of history\n")
    week = forecast.forecast_week(series, storms, len(town.homes), use_ai)
    if use_ai and not week["timesfm_installed"]:
        print("TimesFM is not installed (pip install \"timesfm[mlx]\"); using the baseline only.\n")
    if week["model_error"]:
        print(f"TimesFM failed ({week['model_error']}); using the baseline.\n")
    print("Forecast check on the last 4 weeks (mean error, homes per day; lower is better):")
    for name, mae in week["scores"].items():
        print(f"  {name:<26} {mae:.2f}")
    print(f"\nNext 7 days with {week['model']}: " + ", ".join(f"{v:.1f}" for v in week["predicted"])
          + " homes needing urgent water\n")

    plan = office.run_office_plan(office.plan_facts(town, week["predicted"]),
                                  office.ollama_writer(llm) if use_ai else None)
    source = {"llm": f"reworded by {llm}, every home and number checked",
              "template": "template" + (f"; the {llm} draft failed the check" if plan["draft"] else "")}[plan["source"]]
    if plan["writer_error"]:
        source += f"; {llm} unavailable ({plan['writer_error']})"
    print(f"Note for the water office ({source}):\n")
    print(plan["note"])
    if plan["draft"] and plan["source"] == "template":
        print(f"\n(Rejected {llm} draft, for comparison:)\n{plan['draft']}")


def run_inbox(use_ai: bool) -> None:
    classifier = inbox.LayaClassifier() if use_ai and inbox.laya_available() else None
    if use_ai and classifier is None:
        print("Laya is not installed (pip install laya); every message goes to a person.\n")
    app = inbox.build_inbox_app(classifier)
    print("Messages from homes, triaged locally (SIMULATED examples)\n")
    right = safe = 0
    for text, expected in inbox.EXAMPLES:
        d = inbox.triage(app, text, "home-01")
        ok = d["route"] == expected
        right += ok
        safe += ok or d["route"] == "person"  # unsure: a person reads it, slower but never wrong
        mark = "ok " if ok else ("-> " if d["route"] == "person" else "XX ")
        urgent = " URGENT" if d["urgent"] else ""
        print(f"  {mark} {d['route']:<15}{urgent:<7} {text[:58]}")
    print(f"\nRight: {right}/{len(inbox.EXAMPLES)}. Right or left for a person: {safe}/{len(inbox.EXAMPLES)}.")
    print("ok = right; -> = sent to a person (safe); XX = sent to the wrong place.")


def build_ui() -> None:
    node = shutil.which("node")
    if node is None:
        sys.exit("serve needs Node.js >= 22.13 to build the TypeScript UI (https://nodejs.org)")
    subprocess.run([node, os.path.join(HERE, "ui", "build.mjs")], check=True)


def serve(port: int) -> None:
    build_ui()
    print("Warming up the simulated community (70 days of history)...")
    server = make_server(port=port)
    print(f"TankLight demo at http://127.0.0.1:{server.server_address[1]}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TankLight demo (simulated data)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="terminal walk-through")
    bt = sub.add_parser("backtest", help="warnings and truck dispatch on a simulated community")
    bt.add_argument("--season", choices=["january", "july"], default="january")
    bt.add_argument("--days", type=int, default=30)
    bt.add_argument("--seed", type=int, default=11)
    pl = sub.add_parser("plan", help="the water office's 7-day plan (local TimesFM + small LLM)")
    pl.add_argument("--season", choices=["january", "july"], default="january")
    pl.add_argument("--days", type=int, default=70, help="days of history before today")
    pl.add_argument("--seed", type=int, default=11)
    pl.add_argument("--storm-in", type=int, default=2, help="days until a forecast 3-day blizzard (0: none)")
    pl.add_argument("--llm", default=office.DEFAULT_LLM, help="Ollama model that rewords the plan")
    pl.add_argument("--no-ai", action="store_true", help="baseline forecast and template note only")
    ib = sub.add_parser("inbox", help="residents' messages triaged by a local classifier (Laya)")
    ib.add_argument("--no-ai", action="store_true", help="no model: every message goes to a person")
    sv = sub.add_parser("serve", help="build and serve the TypeScript UI")
    sv.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    if args.command == "backtest":
        run_backtest(args.season, args.days, args.seed)
    elif args.command == "plan":
        run_plan(args.season, args.days, args.seed, args.storm_in, not args.no_ai, args.llm)
    elif args.command == "inbox":
        run_inbox(not args.no_ai)
    elif args.command == "serve":
        serve(args.port)
    else:
        demo()


if __name__ == "__main__":
    main()
