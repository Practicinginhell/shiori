"""A small local server for the TankLight UI: a JSON API over the simulated community.

    GET  /api/state             the snapshot the UI renders
    POST /api/action {action}   one demo action (whitelisted), returns the new snapshot
    POST /api/plan              the water office's 7-day plan (forecast + note; see office.py)
    POST /api/message {text}    a resident's message, triaged by the local classifier (inbox.py)

Static files are served from a fixed whitelist only, never by joining request paths, so
nothing outside ui/ can be read. It binds to 127.0.0.1 by default: a demo, not a service.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import forecast, inbox, office
from .simulate import POLICIES, SEASONS, Community

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
for _module in ("main", "api", "device", "chart", "fleet", "i18n", "icons", "inbox", "office", "sewage", "tour", "types"):
    STATIC[f"/dist/{_module}.js"] = (f"dist/{_module}.js", "text/javascript; charset=utf-8")
MAX_BODY_BYTES = 10_000
MAX_MESSAGE_CHARS = 500
# Ten weeks of history before the demo starts, so the office forecast has enough to learn from.
DEMO_WARMUP_DAYS = 70


def _hours(value: Any) -> float:
    hours = float(value)
    if not 0.5 <= hours <= 72:
        raise ValueError("hours must be between 0.5 and 72")
    return hours


def _days(value: Any) -> int:
    days = int(value)
    if not 1 <= days <= 10:
        raise ValueError("days must be between 1 and 10")
    return days


def _pct(value: Any) -> float:
    pct = float(value)
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")
    return pct


class DemoState:
    """The community behind a lock (the server is threaded)."""

    def __init__(self, community: Community | None = None):
        self.lock = threading.Lock()
        self.community = community or Community(warmup_days=DEMO_WARMUP_DAYS)

    def act(self, body: dict) -> None:
        action, town = body.get("action"), self.community
        simple = {"deliver_now": town.deliver_now, "vitamin_c": town.vitamin_c, "make_murky": town.make_murky,
                  "pull_probe": town.pull_probe, "attach_probe": town.attach_probe,
                  "pump_out_now": town.pump_out_now,
                  "fresh_sewage_tank": town.fresh_sewage_tank, "heater_off": town.heater_off, "heater_on": town.heater_on}
        if action in simple:
            simple[action]()
        elif action == "advance":
            town.advance(_hours(body.get("hours", 1)))
        elif action == "drain":
            town.drain(_pct(body.get("pct", 10)))
        elif action == "storm":
            town.storm(_days(body.get("days", 3)))
        elif action == "set_policy" and body.get("policy") in POLICIES:
            town.set_policy(body["policy"])
        elif action == "reset" and body.get("season", town.season) in SEASONS:
            self.community = Community(body.get("season", town.season), policy=town.policy, seed=town.seed,
                                       warmup_days=DEMO_WARMUP_DAYS)
        else:
            raise ValueError(f"unknown action {action!r}")


def make_plan(demo: DemoState, use_ai: bool) -> dict:
    """The office plan. The models run outside the lock, so the demo stays usable meanwhile."""
    with demo.lock:
        series, storms = forecast.town_inputs(demo.community)
        n_homes = len(demo.community.homes)
    week = forecast.forecast_week(series, storms, n_homes, use_ai)
    with demo.lock:
        facts = office.plan_facts(demo.community, week["predicted"])
    plan = office.run_office_plan(facts, office.ollama_writer() if use_ai else None)
    return {**week, **plan, "llm": office.DEFAULT_LLM if use_ai else None}


def handle_message(demo: DemoState, app: Any, body: dict) -> dict:
    """Triage one message for the featured home; the model runs outside the demo lock."""
    text = body.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"text must be 1 to {MAX_MESSAGE_CHARS} characters")
    with demo.lock:
        home_id = demo.community.featured_home().home_id
    decision = inbox.triage(app, text.strip(), home_id)
    with demo.lock:
        demo.community.record_request(home_id, decision["needs"])
        demo.community.add_message(decision)
        return {"decision": decision, "snapshot": demo.community.snapshot()}


def make_server(port: int = 8008, community: Community | None = None, host: str = "127.0.0.1",
                use_ai: bool = True) -> ThreadingHTTPServer:
    demo = DemoState(community)
    plan_lock = threading.Lock()  # one plan at a time: the model is loaded once and shared
    inbox_app = inbox.build_inbox_app(inbox.LayaClassifier() if use_ai and inbox.laya_available() else None)
    inbox_lock = threading.Lock()  # one message at a time through the shared classifier

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep test output quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                with demo.lock:
                    self._json(HTTPStatus.OK, demo.community.snapshot())
                return
            entry = STATIC.get(path)
            file = UI_DIR / entry[0] if entry else None
            if file is None or not file.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(HTTPStatus.OK, file.read_bytes(), entry[1])

        def do_POST(self) -> None:
            if self.path == "/api/message":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if not 0 <= length <= MAX_BODY_BYTES:
                        raise ValueError("bad Content-Length")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                    with inbox_lock:
                        self._json(HTTPStatus.OK, handle_message(demo, inbox_app, body))
                except (ValueError, TypeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if self.path == "/api/plan":
                with plan_lock:
                    self._json(HTTPStatus.OK, make_plan(demo, use_ai))
                return
            if self.path != "/api/action":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:  # a negative length would make read() wait for the client to hang up
                self._json(HTTPStatus.BAD_REQUEST, {"error": "bad Content-Length"})
                return
            if length > MAX_BODY_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                with demo.lock:
                    demo.act(body)
                    snapshot = demo.community.snapshot()
            except (ValueError, TypeError) as exc:  # json.JSONDecodeError is a ValueError
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, snapshot)

    return ThreadingHTTPServer((host, port), Handler)
