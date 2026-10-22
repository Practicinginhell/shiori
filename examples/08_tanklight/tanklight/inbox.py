"""The water office inbox: residents' messages, triaged by a small local classifier.

    message ──▶ classify (Laya, local) ──▶ decide ──▶ route + action

Laya (convaiinnovations/laya) answers yes/no questions with calibrated probabilities instead
of generating text. Each need gets its own question, and every need above the threshold counts.
The strongest need routes the message, except that a strong health report (bad water) always
leads; a weaker one still marks the message urgent and alerts the water office. A message that
matches nothing, or a model that fails, goes to a person: the classifier only sorts the inbox,
it never closes a message on its own.

It is a graphflow workflow: `classify` uses graphflow's per-node timeout and on_error, so a
missing or slow model routes to a person instead of failing. Everything here is SIMULATED.
"""

from __future__ import annotations

from collections.abc import Callable

from graphflow import Application, State, Workflow

# One yes/no question per need, most serious first (the order breaks ties).
NEEDS = {
    "water_quality": "Does the tap water smell, taste or look bad, or did someone get sick from it?",
    "pump_out": "Is the toilet, drain or sewage tank full, overflowing or backing up?",
    "water_delivery": "Is the household out of drinking water or is their water tank empty or low?",
    "device": "Is the TankLight light, screen or sensor broken, grey or showing needs service?",
}
EMERGENCY = "Is someone's health or safety at risk right now?"
THRESHOLD = 0.5
HEALTH_LEADS = 0.8  # a bad-water report this strong routes the message whatever else scores higher
CLASSIFY_TIMEOUT_S = 60.0  # the first call loads the model (a few seconds per language)

ACTIONS = {
    "water_quality": "Urgent: sent to the water office to check the tank and call the family today.",
    "pump_out": "Added to the sewage truck plan as urgent.",
    "water_delivery": "Added to the water truck plan.",
    "device": "Added to the water keeper's service list.",
    "person": "Not sure what this needs, so a person at the water office reads it.",
}

# Labelled examples: the `inbox` command scores the classifier on them. "person" means it
# should be left for a person (a question, a thank-you).
EXAMPLES = [
    ("Our water tank is almost empty and we have a baby at home.", "water_delivery"),
    ("The toilet is backing up, sewage is coming into the bathroom!", "pump_out"),
    ("The water smells like rotten eggs since yesterday.", "water_quality"),
    ("Le réservoir d'eaux usées est plein, pouvez-vous venir vider?", "pump_out"),
    ("The light on the box in the kitchen is grey.", "device"),
    ("We ran out of water this morning, no truck for 3 days.", "water_delivery"),
    ("My kids have diarrhea after drinking the tap water.", "water_quality"),
    ("When is the next delivery day for our street?", "person"),
    ("Il n'y a plus d'eau dans le réservoir.", "water_delivery"),
    ("The TankLight screen keeps saying needs service.", "device"),
    ("Sewage truck didn't come, tank is overflowing outside.", "pump_out"),
    ("Thanks for coming so quickly yesterday!", "person"),
]

# A classifier takes the message and returns {"probabilities": {question id: P(yes)}, "model": name}.
Classifier = Callable[[str], dict]


class LayaClassifier:
    """Laya's Router, loaded on first use and shared. Routes non-English text to its multilingual model."""

    def __init__(self) -> None:
        self._router = None

    def __call__(self, text: str) -> dict:
        if self._router is None:
            from laya import Router

            self._router = Router()
        questions = {k: {"type": "noul", "instructions": q} for k, q in {**NEEDS, "emergency": EMERGENCY}.items()}
        result = self._router.predict(text, questions)
        return {"probabilities": {k: round(float(a["noul"]), 3) for k, a in result["answers"].items()},
                "model": f"laya {result.get('routing', {}).get('model', '')}".strip()}


def laya_available() -> bool:
    try:
        import laya  # noqa: F401
    except ImportError:
        return False
    return True


def decide(probabilities: dict) -> dict:
    """Every need above the threshold; the strongest routes, unless a strong health report leads."""
    needs = [k for k in NEEDS if probabilities.get(k, 0.0) >= THRESHOLD]
    if not needs:
        route = "person"
    elif probabilities.get("water_quality", 0.0) >= HEALTH_LEADS:
        route = "water_quality"
    else:
        route = max(needs, key=lambda k: probabilities[k])  # ties: the more serious need (NEEDS order)
    health = "water_quality" in needs
    urgent = health or probabilities.get("emergency", 0.0) >= THRESHOLD
    action = ACTIONS[route]
    if health and route != "water_quality":
        action += " Also flagged to the water office as a possible health report."
    return {"needs": needs, "route": route, "urgent": urgent, "action": action}


class InboxState(State):
    text: str | None = None
    home_id: str | None = None
    probabilities: dict | None = None
    model: str | None = None
    error: str | None = None
    decision: dict | None = None


def build_inbox_app(classifier: Classifier | None) -> Application:
    """Compile once per classifier; `None` (no model installed) sends every message to a person."""
    def classify(state: dict) -> dict:
        if classifier is None:
            return {"probabilities": {}, "model": None}
        out = classifier(state["text"])
        return {"probabilities": out["probabilities"], "model": out.get("model")}

    def classify_failed(state: dict, error: Exception) -> dict:
        return {"probabilities": {}, "model": None, "error": f"{type(error).__name__}: {error}"}

    def route(state: dict) -> dict:
        return {"decision": decide(state.get("probabilities") or {})}

    wf = Workflow("inbox", state_schema=InboxState)
    wf.then(classify, timeout=CLASSIFY_TIMEOUT_S, on_error=classify_failed).then(route)
    return Application("tanklight-inbox").register(wf)


def triage(app: Application, text: str, home_id: str) -> dict:
    """Run the inbox workflow on one message: the decision plus what the model said."""
    out = app.run({"text": text, "home_id": home_id})
    return {"text": text, "home_id": home_id, **out["decision"], "probabilities": out.get("probabilities") or {},
            "model": out.get("model"), "error": out.get("error")}
