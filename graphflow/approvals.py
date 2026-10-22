"""Human approval data types and helpers (dependency-free, shared by all layers).

Two kinds of pause use them:

* a workflow approval step (``Workflow.require_approval``), answered with an
  ``ApprovalDecision`` (or bool / dict);
* an agent's tool calls that need approval, answered per call with a
  ``ToolDecision`` (approve, edit the arguments, or reject with feedback), a
  list of them in request order, a ``{"decisions": {tool_call_id: ...}}``
  mapping, or a bool / ``ApprovalDecision`` applied to every pending call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

DecisionType = Literal["approve", "edit", "reject"]
DECISION_TYPES: tuple[str, ...] = ("approve", "edit", "reject")
REJECTED_MESSAGE = "Tool call rejected by human reviewer."
HANDLER_REJECTED_MESSAGE = "Tool call rejected by the approval handler."
REFUSED_MESSAGE = "Tool call refused before running:"


@dataclass
class ApprovalRequest:
    """Surfaced to the caller when execution interrupts for human review."""
    prompt: str
    description: str = ""
    payload: Any = None
    node_name: str = ""
    thread_id: str = ""
    interrupt_id: str = ""

    @property
    def actions(self) -> list[dict[str, Any]]:
        """For tool approvals: the pending tool calls (``tool_call_id``, ``tool``, ``args``, ...)."""
        if isinstance(self.payload, dict):
            return list(self.payload.get("action_requests") or [])
        return []


@dataclass
class ApprovalDecision:
    """Provided by the user to resume an interrupted workflow.

    ``edits`` are applied to the state whether the step is approved or rejected,
    so a reviewer can also record feedback into a state field.
    """
    approved: bool
    feedback: str = ""
    edits: dict[str, Any] = field(default_factory=dict)

    def to_resume_value(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "feedback": self.feedback,
            "edits": self.edits,
        }


@dataclass
class ToolDecision:
    """A reviewer's answer to one pending tool call."""
    type: DecisionType
    args: dict[str, Any] | None = None  # replacement arguments, for "edit"
    feedback: str = ""                  # shown to the model, for "reject"

    @classmethod
    def approve(cls) -> ToolDecision:
        return cls("approve")

    @classmethod
    def edit(cls, args: dict[str, Any]) -> ToolDecision:
        return cls("edit", args=dict(args))

    @classmethod
    def reject(cls, feedback: str = "") -> ToolDecision:
        return cls("reject", feedback=feedback)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.args is not None:
            data["args"] = self.args
        if self.feedback:
            data["feedback"] = self.feedback
        return data


def to_resume_value(decision: Any) -> Any:
    """Normalizes the accepted decision types into plain data for ``Command(resume=...)``."""
    if isinstance(decision, ApprovalDecision):
        return decision.to_resume_value()
    if isinstance(decision, ToolDecision):
        return decision.to_dict()
    if isinstance(decision, bool):
        return {"approved": decision, "feedback": "", "edits": {}}
    if isinstance(decision, dict):
        inner = decision.get("decisions")
        if isinstance(inner, dict):
            return {**decision, "decisions": {k: to_resume_value(v) for k, v in inner.items()}}
        if isinstance(inner, (list, tuple)):
            return {**decision, "decisions": [to_resume_value(v) for v in inner]}
        return decision
    if isinstance(decision, (list, tuple)):
        return {"decisions": [to_resume_value(d) for d in decision]}
    return {"approved": True, "value": decision}


def decision_fields(value: Any) -> tuple[bool, str, dict[str, Any]]:
    """(approved, feedback, edits) of a resume value answering a workflow approval step."""
    if isinstance(value, bool):
        return value, "", {}
    if isinstance(value, dict):
        if "type" in value:
            return value["type"] != "reject", str(value.get("feedback") or ""), {}
        return bool(value.get("approved", True)), str(value.get("feedback") or ""), dict(value.get("edits") or {})
    return True, "", {}


def is_approved(value: Any) -> bool:
    return decision_fields(value)[0]


# ---------------------------------------------------------------------- #
# Tool-call decisions
# ---------------------------------------------------------------------- #

class DecisionError(ValueError):
    """A reviewer's answer that does not fit the pending tool calls (the reviewer is asked again)."""


def _one(value: Any) -> dict[str, Any]:
    """One decision as ``{"type", "args"?, "feedback"?}``."""
    if isinstance(value, ToolDecision):
        return value.to_dict()
    if isinstance(value, ApprovalDecision):
        value = value.to_resume_value()
    if isinstance(value, bool):
        return {"type": "approve" if value else "reject"}
    if isinstance(value, Mapping):
        if "type" in value:
            kind = value["type"]
            if kind not in DECISION_TYPES:
                raise DecisionError(f"unknown decision type {kind!r} (expected one of {', '.join(DECISION_TYPES)})")
            out: dict[str, Any] = {"type": kind}
            if kind == "edit":
                args = value.get("args")
                if not isinstance(args, Mapping):
                    raise DecisionError("an 'edit' decision needs the new arguments as a dict under 'args'")
                out["args"] = dict(args)
            if value.get("feedback"):
                out["feedback"] = str(value["feedback"])
            return out
        if "approved" in value:
            out = {"type": "approve" if value["approved"] else "reject"}
            if value.get("feedback"):
                out["feedback"] = str(value["feedback"])
            return out
    raise DecisionError(f"cannot interpret {value!r} as a tool decision")


def tool_decisions(value: Any, call_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Maps each pending tool call id to its decision; raises DecisionError when `value` doesn't fit."""
    ids = list(call_ids)
    if isinstance(value, Mapping) and "decisions" in value:
        value = value["decisions"]
        if isinstance(value, Mapping):
            missing = [i for i in ids if i not in value]
            extra = [k for k in value if k not in ids]
            if missing or extra:
                raise DecisionError(
                    f"decisions must cover exactly the pending tool calls {ids} "
                    f"(missing: {missing}, unknown: {extra})"
                )
            return {i: _one(value[i]) for i in ids}
    if isinstance(value, (list, tuple)):
        if len(value) != len(ids):
            raise DecisionError(f"expected {len(ids)} decisions (one per pending tool call, in order), got {len(value)}")
        return {i: _one(v) for i, v in zip(ids, value, strict=True)}
    single = _one(value)
    if single["type"] == "edit" and len(ids) > 1:
        raise DecisionError(f"an 'edit' decision must name its tool call; {len(ids)} calls are pending")
    return {i: dict(single) for i in ids}


def rejection_message(feedback: str = "", source: str = "reviewer") -> str:
    """What the model is told about a rejected call; `source` is "reviewer", "handler" or "check"."""
    if source == "check":
        return f"{REFUSED_MESSAGE} {feedback}"
    base = HANDLER_REJECTED_MESSAGE if source == "handler" else REJECTED_MESSAGE
    return f"{base} Feedback: {feedback}" if feedback else base
