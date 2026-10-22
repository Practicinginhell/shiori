"""Graphflow standard streaming event models.

Clean, high-level event representations that abstract away low-level
LangGraph/Pregel stream event formats into structured, intuitive event objects.

Every run emits ``RunStarted`` first and exactly one terminal event:
``RunCompleted``, ``RunInterrupted`` or ``RunFailed``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """Base class for all stream events."""
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""


@dataclass
class RunStarted(StreamEvent):
    """Emitted when application execution begins."""
    workflow_name: str = ""
    initial_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeStarted(StreamEvent):
    """Emitted when a node begins execution."""
    node_name: str = ""
    input_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelToken(StreamEvent):
    """Emitted when a model generates a token during streaming."""
    token: str = ""
    node_name: str = ""
    agent_name: str = ""


@dataclass
class ToolCalled(StreamEvent):
    """Emitted when an agent invokes a tool."""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    node_name: str = ""
    tool_call_id: str = ""


@dataclass
class ToolCompleted(StreamEvent):
    """Emitted when a tool finishes execution (successfully or not)."""
    tool_name: str = ""
    result: Any = None
    node_name: str = ""
    duration_seconds: float = 0.0
    tool_call_id: str = ""
    error: str | None = None


@dataclass
class StateUpdated(StreamEvent):
    """Emitted when state is modified by a node."""
    node_name: str = ""
    updates: dict[str, Any] = field(default_factory=dict)
    full_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanApprovalRequired(StreamEvent):
    """Emitted when a workflow pauses for human input."""
    prompt: str = ""
    node_name: str = ""
    thread_id: str = ""
    payload: Any = None
    interrupt_id: str = ""


@dataclass
class NodeCompleted(StreamEvent):
    """Emitted when a node completes execution."""
    node_name: str = ""
    duration_seconds: float = 0.0
    output_updates: dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomEvent(StreamEvent):
    """Emitted for arbitrary data written with LangGraph's `get_stream_writer()`."""
    data: Any = None


@dataclass
class RunCompleted(StreamEvent):
    """Emitted when the entire workflow execution succeeds."""
    final_output: dict[str, Any] = field(default_factory=dict)
    total_duration_seconds: float = 0.0


@dataclass
class RunInterrupted(StreamEvent):
    """Emitted when execution pauses waiting for human input (resume to continue)."""
    state: dict[str, Any] = field(default_factory=dict)
    interrupts: list[Any] = field(default_factory=list)
    total_duration_seconds: float = 0.0


@dataclass
class RunFailed(StreamEvent):
    """Emitted when workflow execution fails."""
    error: str = ""
    failed_node: str = ""


def emit_event(event: StreamEvent) -> None:
    """Emits `event` into the stream of the currently running graph.

    Safe to call anywhere: outside a Graphflow/LangGraph run it does nothing.
    Fills in ``node_name`` from the running node when the event has one.
    """
    try:
        from langgraph.config import get_config, get_stream_writer

        config = get_config()
        writer = get_stream_writer()
    except (ImportError, RuntimeError):
        return
    node = (config.get("metadata") or {}).get("langgraph_node", "")
    if hasattr(event, "node_name") and not event.node_name and node:
        event.node_name = node  # type: ignore[attr-defined]
    writer(event)
