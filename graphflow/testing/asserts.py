"""Custom assertions for verifying workflow execution."""

from __future__ import annotations

from typing import Any

from graphflow.events import NodeCompleted, NodeStarted, StreamEvent


def assert_node_executed(events: list[StreamEvent], node_name: str) -> None:
    """Asserts that a given node started and completed during the run."""
    started = any(isinstance(e, NodeStarted) and e.node_name == node_name for e in events)
    completed = any(isinstance(e, NodeCompleted) and e.node_name == node_name for e in events)
    assert started, f"Node '{node_name}' was not started during execution."
    assert completed, f"Node '{node_name}' did not complete during execution."


def assert_state_contains(state: dict[str, Any], key: str, expected_val: Any = None) -> None:
    """Asserts that the state contains the specified key and optionally matches expected value."""
    assert key in state, f"Key '{key}' not found in state: {list(state.keys())}"
    if expected_val is not None:
        assert state[key] == expected_val, f"Expected state['{key}'] == {expected_val}, got {state[key]}"
