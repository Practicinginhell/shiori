"""Tests for custom reducers, State serialization, and advanced state features.

These tests verify:
- Custom callable reducers work end-to-end through compilation and execution
- State serialization to/from JSON for checkpoint portability
- State immutability/snapshot capabilities
- State update validation
"""

import pytest

from graphflow import Application, Field, State, Workflow
from graphflow.core.errors import StateValidationError


def test_custom_callable_reducer_end_to_end():
    """A user-defined custom reducer function should work end-to-end in a compiled workflow."""
    def max_reducer(current, update):
        """Keeps the maximum value."""
        if current is None:
            return update
        return max(current, update)

    class MaxState(State):
        best_score: int = Field(default=0, reducer=max_reducer)
        history: list = Field.reducer("append")

    def step1(state: dict) -> dict:
        return {"best_score": 75, "history": ["step1"]}

    def step2(state: dict) -> dict:
        return {"best_score": 42, "history": ["step2"]}

    def step3(state: dict) -> dict:
        return {"best_score": 91, "history": ["step3"]}

    wf = Workflow("max_wf", state_schema=MaxState)
    wf.then(step1).then(step2).then(step3)

    app = Application("max_app")
    app.register(wf)
    res = app.run({"best_score": 0})

    # max_reducer should keep the highest value across all updates
    assert res["best_score"] == 91
    assert res["history"] == ["step1", "step2", "step3"]


def test_state_to_json_and_from_json():
    """State should support serialization to/from JSON dict for portable checkpoints."""
    class SerializableState(State):
        query: str = ""
        messages: list = Field.reducer("append")
        counter: int = Field.reducer("sum", default=0)

    original = SerializableState(query="test", counter=5)
    original.messages = ["hello", "world"]

    # Serialize
    json_data = original.to_json()
    assert isinstance(json_data, str)

    # Deserialize
    restored = SerializableState.from_json(json_data)
    assert restored.query == "test"
    assert restored.counter == 5
    assert restored.messages == ["hello", "world"]


def test_state_snapshot_returns_immutable_copy():
    """State.snapshot() should return a frozen copy that doesn't mutate with the original."""
    class SnapshotState(State):
        items: list = Field.reducer("append")
        counter: int = 0

    state = SnapshotState(counter=10)
    state.items = ["a", "b"]

    snapshot = state.snapshot()

    # Mutate original
    state.items.append("c")
    state.counter = 20

    # Snapshot should be unchanged
    assert snapshot["items"] == ["a", "b"]
    assert snapshot["counter"] == 10


def test_state_update_validation_rejects_unknown_fields():
    """Updating a State with a field not in the schema should raise StateValidationError."""
    class StrictState(State):
        val: int = 0

        class Config:
            strict = True

    state = StrictState(val=5)

    with pytest.raises(StateValidationError, match="unknown"):
        state.validate_update({"nonexistent_field": 42})


def test_state_to_dict_includes_all_fields():
    """to_dict() should include all fields with their current values."""
    class FullState(State):
        a: str = "hello"
        b: int = 42
        c: list = Field.reducer("append")

    state = FullState()
    state.c = [1, 2, 3]

    d = state.to_dict()
    assert d == {"a": "hello", "b": 42, "c": [1, 2, 3]}
