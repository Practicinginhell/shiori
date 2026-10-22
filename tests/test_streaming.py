"""Unit tests for Graphflow high-level streaming events."""

import pytest

from graphflow import (
    Application,
    Field,
    NodeCompleted,
    NodeStarted,
    RunCompleted,
    RunStarted,
    State,
    StateUpdated,
    Workflow,
)


class StreamState(State):
    val: int = 0
    items: list = Field.reducer("append")


def step_a(state: dict) -> dict:
    return {"val": 10, "items": ["a"]}


def step_b(state: dict) -> dict:
    return {"val": 20, "items": ["b"]}


@pytest.mark.asyncio
async def test_async_streaming_lifecycle():
    wf = Workflow("stream_wf", state_schema=StreamState)
    wf.then(step_a).then(step_b)

    app = Application("stream_app")
    app.register(wf)

    events = []
    async for event in app.astream({"val": 0}):
        events.append(event)

    types = [type(e) for e in events]
    assert RunStarted in types
    assert NodeStarted in types
    assert StateUpdated in types
    assert NodeCompleted in types
    assert RunCompleted in types

    # Check node completion order
    completed_nodes = [e.node_name for e in events if isinstance(e, NodeCompleted)]
    assert completed_nodes == ["step_a", "step_b"]


def test_sync_streaming():
    wf = Workflow("sync_stream_wf", state_schema=StreamState)
    wf.then(step_a).then(step_b)

    app = Application("sync_stream_app")
    app.register(wf)

    events = list(app.stream({"val": 0}))
    assert len(events) > 0
    assert any(isinstance(e, RunCompleted) for e in events)
