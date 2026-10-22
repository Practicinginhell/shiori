"""Time travel: checkpoint history, editing a thread's state, replaying and forking.

Follows LangGraph's persistence guidance: browse ``get_state_history``, replay
from a past checkpoint, fork with ``update_state`` (which applies reducers
unless the value is wrapped in ``Overwrite``).
"""

import asyncio

import pytest

from graphflow import (
    Application,
    ApprovalDecision,
    ConfigurationError,
    Field,
    InMemoryCheckpointStore,
    State,
    StateValidationError,
    Workflow,
)
from graphflow.runtime import Checkpoint


class S(State):
    n: int = 0
    log: list = Field.reducer("append")


def build(calls=None):
    calls = calls if calls is not None else []

    def double(state):
        calls.append("double")
        return {"n": state["n"] * 2, "log": ["double"]}

    def inc(state):
        calls.append("inc")
        return {"n": state["n"] + 1, "log": ["inc"]}

    wf = Workflow("w", state_schema=S).then(double).then(inc)
    return Application("a", checkpointer=InMemoryCheckpointStore()).register(wf), calls


def before(app, thread, node):
    """The checkpoint whose next step is `node`."""
    return next(c for c in app.history(thread) if c.next == (node,))


def test_history_lists_checkpoints_newest_first():
    app, _ = build()
    app.run({"n": 3}, thread_id="t")
    history = app.history("t")
    assert all(isinstance(c, Checkpoint) for c in history)
    assert history[0].values == {"n": 7, "log": ["double", "inc"]}
    assert history[0].next == () and history[0].done
    assert not history[-1].done  # the input checkpoint: next is hidden internal steps, not "finished"
    assert [c.step for c in history] == sorted((c.step for c in history), reverse=True)
    assert before(app, "t", "inc").values["n"] == 6
    assert all(not k.startswith("__") for c in history for k in c.values)
    assert history[0].parent_id == history[1].checkpoint_id


def test_history_limit():
    app, _ = build()
    app.run({"n": 1}, thread_id="t")
    assert len(app.history("t", limit=2)) == 2


def test_replay_reruns_from_a_checkpoint():
    app, calls = build()
    app.run({"n": 3}, thread_id="t")
    calls.clear()
    result = app.replay("t", before(app, "t", "inc").checkpoint_id)
    assert calls == ["inc"]  # only the steps after the checkpoint run again
    assert result["n"] == 7


def test_fork_changes_the_past_without_losing_it():
    app, _ = build()
    app.run({"n": 3}, thread_id="t")
    checkpoint = before(app, "t", "inc")
    result = app.fork("t", checkpoint.checkpoint_id, {"n": 100})
    assert result["n"] == 101
    assert app.history("t")[0].values["n"] == 101
    original = next(c for c in app.history("t") if c.checkpoint_id == checkpoint.checkpoint_id)
    assert original.values["n"] == 6  # the original checkpoint is untouched


def test_update_state_applies_reducers_unless_overwritten():
    app, _ = build()
    app.run({"n": 1}, thread_id="t")
    app.update_state("t", {"log": ["note"]})
    assert app.get_state("t").values["log"] == ["double", "inc", "note"]
    app.update_state("t", {"log": ["only"]}, overwrite=True)
    assert app.get_state("t").values["log"] == ["only"]


def test_update_state_validates_fields():
    app, _ = build()
    app.run({"n": 1}, thread_id="t")
    with pytest.raises(StateValidationError, match="unknown"):
        app.update_state("t", {"nope": 1})
    with pytest.raises(StateValidationError):
        app.update_state("t", {"n": "not an int"})


def test_edit_state_while_paused_for_approval():
    seen = []

    def publish(state):
        seen.append(state["n"])
        return {}

    wf = Workflow("w", state_schema=S).then(lambda s: {"n": 1}, name="draft").require_approval(publish, "Publish?")
    app = Application("a", checkpointer=InMemoryCheckpointStore()).register(wf)
    assert app.run({}, thread_id="t").interrupted
    app.update_state("t", {"n": 42})
    app.resume("t", ApprovalDecision(approved=True))
    assert seen == [42]


def test_time_travel_needs_a_checkpointer():
    wf = Workflow("w", state_schema=S).then(lambda s: {"n": 1}, name="one")
    app = Application("a").register(wf)
    with pytest.raises(ConfigurationError, match="checkpointer"):
        app.history("t")


def test_unknown_checkpoint_is_a_clear_error():
    app, _ = build()
    app.run({"n": 1}, thread_id="t")
    with pytest.raises(ConfigurationError, match="no checkpoint"):
        app.replay("t", "does-not-exist")


def test_async_time_travel():
    app, _ = build()
    app.run({"n": 3}, thread_id="t")

    async def go():
        history = await app.ahistory("t")
        checkpoint = next(c for c in history if c.next == ("inc",))
        forked = await app.afork("t", checkpoint.checkpoint_id, {"n": 10})
        replayed = await app.areplay("t", checkpoint.checkpoint_id)
        return forked, replayed

    forked, replayed = asyncio.run(go())
    assert forked["n"] == 11
    assert replayed["n"] == 7
