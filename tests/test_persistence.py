"""Unit tests for Persistence and Thread Isolation (Memory and SQLite)."""

import os
import tempfile

from graphflow import (
    Application,
    Field,
    InMemoryCheckpointStore,
    SqliteCheckpointStore,
    State,
    Workflow,
)


class CounterState(State):
    history: list = Field.reducer("append")


def increment(state: dict) -> dict:
    return {"history": ["hit"]}


def test_in_memory_persistence_and_thread_isolation():
    wf = Workflow("persist_wf", state_schema=CounterState)
    wf.then(increment)

    app = Application("persist_app", checkpointer=InMemoryCheckpointStore())
    app.register(wf)

    # Thread Alice: 2 turns
    app.run({}, thread_id="alice")
    res_alice = app.run({}, thread_id="alice")
    assert len(res_alice["history"]) == 2

    # Thread Bob: 1 turn, independent
    res_bob = app.run({}, thread_id="bob")
    assert len(res_bob["history"]) == 1


def test_sqlite_durable_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_checkpoints.db")
        store = SqliteCheckpointStore(db_path=db_path)

        wf = Workflow("sqlite_wf", state_schema=CounterState)
        wf.then(increment)

        app = Application("sqlite_app", checkpointer=store)
        app.register(wf)

        # Run turn 1
        app.run({}, thread_id="thread-sql-1")
        # Run turn 2
        res = app.run({}, thread_id="thread-sql-1")

        assert len(res["history"]) == 2
        store.close()
        assert os.path.exists(db_path)
