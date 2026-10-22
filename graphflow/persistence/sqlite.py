"""SQLite persistent checkpoint store adapter.

Requires the ``sqlite`` extra: ``pip install graphflow[sqlite]``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from graphflow.persistence.base import BaseCheckpointStore


class SqliteCheckpointStore(BaseCheckpointStore):
    """Durable, SQLite-backed checkpoint store for local file persistence."""

    supports_async = False  # LangGraph's SqliteSaver is sync-only

    def __init__(self, db_path: str = "checkpoints.db") -> None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as e:
            raise ImportError(
                "SqliteCheckpointStore requires `pip install graphflow[sqlite]` "
                "(langgraph-checkpoint-sqlite)"
            ) from e
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = sqlite3.connect(db_path, check_same_thread=False)
        self._saver = SqliteSaver(self._conn)

    def get_langgraph_checkpointer(self) -> Any:
        return self._saver

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
