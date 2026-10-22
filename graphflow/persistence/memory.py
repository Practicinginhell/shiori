"""In-memory checkpoint store adapter."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from graphflow.persistence.base import BaseCheckpointStore


class InMemoryCheckpointStore(BaseCheckpointStore):
    """Ephemeral, in-memory checkpointer suitable for testing and local exploration."""

    def __init__(self) -> None:
        self._saver = MemorySaver()

    def get_langgraph_checkpointer(self) -> Any:
        return self._saver
