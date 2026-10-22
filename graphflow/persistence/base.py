"""Graphflow Persistence abstraction.

Interface and adapters for persisting execution state, enabling conversational
memory across turns and human-in-the-loop pause/resume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCheckpointStore(ABC):
    """Abstract interface for checkpoint persistence."""

    #: False for checkpointers that only implement LangGraph's sync API
    #: (async runs then execute the graph in a worker thread).
    supports_async: bool = True

    @abstractmethod
    def get_langgraph_checkpointer(self) -> Any:
        """Returns the underlying LangGraph checkpointer instance."""

    def close(self) -> None:  # noqa: B027 - optional hook; most stores hold nothing to release
        """Releases any resources held by the store."""

    def __enter__(self) -> BaseCheckpointStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def checkpointer_supports_async(checkpointer: Any) -> bool:
    """Whether a raw LangGraph checkpointer implements the async API."""
    if checkpointer is None:
        return True
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return True
    return not isinstance(checkpointer, SqliteSaver)
