"""Graphflow persistence adapters."""

from typing import Any

from graphflow.persistence.base import BaseCheckpointStore
from graphflow.persistence.memory import InMemoryCheckpointStore

__all__ = [
    "BaseCheckpointStore",
    "InMemoryCheckpointStore",
    "SqliteCheckpointStore",
]


def __getattr__(name: str) -> Any:
    # Imported lazily so the optional sqlite dependency isn't needed to import graphflow.
    if name == "SqliteCheckpointStore":
        from graphflow.persistence.sqlite import SqliteCheckpointStore

        return SqliteCheckpointStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
