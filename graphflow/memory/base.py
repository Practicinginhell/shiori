"""Graphflow Tri-Layer Memory abstractions:
1. Working Memory: Transient state during the current execution.
2. Conversation Memory: Thread-scoped state persisted across turns (checkpointer).
3. Long-term Memory: Cross-thread persistent store for facts, preferences, profiles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemoryStore(ABC):
    """Abstract interface for cross-session long-term memory.

    Pass an instance as ``Application(store=...)``; nodes can then reach it via
    LangGraph's ``get_store()``.
    """

    @abstractmethod
    def get(self, namespace: tuple[str, ...], key: str) -> Any | None:
        """Retrieve a value by namespace and key."""

    @abstractmethod
    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Store a value at namespace and key."""

    @abstractmethod
    def search(self, namespace_prefix: tuple[str, ...], query: str | None = None) -> list[dict[str, Any]]:
        """Search items within a namespace prefix."""

    @abstractmethod
    def get_langgraph_store(self) -> Any:
        """Returns the LangGraph BaseStore backing this memory."""
