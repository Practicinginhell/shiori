"""In-memory and durable long-term key-value memory store."""

from __future__ import annotations

from typing import Any

from langgraph.store.memory import InMemoryStore

from graphflow.memory.base import BaseMemoryStore


class LongTermMemory(BaseMemoryStore):
    """Long-term memory backed by LangGraph's InMemoryStore."""

    def __init__(self) -> None:
        self._store = InMemoryStore()

    def get(self, namespace: tuple[str, ...], key: str) -> Any | None:
        item = self._store.get(namespace, key)
        return item.value if item else None

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        self._store.put(namespace, key, value)

    def search(self, namespace_prefix: tuple[str, ...], query: str | None = None) -> list[dict[str, Any]]:
        items = self._store.search(namespace_prefix, query=query)
        return [i.value for i in items]

    def get_langgraph_store(self) -> Any:
        return self._store
