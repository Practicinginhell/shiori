"""Checkpoint history and state editing for threads (time travel).

Wraps LangGraph's ``get_state_history`` / ``update_state``. Replaying from a
checkpoint re-runs the steps after it; forking first writes new values at that
checkpoint (a new branch of the thread's history, leaving the original intact).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from graphflow.errors import ConfigurationError


def _hidden(name: str) -> bool:
    return name.startswith("__")


@dataclass(frozen=True)
class Checkpoint:
    """One saved step of a thread."""
    checkpoint_id: str
    values: dict[str, Any]           # the state (internal channels hidden)
    next: tuple[str, ...]            # nodes that run next from here (internal nodes hidden)
    step: int                        # -1 for the input checkpoint, then 0, 1, ...
    done: bool = False               # nothing left to run (``next`` alone can't tell: internal steps are hidden)
    source: str = ""                 # "input", "loop", "update" or "fork"
    created_at: str = ""
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snapshot: Any) -> Checkpoint:
        metadata = dict(snapshot.metadata or {})
        parent = (snapshot.parent_config or {}).get("configurable", {}).get("checkpoint_id")
        return cls(
            checkpoint_id=snapshot.config["configurable"]["checkpoint_id"],
            values={k: v for k, v in (snapshot.values or {}).items() if not _hidden(k)},
            next=tuple(n for n in snapshot.next if not _hidden(n)),
            step=int(metadata.get("step", -1)),
            done=not snapshot.next,
            source=str(metadata.get("source", "")),
            created_at=str(snapshot.created_at or ""),
            parent_id=parent,
            metadata=metadata,
        )


class ThreadHistory:
    """Time-travel operations on one compiled graph (used by CompiledApplication)."""

    def __init__(self, compiled_graph: Any, name: str, sync_only: bool = False):
        self.graph = compiled_graph
        self.name = name
        self.sync_only = sync_only  # a sync-only checkpointer (SqliteSaver): async calls run in a thread

    def config(self, thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        if getattr(self.graph, "checkpointer", None) is None:
            raise ConfigurationError(
                f"Application '{self.name}' has no checkpointer; history and state editing need one "
                "(e.g. Application(checkpointer=InMemoryCheckpointStore()))"
            )
        configurable: dict[str, Any] = {"thread_id": thread_id}
        if checkpoint_id:
            configurable.update(checkpoint_id=checkpoint_id, checkpoint_ns="")
        return {"configurable": configurable}

    @staticmethod
    def _require(snapshot: Any, thread_id: str, checkpoint_id: str) -> None:
        # For an unknown id LangGraph returns an empty snapshot echoing the requested config.
        if snapshot is None or snapshot.created_at is None:
            raise ConfigurationError(f"Thread '{thread_id}' has no checkpoint '{checkpoint_id}'")

    def checkpoint_config(self, thread_id: str, checkpoint_id: str) -> dict[str, Any]:
        """Config pointing at an existing checkpoint (ConfigurationError if there is none)."""
        cfg = self.config(thread_id, checkpoint_id)
        self._require(self.graph.get_state(cfg), thread_id, checkpoint_id)
        return cfg

    async def acheckpoint_config(self, thread_id: str, checkpoint_id: str) -> dict[str, Any]:
        if self.sync_only:
            return await asyncio.to_thread(self.checkpoint_config, thread_id, checkpoint_id)
        cfg = self.config(thread_id, checkpoint_id)
        self._require(await self.graph.aget_state(cfg), thread_id, checkpoint_id)
        return cfg

    def history(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        return [Checkpoint.from_snapshot(s) for s in self.graph.get_state_history(self.config(thread_id), limit=limit)]

    async def ahistory(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        if self.sync_only:
            return await asyncio.to_thread(self.history, thread_id, limit)
        return [
            Checkpoint.from_snapshot(s)
            async for s in self.graph.aget_state_history(self.config(thread_id), limit=limit)
        ]

    @staticmethod
    def _overwritten(values: dict[str, Any]) -> dict[str, Any]:
        from langgraph.types import Overwrite

        return {k: Overwrite(v) for k, v in values.items()}

    def update(
        self, cfg: dict[str, Any], values: dict[str, Any], as_node: str | None, overwrite: bool
    ) -> str:
        """Writes `values` at `cfg` and returns the new checkpoint id."""
        new = self.graph.update_state(cfg, self._overwritten(values) if overwrite else values, as_node=as_node)
        return new["configurable"]["checkpoint_id"]

    async def aupdate(
        self, cfg: dict[str, Any], values: dict[str, Any], as_node: str | None, overwrite: bool
    ) -> str:
        if self.sync_only:
            return await asyncio.to_thread(self.update, cfg, values, as_node, overwrite)
        new = await self.graph.aupdate_state(cfg, self._overwritten(values) if overwrite else values, as_node=as_node)
        return new["configurable"]["checkpoint_id"]
