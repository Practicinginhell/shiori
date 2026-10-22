"""Graphflow IR - retry policy (plain data)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryIR:
    """A node's or tool's retry policy. `retry_on` is a registry reference, or None for transient errors."""
    max_attempts: int = 3
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    jitter: bool = True
    retry_on: str | None = None
