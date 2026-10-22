"""Graphflow unified error hierarchy.

This module sits at the bottom of the dependency graph: every other layer may
import it, and it imports nothing from graphflow.
"""

from __future__ import annotations

import builtins
from typing import Any


class GraphflowError(Exception):
    """Base exception for all Graphflow errors."""


class ConfigurationError(GraphflowError):
    """Raised when application or workflow configuration is invalid."""


class CompilationError(GraphflowError):
    """Raised when workflow compilation into LangGraph fails."""


class ExecutionError(GraphflowError):
    """Raised when running a workflow fails.

    Errors raised by graphflow keep their type when they leave a node
    (``ModelError``, ``ToolExecutionError``, ``TimeoutError``, ...); any other
    exception is wrapped in an ``ExecutionError`` whose ``__cause__`` is the
    original. Either way ``node`` is the innermost workflow node it failed in and
    ``attempts`` how many times that node ran (both None outside a workflow).
    """

    node: str | None = None
    attempts: int | None = None


class ModelError(ExecutionError):
    """A model call failed (provider error, network, rate limit, ...).

    ``agent`` is the agent that called the model, ``status_code`` the HTTP status
    when the provider returned one, and ``transient`` whether retrying may help
    (what the default retry policy uses). ``__cause__`` is the provider's exception.
    """

    def __init__(self, message: str, *, agent: str = "", status_code: int | None = None, transient: bool = False):
        super().__init__(message)
        self.agent = agent
        self.status_code = status_code
        self.transient = transient


class StateValidationError(GraphflowError):
    """Raised when state updates violate schema constraints."""


class ToolExecutionError(ExecutionError):
    """Raised when tool execution fails."""


class TimeoutError(ExecutionError, builtins.TimeoutError):
    """Raised when a node or tool exceeds its time limit.

    Subclasses the builtin ``TimeoutError`` so ``except TimeoutError`` catches it
    whichever of the two names is in scope.
    """


class ToolTimeoutError(ToolExecutionError, TimeoutError):
    """A tool exceeded its time limit: a tool failure (reported to the model) and a TimeoutError."""


# Mistakes in how the application is set up or called (a missing run-context value,
# an invalid state update, a bad workflow). Retrying or falling back can't fix them,
# so nodes never retry them, never pass them to on_error, and raise them unchanged.
SETUP_ERRORS: tuple[type[GraphflowError], ...] = (ConfigurationError, CompilationError, StateValidationError)


class HumanApprovalInterrupted(GraphflowError):
    """Signals that execution is paused waiting for human approval."""

    def __init__(self, prompt: str, interrupt_id: str, state_snapshot: dict[str, Any]):
        super().__init__(prompt)
        self.prompt = prompt
        self.interrupt_id = interrupt_id
        self.state_snapshot = state_snapshot
