"""Per-run context and tool arguments injected from it.

The run context is data that comes with each run rather than living in the
conversation state: who the user is, their tenant, their permissions. Pass it as
``app.run(..., context={...})`` (also ``resume``, ``stream``, ...). It is
LangGraph's run context: every node sees it, nested agents included, and it is
never saved in checkpoints, so each run and each resume passes it again.

A tool declares an argument the model must not see or set with ``Annotated``::

    @tool
    def create_booking(slot_id: str, phone: Annotated[str, FromContext("customer_phone")]) -> str: ...

The argument is left out of the schema sent to the model; when an agent calls the
tool, graphflow fills it from the run context (``FromContext``) or the workflow
state (``FromState``) and drops any value the model supplied for it.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FromContext:
    """Fill this tool argument from the run context's `key`."""
    key: str

    def ref(self) -> str:
        return f"context:{self.key}"


@dataclass(frozen=True)
class FromState:
    """Fill this tool argument from the workflow state's `key`."""
    key: str

    def ref(self) -> str:
        return f"state:{self.key}"


Injection = FromContext | FromState

_MISSING = object()


def get_context() -> Any:
    """The current run's context (as passed to ``run(..., context=...)``), or None outside a run."""
    try:
        from langgraph.runtime import get_runtime

        return get_runtime().context
    except RuntimeError:  # not inside a graph run
        return None


def injected_params(func: Callable[..., Any]) -> dict[str, Injection]:
    """The parameters of `func` annotated ``Annotated[T, FromContext(...) | FromState(...)]``."""
    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except (NameError, TypeError):
        return {}
    found: dict[str, Injection] = {}
    for name, hint in hints.items():
        if typing.get_origin(hint) is typing.Annotated:
            marks = [m for m in typing.get_args(hint)[1:] if isinstance(m, (FromContext, FromState))]
            if marks:
                found[name] = marks[0]
    return found


def lookup(source: Any, key: str) -> Any:
    """`source[key]` for mappings, `source.key` for objects; _MISSING when absent."""
    if source is None:
        return _MISSING
    if isinstance(source, Mapping):
        return source.get(key, _MISSING)
    return getattr(source, key, _MISSING)


def has_default(func: Callable[..., Any], name: str) -> bool:
    try:
        return inspect.signature(func).parameters[name].default is not inspect.Parameter.empty
    except (KeyError, TypeError, ValueError):
        return False


def resolve(
    injections: Mapping[str, Injection], func: Callable[..., Any], context: Any, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Values for `injections` from the run context / state (parameters with defaults may be absent).

    Raises ConfigurationError for a required value the run didn't provide.
    """
    from graphflow.errors import ConfigurationError

    values: dict[str, Any] = {}
    for name, injection in injections.items():
        source = context if isinstance(injection, FromContext) else state
        value = lookup(source, injection.key)
        if value is not _MISSING:
            values[name] = value
        elif not has_default(func, name):
            where = "run context" if isinstance(injection, FromContext) else "workflow state"
            raise ConfigurationError(
                f"'{getattr(func, '__name__', 'tool')}' needs '{injection.key}' from the {where} "
                f"(argument '{name}'), and this run doesn't provide it"
                + (f"; pass run(..., context={{'{injection.key}': ...}})" if isinstance(injection, FromContext) else "")
            )
    return values
