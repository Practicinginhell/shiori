"""Built-in state reducers.

Reducers are referenced by name in the IR ("append", "sum", ...) and resolved to
functions here, so both the user-facing State API and the compiler backend can
share them without depending on each other.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

Reducer = Callable[[Any, Any], Any]


def replace_reducer(current: Any, update: Any) -> Any:
    return update


def append_reducer(current: Any, update: Any) -> list[Any]:
    """List append reducer that accepts either single items or lists."""
    if current is None:
        current = []
    elif not isinstance(current, list):
        current = [current]
    if update is None:
        return current
    if isinstance(update, list):
        return current + update
    return current + [update]


def sum_reducer(current: Any, update: Any) -> Any:
    if current is None:
        return update
    if update is None:
        return current
    return operator.add(current, update)


def merge_dict_reducer(current: Any, update: Any) -> dict[Any, Any]:
    curr = current if isinstance(current, dict) else {}
    upd = update if isinstance(update, dict) else {}
    return {**curr, **upd}


BUILTIN_REDUCERS: dict[str, Reducer] = {
    "replace": replace_reducer,
    "append": append_reducer,
    "sum": sum_reducer,
    "merge_dict": merge_dict_reducer,
}


def resolve_reducer(reducer: str | Reducer | None) -> Reducer:
    """Returns the reducer function for a reducer name or callable."""
    if reducer is None:
        return replace_reducer
    if callable(reducer):
        return reducer
    try:
        return BUILTIN_REDUCERS[reducer]
    except KeyError:
        raise ValueError(
            f"Unknown reducer '{reducer}'. Expected one of {sorted(BUILTIN_REDUCERS)} or a callable."
        ) from None
