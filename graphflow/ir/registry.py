"""Registry of live Python objects referenced by the IR.

The IR is plain data: wherever a workflow needs code or a non-data value
(a node function, a condition, a reducer, a type, a model, a tool...), the IR
holds a string reference such as ``"fn:prepare_tx"`` and the Registry maps it
to the object. The compiler receives both.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from graphflow.errors import CompilationError


def is_ref(value: Any) -> bool:
    """True for registry reference strings ("kind:name")."""
    return isinstance(value, str) and ":" in value


def object_name(obj: Any) -> str:
    """A readable name for an object, used to build reference names."""
    for attr in ("name", "__qualname__", "__name__"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value.replace("<", "").replace(">", "")
    return type(obj).__name__


class Registry:
    """Maps reference strings to objects. The same object always gets the same reference."""

    def __init__(self) -> None:
        self._objects: dict[str, Any] = {}
        self._refs_by_id: dict[int, str] = {}

    def register(self, obj: Any, kind: str, name: str | None = None) -> str:
        """Registers `obj` and returns its reference (reusing an existing one for the same object)."""
        existing = self._refs_by_id.get(id(obj))
        if existing is not None and existing.startswith(f"{kind}:"):
            return existing
        base = f"{kind}:{name or object_name(obj)}"
        ref, n = base, 1
        while ref in self._objects and self._objects[ref] is not obj:
            n += 1
            ref = f"{base}#{n}"
        self._objects[ref] = obj
        self._refs_by_id.setdefault(id(obj), ref)
        return ref

    def put(self, ref: str, obj: Any) -> None:
        """Binds an explicit reference (e.g. ``"apptool:search"``) to `obj`."""
        self._objects[ref] = obj
        self._refs_by_id.setdefault(id(obj), ref)

    def resolve(self, ref: str) -> Any:
        try:
            return self._objects[ref]
        except KeyError:
            raise CompilationError(f"Unresolved reference '{ref}'") from None

    def get(self, ref: str, default: Any = None) -> Any:
        return self._objects.get(ref, default)

    def __contains__(self, ref: object) -> bool:
        return ref in self._objects

    def __iter__(self) -> Iterator[str]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)
