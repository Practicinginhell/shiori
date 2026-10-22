"""Graphflow IR - State definitions.

Data structures representing state schema and reducers, independent of any
execution engine.

Two stages share these classes:
  - *builder stage* (while a State class or Workflow is being defined):
    ``type_annotation``, ``default_factory`` and custom ``reducer`` may hold
    live Python objects;
  - *lowered* (produced by ``graphflow.ir.lower``): every such value is replaced
    by a registry reference string, so the schema is plain, serializable data.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

ReducerType = Literal["replace", "append", "sum", "merge_dict"] | Callable[[Any, Any], Any] | str
ValidationMode = Literal["off", "boundaries", "updates"]

BUILTIN_REDUCER_NAMES = ("replace", "append", "sum", "merge_dict")
VALIDATION_MODES = ("off", "boundaries", "updates")


@dataclass(frozen=True)
class FieldIR:
    """A state field with an optional reducer strategy."""
    name: str
    type_annotation: Any = Any          # builder stage: the Python type
    default: Any = None                 # plain data default
    default_factory: Callable[[], Any] | str | None = None  # callable (builder) or ref (lowered)
    reducer: ReducerType = "replace"    # builtin name, callable (builder) or ref (lowered)
    description: str = ""
    type_repr: str = ""                 # lowered: readable type, e.g. "list[str]"
    type_ref: str | None = None         # lowered: registry ref of the type
    default_ref: str | None = None      # lowered: ref of a default that isn't plain data

    @property
    def has_reducer(self) -> bool:
        return self.reducer != "replace"

    @property
    def has_default(self) -> bool:
        return self.default_factory is not None or self.default is not None or self.default_ref is not None

    def make_default(self) -> Any:
        """Returns a fresh default value (builder stage; mutable defaults are never shared)."""
        if callable(self.default_factory):
            return self.default_factory()
        return copy.deepcopy(self.default)


@dataclass
class StateSchemaIR:
    """The schema of a workflow's state."""
    name: str = "State"
    fields: dict[str, FieldIR] = field(default_factory=dict)
    raw_cls: type | None = None  # builder stage only
    strict: bool = False  # Reject updates/inputs with keys not declared in `fields`
    validation: ValidationMode = "boundaries"  # Type checks: off | boundaries (input + human edits) | updates (+ every node update)

    def add_field(self, field_ir: FieldIR) -> None:
        self.fields[field_ir.name] = field_ir

    def get_field(self, name: str) -> FieldIR | None:
        return self.fields.get(name)
