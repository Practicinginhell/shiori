"""Type validation for state values.

Checks values against the *declared* field types (``int``, ``list[str]``, a
dataclass, a Pydantic model, ...) using Pydantic's ``TypeAdapter`` under the
hood, so users never have to write Pydantic models. Values are coerced the way
Pydantic's lax mode does (``"5"`` -> ``5`` for an ``int`` field).

Validation modes (``StateSchemaIR.validation`` / ``State.Config.validation``):
  - ``"off"``         no type checks
  - ``"boundaries"``  (default) run input and human approval edits
  - ``"updates"``     boundaries + every node's state update
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import PydanticUserError, TypeAdapter, ValidationError

from graphflow.errors import StateValidationError

_SKIP = object()  # marker: nothing to validate


@dataclass(frozen=True)
class FieldSpec:
    """A field as the validator sees it: resolved type + reducer kind."""
    name: str
    annotation: Any = Any
    reducer: str = "replace"  # replace | append | sum | merge_dict | custom
    nullable: bool = True     # None is accepted (the field's default is None)


def _checkable(ann: Any) -> bool:
    return ann is not Any and ann is not None and ann is not object and not isinstance(ann, (str, typing.ForwardRef))


class _Checker:
    """Validates one value against one annotation, falling back to isinstance for arbitrary classes."""

    def __init__(self, ann: Any):
        self.ann = ann
        try:
            self.adapter: TypeAdapter[Any] | None = TypeAdapter(ann)
        except (PydanticUserError, TypeError):  # no pydantic schema (arbitrary class): isinstance check instead
            self.adapter = None

    def __call__(self, value: Any) -> Any:
        if self.adapter is not None:
            return self.adapter.validate_python(value)
        origin = typing.get_origin(self.ann) or self.ann
        if isinstance(origin, type) and not isinstance(value, origin):
            raise TypeError(f"expected {getattr(origin, '__name__', origin)}, got {type(value).__name__}")
        return value


def _summarize(error: Exception) -> str:
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        where = f" at {loc}" if loc else ""
        return f"{first.get('msg', 'invalid value')}{where} (got {first.get('input')!r})"
    return str(error)


class SchemaValidator:
    """Validates and coerces values for a state schema."""

    def __init__(self, schema_name: str, fields: Mapping[str, FieldSpec], strict_keys: bool = False):
        self.schema_name = schema_name
        self.fields = dict(fields)
        self.strict_keys = strict_keys
        self._checkers: dict[tuple[str, str], _Checker | object] = {}

    def _checker(self, spec: FieldSpec, shape: str) -> _Checker | object:
        key = (spec.name, shape)
        if key not in self._checkers:
            ann = spec.annotation
            if shape == "item":  # one element appended to a list[X] field
                args = typing.get_args(ann)
                ann = args[0] if typing.get_origin(ann) in (list, list) and args else Any
            self._checkers[key] = _Checker(ann) if _checkable(ann) else _SKIP
        return self._checkers[key]

    def check_keys(self, values: Mapping[str, Any], where: str) -> None:
        if not self.strict_keys or not self.fields:
            return
        unknown = sorted(k for k in values if k not in self.fields)
        if unknown:
            raise StateValidationError(
                f"{self.schema_name} received unknown field(s) in {where}: {', '.join(unknown)}"
            )

    def validate(self, values: Mapping[str, Any], where: str) -> dict[str, Any]:
        """Validates a state update (reducer-aware) and returns the coerced values."""
        self.check_keys(values, where)
        out = dict(values)
        problems: list[str] = []
        for key, value in values.items():
            spec = self.fields.get(key)
            if spec is None or spec.reducer == "custom" or (value is None and spec.nullable):
                continue
            shape = "item" if spec.reducer == "append" and not isinstance(value, list) else "value"
            checker = self._checker(spec, shape)
            if checker is _SKIP:
                continue
            try:
                out[key] = checker(value)  # type: ignore[operator]
            except (ValidationError, TypeError, ValueError) as e:
                problems.append(f"'{key}': {_summarize(e)}")
        if problems:
            raise StateValidationError(
                f"Invalid {where} for {self.schema_name}: " + "; ".join(problems)
            )
        return out
