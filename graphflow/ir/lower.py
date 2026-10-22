"""Lowering helpers: replace live Python objects in builder-stage IR with registry references."""

from __future__ import annotations

import json
import typing
from dataclasses import replace
from typing import Any

from graphflow.ir.registry import Registry, is_ref
from graphflow.ir.state import BUILTIN_REDUCER_NAMES, FieldIR, StateSchemaIR


def type_repr(ann: Any) -> str:
    """A readable rendering of a type annotation (``list[str]``, ``TaskResult``, ``Any``)."""
    if ann is Any or ann is None:
        return "Any"
    if isinstance(ann, str):
        return ann
    if isinstance(ann, type) and not typing.get_args(ann):
        return ann.__qualname__
    return str(ann).replace("typing.", "")


def is_plain_data(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def ref_for(value: Any, registry: Registry, kind: str, name: str | None = None) -> str | None:
    """Returns a reference for `value`: None stays None, refs stay refs, objects get registered."""
    if value is None:
        return None
    if is_ref(value):
        return value
    return registry.register(value, kind, name)


def lower_field(f: FieldIR, registry: Registry) -> FieldIR:
    ann = f.type_annotation  # None once lowered; lowering is idempotent
    t_ref = None
    if ann is not Any and ann is not None and not isinstance(ann, str):
        t_ref = registry.register(ann, "type", type_repr(ann))

    default, default_ref = f.default, f.default_ref
    if default is not None and not is_plain_data(default):
        default_ref, default = registry.register(default, "default", f.name), None

    reducer = f.reducer
    if callable(reducer):
        reducer = registry.register(reducer, "reducer")
    elif reducer not in BUILTIN_REDUCER_NAMES and not is_ref(reducer):
        raise ValueError(f"Field '{f.name}' has unknown reducer {reducer!r}")

    return replace(
        f,
        type_annotation=None,
        type_repr=f.type_repr or type_repr(ann),
        type_ref=t_ref or f.type_ref,
        default=default,
        default_ref=default_ref,
        default_factory=ref_for(f.default_factory, registry, "factory"),
        reducer=reducer,
    )


def lower_schema(schema: StateSchemaIR, registry: Registry) -> StateSchemaIR:
    return StateSchemaIR(
        name=schema.name,
        fields={name: lower_field(f, registry) for name, f in schema.fields.items()},
        raw_cls=None,
        strict=schema.strict,
        validation=schema.validation,
    )
