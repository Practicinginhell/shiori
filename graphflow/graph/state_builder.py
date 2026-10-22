"""Resolves state schemas against the registry and builds LangGraph TypedDict schemas."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from graphflow.ir.registry import Registry, is_ref
from graphflow.ir.state import BUILTIN_REDUCER_NAMES, FieldIR, StateSchemaIR
from graphflow.reducers import resolve_reducer
from graphflow.validation import FieldSpec, SchemaValidator

# Used when a workflow declares no state fields (e.g. a single-Agent app):
# the keys Agents read and write, so inputs such as {"query": ...} aren't dropped.
DEFAULT_SCHEMA = StateSchemaIR(
    name="DefaultState",
    fields={
        "messages": FieldIR(name="messages", reducer="append", default_factory=list),
        "input": FieldIR(name="input"),
        "query": FieldIR(name="query"),
        "output": FieldIR(name="output"),
        "response": FieldIR(name="response"),
        "structured_output": FieldIR(name="structured_output"),
        "data": FieldIR(name="data", reducer="merge_dict"),
    },
)


def effective_schema(schema_ir: StateSchemaIR) -> StateSchemaIR:
    """The schema actually compiled: the declared one, or DEFAULT_SCHEMA if it has no fields."""
    return schema_ir if schema_ir.fields else DEFAULT_SCHEMA


@dataclass
class ResolvedField:
    """A field with every reference resolved to a live object."""
    name: str
    annotation: Any
    reducer_name: str  # builtin name, or "custom"
    reducer: Callable[[Any, Any], Any]
    default_maker: Callable[[], Any] | None

    @property
    def has_default(self) -> bool:
        return self.default_maker is not None


def _resolve(value: Any, registry: Registry | None) -> Any:
    if is_ref(value):
        if registry is None:
            raise ValueError(f"Reference '{value}' needs a registry to resolve")
        return registry.resolve(value)
    return value


def resolve_field(f: FieldIR, registry: Registry | None) -> ResolvedField:
    annotation = _resolve(f.type_ref, registry) if f.type_ref else f.type_annotation
    if annotation is None:
        annotation = Any

    if f.reducer in BUILTIN_REDUCER_NAMES:
        reducer_name, reducer = str(f.reducer), resolve_reducer(f.reducer)
    else:
        reducer_name, reducer = "custom", _resolve(f.reducer, registry)

    factory = _resolve(f.default_factory, registry)
    if factory is not None:
        default_maker: Callable[[], Any] | None = factory
    elif f.default_ref is not None:
        value = _resolve(f.default_ref, registry)
        default_maker = lambda: copy.deepcopy(value)  # noqa: E731
    elif f.default is not None:
        data = f.default
        default_maker = lambda: copy.deepcopy(data)  # noqa: E731
    else:
        default_maker = None

    return ResolvedField(f.name, annotation, reducer_name, reducer, default_maker)


def resolve_schema(schema_ir: StateSchemaIR, registry: Registry | None = None) -> dict[str, ResolvedField]:
    return {name: resolve_field(f, registry) for name, f in effective_schema(schema_ir).fields.items()}


def build_validator(schema_ir: StateSchemaIR, registry: Registry | None = None) -> SchemaValidator:
    schema = effective_schema(schema_ir)
    specs = {
        name: FieldSpec(
            name=name,
            annotation=rf.annotation,
            reducer=rf.reducer_name,
            nullable=not rf.has_default,
        )
        for name, rf in resolve_schema(schema_ir, registry).items()
    }
    return SchemaValidator(schema.name, specs, strict_keys=schema.strict)


def _concrete_type(ann: Any, fallback: Any = Any) -> Any:
    # Forward refs and generic aliases can't be resolved reliably inside LangGraph.
    return ann if isinstance(ann, type) else fallback


def _annotation_for(rf: ResolvedField) -> Any:
    if rf.reducer_name == "replace":
        return _concrete_type(rf.annotation)
    if rf.reducer_name == "append":
        return Annotated[list, rf.reducer]
    if rf.reducer_name == "sum":
        return Annotated[rf.annotation if rf.annotation in (int, float) else int, rf.reducer]
    if rf.reducer_name == "merge_dict":
        return Annotated[dict, rf.reducer]
    return Annotated[_concrete_type(rf.annotation), rf.reducer]


def build_langgraph_state_schema(
    schema_ir: StateSchemaIR,
    extra_fields: dict[str, Any] | None = None,
    registry: Registry | None = None,
) -> type:
    """Creates a TypedDict with Annotated reducers for LangGraph's StateGraph.

    `extra_fields` adds compiler-internal channels (e.g. loop counters).
    """
    schema = effective_schema(schema_ir)
    annotations: dict[str, Any] = {
        name: _annotation_for(rf) for name, rf in resolve_schema(schema_ir, registry).items()
    }
    annotations.update(extra_fields or {})
    return TypedDict(f"Compiled{schema.name}TypedDict", annotations)  # type: ignore[operator]
