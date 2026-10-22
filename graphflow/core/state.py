"""Graphflow State System.

First-class typed state abstraction supporting declarative reducers, validation,
serialization, and seamless compilation into LangGraph TypedDict states with
Annotated reducers.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import typing
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, get_type_hints

from graphflow.errors import ConfigurationError
from graphflow.ir.state import (
    BUILTIN_REDUCER_NAMES,
    VALIDATION_MODES,
    FieldIR,
    ReducerType,
    StateSchemaIR,
)
from graphflow.reducers import (
    BUILTIN_REDUCERS,
    resolve_reducer,
)
from graphflow.reducers import (
    append_reducer as _append_reducer,
)
from graphflow.reducers import (
    merge_dict_reducer as _merge_dict_reducer,
)
from graphflow.validation import FieldSpec, SchemaValidator

__all__ = [
    "BUILTIN_REDUCERS",
    "Field",
    "State",
    "_append_reducer",
    "_merge_dict_reducer",
    "create_state_schema",
    "schema_from_any",
    "schema_validator",
]


class Field:
    """Field descriptor for defining state properties and reducers."""

    def __init__(
        self,
        default: Any = None,
        default_factory: Callable[[], Any] | None = None,
        reducer: ReducerType = "replace",
        description: str = "",
    ):
        self.default = default
        self.default_factory = default_factory
        self.reducer = reducer
        self.description = description

    @classmethod
    def reducer(
        cls,
        reducer: ReducerType,
        default: Any = None,
        default_factory: Callable[[], Any] | None = None,
        description: str = "",
    ) -> Field:
        """Convenient constructor to declare a field with a specific reducer."""
        if default is None and default_factory is None:
            if reducer == "append":
                default_factory = list
            elif reducer == "sum":
                default = 0
            elif reducer == "merge_dict":
                default_factory = dict
        return cls(
            default=default,
            default_factory=default_factory,
            reducer=reducer,
            description=description,
        )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "content") and hasattr(obj, "type"):  # LangChain messages
        return {"role": obj.type, "content": obj.content}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class State(Mapping):
    """Base class for user-defined state.

    Declare fields as annotated class attributes. Supports attribute access
    (``state.query``) and dict access (``state["query"]``), defaults, reducers,
    JSON serialization, and optional strict validation::

        class ChatState(State):
            query: str = ""
            messages: list = Field.reducer("append")

            class Config:
                strict = True   # reject unknown fields
    """

    _gf_schema: ClassVar[StateSchemaIR] = StateSchemaIR(name="State")
    _gf_validator: ClassVar[SchemaValidator | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._gf_schema = cls._build_schema()
        # Field defaults live in the schema; removing the class attributes lets
        # instance attribute access fall through to __getattr__ and the data dict.
        for name in cls._gf_schema.fields:
            if name in cls.__dict__:
                delattr(cls, name)

    @classmethod
    def _build_schema(cls) -> StateSchemaIR:
        inherited: dict[str, FieldIR] = {}
        for base in reversed(cls.__mro__[1:]):
            base_schema = base.__dict__.get("_gf_schema")
            if isinstance(base_schema, StateSchemaIR):
                inherited.update(base_schema.fields)

        try:
            hints = get_type_hints(cls, include_extras=True)
        except (NameError, TypeError, AttributeError):  # unresolvable forward refs: use raw annotations
            hints = {}
        own_annotations = inspect.get_annotations(cls)

        fields = dict(inherited)
        for name, raw_ann in own_annotations.items():
            if name.startswith("_"):
                continue
            ann = hints.get(name, raw_ann)
            if typing.get_origin(ann) is ClassVar or (
                isinstance(ann, str) and ann.startswith("ClassVar")
            ):
                continue

            attr = cls.__dict__.get(name, None)
            if isinstance(attr, Field):
                fields[name] = FieldIR(
                    name=name,
                    type_annotation=ann,
                    default=attr.default,
                    default_factory=attr.default_factory,
                    reducer=attr.reducer,
                    description=attr.description,
                )
            else:
                fields[name] = FieldIR(name=name, type_annotation=ann, default=attr)

        config = getattr(cls, "Config", None)
        strict = bool(getattr(config, "strict", False))
        validation = getattr(config, "validation", "boundaries")
        if validation not in VALIDATION_MODES:
            raise ConfigurationError(
                f"{cls.__name__}.Config.validation must be one of {VALIDATION_MODES}, got {validation!r}"
            )
        return StateSchemaIR(
            name=cls.__name__,
            fields=fields,
            raw_cls=cls,
            strict=strict,
            validation=validation,
        )

    @classmethod
    def _validator(cls) -> SchemaValidator:
        cached = cls.__dict__.get("_gf_validator")
        if cached is None:
            cached = schema_validator(cls._gf_schema)
            cls._gf_validator = cached
        return cached

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, **kwargs: Any) -> None:
        schema = type(self)._gf_schema
        data = {name: f.make_default() for name, f in schema.fields.items()}
        object.__setattr__(self, "_data", data)
        if schema.strict:
            type(self)._validator().check_keys(kwargs, "constructor")
        data.update(kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> State:
        return cls(**dict(data))

    @classmethod
    def from_json(cls, payload: str) -> State:
        return cls.from_dict(json.loads(payload))

    @classmethod
    def get_schema_ir(cls) -> StateSchemaIR:
        """Returns this state's StateSchemaIR."""
        return cls._gf_schema

    # ------------------------------------------------------------------ #
    # Attribute / mapping access
    # ------------------------------------------------------------------ #

    def __getattribute__(self, name: str) -> Any:
        # Field values take precedence over inherited Mapping methods, so a field
        # named e.g. `items` or `values` is reachable as an attribute.
        if not name.startswith("_"):
            try:
                data = object.__getattribute__(self, "_data")
            except AttributeError:
                data = {}
            if name in data:
                return data[name]
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' has no field '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if type(self)._gf_schema.strict:
            type(self)._validator().check_keys({key: value}, "assignment")
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"{type(self).__name__}({fields})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ------------------------------------------------------------------ #
    # Serialization / snapshots / validation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self._data, default=_json_default, **kwargs)

    def snapshot(self) -> Mapping[str, Any]:
        """Returns a read-only deep copy that won't change when this state does."""
        return MappingProxyType(copy.deepcopy(self._data))

    def validate_update(self, update: Mapping[str, Any]) -> dict[str, Any]:
        """Validates a state update against the declared field types and returns it coerced.

        Raises StateValidationError for values of the wrong type, and (for strict
        schemas, ``class Config: strict = True``) for undeclared fields.
        """
        return type(self)._validator().validate(update, "update")

    def apply(self, update: Mapping[str, Any]) -> State:
        """Returns a new state with `update` merged in using each field's reducer."""
        update = self.validate_update(update)
        schema = type(self)._gf_schema
        new = type(self)(**copy.deepcopy(self._data))
        for key, value in update.items():
            f = schema.fields.get(key)
            reducer = resolve_reducer(f.reducer if f else "replace")
            new._data[key] = reducer(new._data.get(key), value)
        return new


def schema_validator(schema: StateSchemaIR) -> SchemaValidator:
    """A validator for a builder-stage schema (live types)."""
    specs = {
        name: FieldSpec(
            name=name,
            annotation=f.type_annotation,
            reducer=f.reducer if f.reducer in BUILTIN_REDUCER_NAMES else "custom",
            nullable=not f.has_default,
        )
        for name, f in schema.fields.items()
    }
    return SchemaValidator(schema.name, specs, strict_keys=schema.strict)


def create_state_schema(
    fields: dict[str, Any],
    name: str = "DynamicState",
    strict: bool = False,
    validation: str = "boundaries",
) -> StateSchemaIR:
    """Helper to create a StateSchemaIR from a dict of field names to defaults/Fields."""
    fields_ir: dict[str, FieldIR] = {}
    for k, v in fields.items():
        if isinstance(v, Field):
            fields_ir[k] = FieldIR(
                name=k,
                default=v.default,
                default_factory=v.default_factory,
                reducer=v.reducer,
                description=v.description,
            )
        elif isinstance(v, FieldIR):
            fields_ir[k] = v
        else:
            fields_ir[k] = FieldIR(name=k, default=v)
    return StateSchemaIR(
        name=name, fields=fields_ir, strict=strict, validation=validation
    )  # type: ignore[arg-type]


def _schema_from_typeddict(cls: type) -> StateSchemaIR:
    hints = get_type_hints(cls, include_extras=True)
    fields: dict[str, FieldIR] = {}
    for name, ann in hints.items():
        reducer: ReducerType = "replace"
        base_ann = ann
        if typing.get_origin(ann) is typing.Annotated:
            base_ann, *meta = typing.get_args(ann)
            fn = next((m for m in meta if callable(m)), None)
            if fn is not None:
                reducer = fn
        fields[name] = FieldIR(name=name, type_annotation=base_ann, reducer=reducer)
    return StateSchemaIR(name=cls.__name__, fields=fields, raw_cls=cls)


def _schema_from_pydantic(cls: type) -> StateSchemaIR:
    from pydantic_core import PydanticUndefined

    fields: dict[str, FieldIR] = {}
    for name, info in cls.model_fields.items():
        default = None if info.default is PydanticUndefined else info.default
        reducer: ReducerType = "replace"
        fn = next((m for m in info.metadata if callable(m)), None)
        if fn is not None:
            reducer = fn
        fields[name] = FieldIR(
            name=name,
            type_annotation=info.annotation,
            default=default,
            default_factory=info.default_factory,
            reducer=reducer,
            description=info.description or "",
        )
    return StateSchemaIR(name=cls.__name__, fields=fields, raw_cls=cls)


def schema_from_any(schema: Any) -> StateSchemaIR:
    """Builds a StateSchemaIR from a State subclass, dict, TypedDict, Pydantic model or IR."""
    if schema is None:
        return StateSchemaIR(name="State")
    if isinstance(schema, StateSchemaIR):
        return schema
    if isinstance(schema, dict):
        return create_state_schema(schema)
    if isinstance(schema, type):
        if issubclass(schema, State):
            return schema.get_schema_ir()
        if hasattr(schema, "model_fields"):
            return _schema_from_pydantic(schema)
        if issubclass(schema, dict) and hasattr(schema, "__annotations__"):
            return _schema_from_typeddict(schema)
    raise TypeError(
        f"Unsupported state_schema {schema!r}: expected a State subclass, TypedDict, "
        "Pydantic model, dict of fields, or StateSchemaIR."
    )
