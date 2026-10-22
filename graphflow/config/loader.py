"""Graphflow configuration loader.

Loads configuration from a YAML file, a dict, or nothing, then applies
environment overrides. The result is a plain nested dict (``Config``) with
typed helpers::

    app:
      name: my-app
      model:        {provider: openai, model_name: gpt-4o-mini, temperature: 0.2,
                     api_key: ..., base_url: ..., max_tokens: 4096}
      persistence:  {backend: memory | sqlite | none, db_path: checkpoints.db}
      runtime:      {max_retries: 3, timeout: 120}

The legacy top-level ``model:`` section (with ``connection``/``generation``/
``reliability`` sub-sections) is still understood.

Environment overrides:
  - ``GRAPHFLOW_APP_NAME=x``            -> app.name
  - ``GRAPHFLOW__APP__MODEL__TEMPERATURE=0.3`` -> app.model.temperature
    (double underscores separate nesting levels; values are parsed as YAML scalars)
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from graphflow.errors import ConfigurationError

ENV_PREFIX = "GRAPHFLOW"


@dataclass
class ModelConfig:
    name: str = "gpt-4o-mini"
    provider: str = "openai"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class RuntimeConfig:
    max_retries: int = 3
    timeout_seconds: float = 120.0
    checkpointer_type: str = "memory"
    db_path: str = "checkpoints.db"


@dataclass
class AppConfig:
    name: str = "graphflow-app"
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    extra: dict[str, Any] = field(default_factory=dict)


class Config(dict):
    """Nested configuration dict with typed accessors."""

    def _section(self, *path: str) -> dict[str, Any]:
        node: Any = self
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        return dict(node) if isinstance(node, Mapping) else {}

    @property
    def app_name(self) -> str:
        return str(self._section("app").get("name") or "graphflow-app")

    def model_config(self) -> ModelConfig:
        m = self._section("app", "model")
        legacy = self._section("model")
        conn, gen = legacy.get("connection") or {}, legacy.get("generation") or {}
        return ModelConfig(
            name=m.get("model_name") or m.get("name") or legacy.get("name") or "gpt-4o-mini",
            provider=m.get("provider") or legacy.get("provider") or "openai",
            api_key=m.get("api_key") or conn.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            base_url=m.get("base_url") or conn.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
            temperature=float(m.get("temperature", gen.get("temperature", 0.0))),
            max_tokens=int(m.get("max_tokens", gen.get("max_tokens", 4096))),
        )

    def runtime_config(self) -> RuntimeConfig:
        r = self._section("app", "runtime") or self._section("runtime")
        reliability = self._section("model", "reliability")
        p = self._section("app", "persistence")
        return RuntimeConfig(
            max_retries=int(r.get("max_retries", reliability.get("max_retries", 3))),
            timeout_seconds=float(r.get("timeout", reliability.get("timeout", 120.0))),
            checkpointer_type=str(p.get("backend") or r.get("checkpointer") or "memory"),
            db_path=str(p.get("db_path") or r.get("db_path") or "checkpoints.db"),
        )

    def to_app_config(self) -> AppConfig:
        return AppConfig(
            name=self.app_name,
            model=self.model_config(),
            runtime=self.runtime_config(),
            extra=self._section("extra") or self._section("app", "extra"),
        )

    def create_model(self) -> Any:
        """Instantiates the configured model adapter."""
        mc = self.model_config()
        if mc.provider == "openai":
            from graphflow.models.openai import OpenAIModel

            return OpenAIModel(
                model_name=mc.name,
                api_key=mc.api_key,
                base_url=mc.base_url,
                temperature=mc.temperature,
                max_tokens=mc.max_tokens,
            )
        from graphflow.models.registry import resolve_model

        return resolve_model(f"{mc.provider}:{mc.name}")

    def create_checkpointer(self) -> Any:
        """Instantiates the configured checkpoint store (or None)."""
        rc = self.runtime_config()
        backend = rc.checkpointer_type.lower()
        if backend in ("none", "", "null"):
            return None
        if backend == "memory":
            from graphflow.persistence.memory import InMemoryCheckpointStore

            return InMemoryCheckpointStore()
        if backend == "sqlite":
            from graphflow.persistence.sqlite import SqliteCheckpointStore

            return SqliteCheckpointStore(db_path=rc.db_path)
        raise ConfigurationError(f"Unknown persistence backend '{rc.checkpointer_type}'")


def _set_path(data: dict[str, Any], path: list[str], value: Any) -> None:
    node = data
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[path[-1]] = value


def _apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    if f"{ENV_PREFIX}_APP_NAME" in environ:
        _set_path(data, ["app", "name"], environ[f"{ENV_PREFIX}_APP_NAME"])
    nested_prefix = f"{ENV_PREFIX}__"
    for key, raw in environ.items():
        if key.startswith(nested_prefix):
            path = [p.lower() for p in key[len(nested_prefix):].split("__") if p]
            if path:
                _set_path(data, path, yaml.safe_load(raw) if raw else raw)


def load_config(
    source: str | Path | Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Loads configuration from a YAML path or dict, then applies GRAPHFLOW_* env overrides.

    A missing file raises ConfigurationError; ``None`` yields defaults + env overrides.
    """
    data: dict[str, Any] = {}
    if isinstance(source, Mapping):
        data = copy.deepcopy(dict(source))
    elif source is not None:
        path = Path(source)
        if not path.exists():
            raise ConfigurationError(f"Config file '{path}' does not exist")
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ConfigurationError(f"Config file '{path}' must contain a mapping at the top level")
        data = loaded

    _apply_env_overrides(data, os.environ if environ is None else environ)
    data.setdefault("app", {}).setdefault("name", "graphflow-app")
    return Config(data)
