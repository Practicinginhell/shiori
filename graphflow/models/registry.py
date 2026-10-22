"""Resolves model specifications (strings, LangChain models, adapters) into adapters."""

from __future__ import annotations

from typing import Any

from graphflow.errors import ConfigurationError
from graphflow.models.base import BaseModelAdapter


def resolve_model(spec: BaseModelAdapter | str | Any, *, owner: str = "") -> BaseModelAdapter:
    """Turns a model specification into a BaseModelAdapter.

    Accepted forms:
      - a BaseModelAdapter instance (returned as-is)
      - a LangChain ``BaseChatModel`` instance
      - ``"mock"`` or ``"mock:<fixed reply>"``
      - ``"openai:<model>"`` or a bare model name (treated as OpenAI, e.g. ``"gpt-4o-mini"``)
      - ``"<provider>:<model>"`` for any provider LangChain's ``init_chat_model`` supports
    """
    if isinstance(spec, BaseModelAdapter):
        return spec

    if isinstance(spec, str):
        provider, sep, name = spec.partition(":")
        if provider == "mock":
            from graphflow.models.mock import MockModel

            return MockModel(responses=name if sep else f"Response from {owner or 'mock'}")
        if not sep or provider == "openai":
            from graphflow.models.openai import OpenAIModel

            return OpenAIModel(model_name=name if sep else spec)

        from langchain.chat_models import init_chat_model  # type: ignore[import-not-found]

        from graphflow.models.langchain import LangChainModel

        return LangChainModel(init_chat_model(spec))

    try:
        from langchain_core.language_models import BaseChatModel
    except ImportError:  # pragma: no cover
        BaseChatModel = None  # type: ignore[assignment]
    if BaseChatModel is not None and isinstance(spec, BaseChatModel):
        from graphflow.models.langchain import LangChainModel

        return LangChainModel(spec)

    raise ConfigurationError(f"Unsupported model specification for {owner or 'agent'}: {spec!r}")
