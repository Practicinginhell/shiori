"""OpenAI and OpenAI-compatible (vLLM, Ollama, LM Studio) model adapter."""

from __future__ import annotations

import copy
import os
from typing import Any

from graphflow.models.langchain import LangChainModel


class OpenAIModel(LangChainModel):
    """Adapter for OpenAI models and OpenAI-compatible inference servers.

    Requires the ``openai`` extra: ``pip install graphflow[openai]``.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **extra_kwargs: Any,
    ):
        super().__init__(chat_model=None)
        self.model_name = model_name
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        # Self-hosted OpenAI-compatible servers usually accept any key; the real
        # OpenAI API gets no placeholder, so a missing key fails loudly.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ("not-needed" if self.base_url else None)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_kwargs = extra_kwargs

    def to_langchain_model(self) -> Any:
        if self._chat_model is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                raise ImportError("OpenAIModel requires `pip install graphflow[openai]`") from e
            self._chat_model = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                **self.extra_kwargs,
            )
        return self._chat_model

    def clone(self) -> OpenAIModel:
        cloned = copy.copy(self)
        cloned._chat_model = None
        cloned.extra_kwargs = dict(self.extra_kwargs)
        return cloned
