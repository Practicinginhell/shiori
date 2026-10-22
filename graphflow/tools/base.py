"""Graphflow universal Tool abstraction.

Provides a type-safe @tool decorator and Tool class with schema validation,
sync/async execution, timeout handling, retries, and LangChain/LangGraph compatibility.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, PydanticUserError, ValidationError, create_model

from graphflow._async import call_with_timeout, is_control_flow, run_coroutine_sync
from graphflow.context import Injection, has_default, injected_params
from graphflow.errors import ConfigurationError, ToolExecutionError, ToolTimeoutError
from graphflow.ir.agent import ToolIR
from graphflow.ir.registry import Registry
from graphflow.retry import RetryPolicy, check_retry_args, lower_retry, resolve_retrier


class Tool:
    """Universal tool wrapper.

    ``requires_approval`` is enforced by the Agent that calls the tool (approval
    is an orchestration concern); calling ``execute`` directly bypasses it.

    ``retries=n`` retries any error n times; ``retry=RetryPolicy(...)`` controls
    backoff and which errors are retried (transient ones by default).

    Arguments annotated ``Annotated[T, FromContext(...)]`` / ``FromState(...)``
    (see ``graphflow.context``) are not part of the model-facing schema; agents
    fill them, and trusted code may pass them to ``execute`` directly.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
        timeout: float | None = None,
        retries: int = 0,
        requires_approval: bool = False,
        retry: RetryPolicy | None = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip() or f"Tool {self.name}"
        self.timeout = timeout
        self.retries = retries
        self.retry = check_retry_args(retries, retry, f"Tool '{self.name}'")
        self._retrier = resolve_retrier(self.retry, retries, None)
        self.requires_approval = requires_approval
        self.is_async = inspect.iscoroutinefunction(func)
        self.injected: dict[str, Injection] = injected_params(func)
        self.args_schema = args_schema or self._create_pydantic_schema(func)

    def _create_pydantic_schema(self, fn: Callable[..., Any]) -> type[BaseModel]:
        """Dynamically creates a Pydantic schema from function type hints."""
        sig = inspect.signature(fn)
        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param_name in self.injected:  # filled by the agent, never by the model
                continue
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = param.default if param.default != inspect.Parameter.empty else ...
            fields[param_name] = (annotation, default)

        return create_model(f"{self.name}Schema", **fields)

    def _validate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validates arguments; invalid input is not retried."""
        if not self.args_schema:
            return kwargs
        try:
            validated = self.args_schema(**kwargs)
        except ValidationError as e:
            raise ToolExecutionError(f"Invalid arguments for tool '{self.name}': {e}") from e
        # Keep validated values as objects (model_dump would flatten nested models to dicts).
        return {k: getattr(validated, k) for k in type(validated).model_fields}

    def _call_args(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validated model-facing arguments plus the injected ones the (trusted) caller passed."""
        injected = {k: kwargs.pop(k) for k in list(kwargs) if k in self.injected}
        for name, injection in self.injected.items():
            if name not in injected and not has_default(self.func, name):
                raise ConfigurationError(
                    f"Tool '{self.name}' needs its injected argument '{name}' ({injection.ref()})"
                )
        return {**self._validate(kwargs), **injected}

    def _failure(self, error: Exception, attempts: int) -> ToolExecutionError:
        if self.timeout and isinstance(error, (concurrent.futures.TimeoutError, asyncio.TimeoutError)):
            return ToolTimeoutError(f"Tool '{self.name}' timed out after {self.timeout}s ({attempts} attempt(s))")
        return ToolExecutionError(f"Tool '{self.name}' failed after {attempts} attempt(s): {error}")

    def execute(self, **kwargs: Any) -> Any:
        """Synchronously execute the tool with validation, retries and timeout."""
        if self.is_async:
            return run_coroutine_sync(self.aexecute(**kwargs))

        call_args = self._call_args(kwargs)
        attempt = 0
        while True:
            try:
                return call_with_timeout(self.func, self.timeout, **call_args)
            except Exception as e:
                if is_control_flow(e):
                    raise
                if not self._retrier.should_retry(e, attempt):
                    raise self._failure(e, attempt + 1) from e
                time.sleep(self._retrier.delay(attempt))
                attempt += 1

    async def aexecute(self, **kwargs: Any) -> Any:
        """Asynchronously execute the tool with validation, retries and timeout."""
        call_args = self._call_args(kwargs)
        attempt = 0
        while True:
            try:
                if self.is_async:
                    coro = self.func(**call_args)
                else:
                    coro = asyncio.to_thread(self.func, **call_args)
                if self.timeout:
                    return await asyncio.wait_for(coro, timeout=self.timeout)
                return await coro
            except Exception as e:
                if is_control_flow(e):
                    raise
                if not self._retrier.should_retry(e, attempt):
                    raise self._failure(e, attempt + 1) from e
                await asyncio.sleep(self._retrier.delay(attempt))
                attempt += 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling the tool directly."""
        if args:
            sig = inspect.signature(self.func)
            bound = sig.bind_partial(*args, **kwargs)
            kwargs = bound.arguments
        return self.execute(**kwargs)

    def to_ir(self, registry: Registry | None = None) -> ToolIR:
        """Plain-data description of this tool (registered in `registry` when given)."""
        try:
            input_schema = self.args_schema.model_json_schema() if self.args_schema else {}
        except (PydanticUserError, TypeError):  # arbitrary argument types may have no JSON schema
            input_schema = {}
        ref = registry.register(self, "tool", self.name) if registry is not None else f"tool:{self.name}"
        return ToolIR(
            name=self.name,
            ref=ref,
            description=self.description,
            input_schema=input_schema,
            is_async=self.is_async,
            timeout=self.timeout,
            retries=self.retries,
            retry=lower_retry(self.retry, registry, self.name),
            requires_approval=self.requires_approval,
            injected={name: injection.ref() for name, injection in self.injected.items()},
        )

    def to_langchain_tool(self) -> Any:
        """Convert to LangChain StructuredTool for native LangChain/LangGraph agent execution."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.execute,
            coroutine=self.aexecute,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


def tool(
    name_or_func: str | Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    timeout: float | None = None,
    retries: int = 0,
    requires_approval: bool = False,
    retry: RetryPolicy | None = None,
) -> Any:
    """Decorator to define a Graphflow Tool.

    Usage:
        @tool
        def search(query: str) -> list[str]:
            '''Searches for documents.'''
            return ["doc1", "doc2"]

        @tool(name="lookup", retries=2)
        def lookup_order(order_id: str) -> dict:
            return {"order_id": order_id}
    """
    def decorator(fn: Callable[..., Any]) -> Tool:
        return Tool(
            func=fn,
            name=name or (name_or_func if isinstance(name_or_func, str) else None),
            description=description,
            args_schema=args_schema,
            timeout=timeout,
            retries=retries,
            requires_approval=requires_approval,
            retry=retry,
        )

    if callable(name_or_func):
        return decorator(name_or_func)

    return decorator
