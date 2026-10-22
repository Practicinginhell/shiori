"""Wraps workflow node functions with retries, timeouts, error handling and update normalization.

Each node becomes a LangChain ``RunnableLambda`` with both a sync and an async
implementation, so the same compiled graph serves ``invoke``/``stream`` and
``ainvoke``/``astream`` natively.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import inspect
import time
import typing
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.types import Command, Send

from graphflow._async import call_with_timeout, is_control_flow, run_coroutine_sync
from graphflow.errors import SETUP_ERRORS, ExecutionError, GraphflowError, TimeoutError
from graphflow.ir.graph import RESERVED_PREFIX, ErrorHandler, NodeIR
from graphflow.ir.state import StateSchemaIR
from graphflow.retry import Retrier, legacy_retrier
from graphflow.validation import SchemaValidator

SyncCall = Callable[[dict[str, Any], RunnableConfig], Any]
AsyncCall = Callable[[dict[str, Any], RunnableConfig], Any]


# ---------------------------------------------------------------------- #
# Calling conventions
# ---------------------------------------------------------------------- #

def _accepts_config(func: Callable[..., Any]) -> bool:
    try:
        params = [
            p for p in inspect.signature(func).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return False
    return len(params) >= 2


def _is_state_class(obj: Any) -> bool:
    # Duck-typed so the backend doesn't depend on graphflow.core.
    return isinstance(obj, type) and hasattr(obj, "_gf_schema") and hasattr(obj, "to_dict")


def _state_class(func: Callable[..., Any]) -> type | None:
    """The State subclass the node's first parameter is annotated with, if any."""
    try:
        hints = typing.get_type_hints(func)
        first = next(iter(inspect.signature(func).parameters))
    except (NameError, TypeError, ValueError, StopIteration):
        return None
    ann = hints.get(first)
    return ann if _is_state_class(ann) else None


def _is_async_callable(func: Any) -> bool:
    # An object whose __call__ is `async def` (not a callable check, hence the noqa).
    return inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(getattr(func, "__call__", None))  # noqa: B004


def make_invokers(func: Any) -> tuple[SyncCall, AsyncCall]:
    """Returns (sync, async) callables with signature (state, config) for any supported node target."""
    # Objects following the executable protocol (e.g. Agent, sub-workflow executors).
    if hasattr(func, "execute") and hasattr(func, "aexecute"):
        accepts = _accepts_config(func.execute)

        def sync_exec(state: dict[str, Any], config: RunnableConfig) -> Any:
            return func.execute(state, config) if accepts else func.execute(state)

        async def async_exec(state: dict[str, Any], config: RunnableConfig) -> Any:
            return await (func.aexecute(state, config) if accepts else func.aexecute(state))

        return sync_exec, async_exec

    accepts = _accepts_config(func)
    state_cls = _state_class(func)

    def args(state: dict[str, Any], config: RunnableConfig) -> tuple[Any, ...]:
        arg = state_cls(**state) if state_cls is not None else state
        return (arg, config) if accepts else (arg,)

    if _is_async_callable(func):
        async def async_fn(state: dict[str, Any], config: RunnableConfig) -> Any:
            return await func(*args(state, config))

        def sync_fn(state: dict[str, Any], config: RunnableConfig) -> Any:
            return run_coroutine_sync(func(*args(state, config)))

        return sync_fn, async_fn

    def sync_plain(state: dict[str, Any], config: RunnableConfig) -> Any:
        return func(*args(state, config))

    async def async_plain(state: dict[str, Any], config: RunnableConfig) -> Any:
        # asyncio.to_thread copies context vars, keeping LangGraph's config/writer available.
        return await asyncio.to_thread(func, *args(state, config))

    return sync_plain, async_plain


# ---------------------------------------------------------------------- #
# Update normalization
# ---------------------------------------------------------------------- #

def _diff_update(before: Mapping[str, Any], after: Mapping[str, Any], schema: StateSchemaIR) -> dict[str, Any]:
    """Computes the update a node made when it returns a mutated State object."""
    update: dict[str, Any] = {}
    for key, new in after.items():
        if key in before and before[key] == new:
            continue
        old = before.get(key)
        f = schema.fields.get(key)
        reducer = f.reducer if f else "replace"
        if reducer == "append" and isinstance(old, list) and isinstance(new, list) and new[: len(old)] == old:
            update[key] = new[len(old):]
        elif reducer == "sum" and isinstance(old, (int, float)) and isinstance(new, (int, float)):
            update[key] = new - old
        elif reducer == "merge_dict" and isinstance(old, dict) and isinstance(new, dict):
            update[key] = {k: v for k, v in new.items() if k not in old or old[k] != v}
        else:
            update[key] = new
    return update


def normalize_update(res: Any, input_state: Mapping[str, Any], schema: StateSchemaIR, node_name: str) -> Any:
    """Turns a node's return value into something LangGraph accepts as an update."""
    if res is None:
        return {}
    if isinstance(res, (Command, Send)):
        return res
    if isinstance(res, (list, tuple)) and res and all(isinstance(r, (Command, Send)) for r in res):
        return list(res)
    if _is_state_class(type(res)):
        return _diff_update(input_state, res.to_dict(), schema)
    if isinstance(res, Mapping):
        return dict(res)
    if hasattr(res, "model_dump"):
        return res.model_dump(exclude_unset=True)
    if dataclasses.is_dataclass(res) and not isinstance(res, type):
        return dataclasses.asdict(res)
    if isinstance(res, (str, int, float, bool, list, tuple)):
        return {"output": res}
    raise ExecutionError(
        f"Node '{node_name}' returned unsupported type {type(res).__name__}; "
        "return a dict of state updates, a State, a Command, or None."
    )


# ---------------------------------------------------------------------- #
# Node builder
# ---------------------------------------------------------------------- #

def _handler_arity(handler: Callable[..., Any]) -> int:
    try:
        return len(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return 2


def isolation_handler(node_name: str) -> ErrorHandler:
    """Error handler recording ``"<node>: <error>"`` in the `errors` state field."""
    def record(state: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {"errors": [f"{node_name}: {error}"]}

    return record


def _locate(error: Exception, node: str, attempts: int) -> None:
    """Records the innermost node an error came from (outer nodes leave it alone)."""
    if getattr(error, "node", None) is not None:
        return
    try:
        error.node, error.attempts = node, attempts  # type: ignore[attr-defined]
    except AttributeError:  # an exception type that forbids new attributes
        return
    if hasattr(error, "add_note"):  # Python 3.11+
        error.add_note(f"(in node '{node}', after {attempts} attempt(s))")


def _failure(node_ir: NodeIR, error: Exception, attempts: int) -> Exception:
    """The exception a node raises once it gives up: graphflow errors keep their type."""
    if node_ir.timeout_seconds and isinstance(error, (concurrent.futures.TimeoutError, asyncio.TimeoutError)) \
            and not isinstance(error, GraphflowError):
        failure: Exception = TimeoutError(
            f"Node '{node_ir.name}' timed out after {node_ir.timeout_seconds}s ({attempts} attempt(s))"
        )
    elif isinstance(error, GraphflowError):
        failure = error
    else:
        failure = ExecutionError(f"Node '{node_ir.name}' failed after {attempts} attempt(s): {error}")
    _locate(failure, node_ir.name, attempts)
    return failure


def _raise_failure(node_ir: NodeIR, error: Exception, attempts: int) -> typing.NoReturn:
    failure = _failure(node_ir, error, attempts)
    if failure is error:
        raise error
    raise failure from error


def create_node_executor(
    node_ir: NodeIR,
    schema: StateSchemaIR | None = None,
    error_handler: ErrorHandler | None = None,
    target: Any = None,
    validator: SchemaValidator | None = None,
    retrier: Retrier | None = None,
) -> RunnableLambda:
    """Creates a runnable suitable for LangGraph StateGraph.add_node.

    Args:
        schema: The compiled state schema (used to diff returned State objects).
        error_handler: ``(state, error[, node_name]) -> update`` used once the last attempt fails.
        target: The resolved node callable / executor object.
        validator: When given, every update the node returns is type-checked.
        retrier: The node's resolved retry policy (default: its legacy ``retries``).
    """
    schema = schema or StateSchemaIR()
    sync_call, async_call = make_invokers(target)
    handler = error_handler
    retrier = retrier or legacy_retrier(node_ir.retries)
    timeout, name = node_ir.timeout_seconds, node_ir.name

    def finish(res: Any, state: dict[str, Any]) -> Any:
        update = normalize_update(res, state, schema, name)
        if validator is None:
            return update
        where = f"update from node '{name}'"
        if isinstance(update, dict):
            return validator.validate(update, where)
        if isinstance(update, Command) and isinstance(update.update, dict):
            return dataclasses.replace(update, update=validator.validate(update.update, where))
        return update

    def call_handler_sync(state: dict[str, Any], error: Exception) -> Any:
        assert handler is not None
        args = (state, error, name) if _handler_arity(handler) >= 3 else (state, error)
        out = handler(*args)
        if inspect.isawaitable(out):
            out = run_coroutine_sync(out)
        return finish(out, state)

    async def call_handler_async(state: dict[str, Any], error: Exception) -> Any:
        assert handler is not None
        args = (state, error, name) if _handler_arity(handler) >= 3 else (state, error)
        out = handler(*args)
        if inspect.isawaitable(out):
            out = await out
        return finish(out, state)

    # Retries cover running the node; validating its result happens once, outside
    # the retry loop, so a bad update fails immediately with StateValidationError.
    def run_sync(state: dict[str, Any], config: RunnableConfig) -> Any:
        attempt = 0
        while True:
            try:
                res = call_with_timeout(sync_call, timeout, state, config)
            except Exception as e:
                if is_control_flow(e) or isinstance(e, SETUP_ERRORS):
                    raise
                if retrier.should_retry(e, attempt):
                    time.sleep(retrier.delay(attempt))
                    attempt += 1
                    continue
                if handler is not None:
                    return call_handler_sync(state, e)
                _raise_failure(node_ir, e, attempt + 1)
            return finish(res, state)

    async def run_async(state: dict[str, Any], config: RunnableConfig) -> Any:
        attempt = 0
        while True:
            try:
                coro = async_call(state, config)
                res = await (asyncio.wait_for(coro, timeout) if timeout else coro)
            except Exception as e:
                if is_control_flow(e) or isinstance(e, SETUP_ERRORS):
                    raise
                if retrier.should_retry(e, attempt):
                    await asyncio.sleep(retrier.delay(attempt))
                    attempt += 1
                    continue
                if handler is not None:
                    return await call_handler_async(state, e)
                _raise_failure(node_ir, e, attempt + 1)
            return finish(res, state)

    return RunnableLambda(run_sync, afunc=run_async, name=name)


# ---------------------------------------------------------------------- #
# Sub-workflows
# ---------------------------------------------------------------------- #

class SubgraphExecutor:
    """Runs a compiled child graph as a single node of a parent graph.

    The child's per-node updates are folded with the child's reducers into one
    update, so the parent applies each change exactly once (returning the child's
    final state instead would re-append every list the parent already has).
    Interrupts inside the child propagate, and resume through the parent thread.
    """

    def __init__(self, compiled_graph: Any, reducers: dict[str, Callable[[Any, Any], Any]]):
        self.compiled_graph = compiled_graph
        self.reducers = reducers

    def _fold(self, acc: dict[str, Any], chunk: Mapping[str, Any]) -> None:
        for update in chunk.values():
            if not isinstance(update, Mapping):
                continue
            for key, value in update.items():
                if key.startswith(RESERVED_PREFIX):
                    continue
                if key in acc and key in self.reducers:
                    acc[key] = self.reducers[key](acc[key], value)
                else:
                    acc[key] = value

    def execute(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        acc: dict[str, Any] = {}
        for chunk in self.compiled_graph.stream(state, config, stream_mode="updates"):
            self._fold(acc, chunk)
        return acc

    async def aexecute(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        acc: dict[str, Any] = {}
        async for chunk in self.compiled_graph.astream(state, config, stream_mode="updates"):
            self._fold(acc, chunk)
        return acc
