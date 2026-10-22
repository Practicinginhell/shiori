"""Sync/async bridging helpers.

Graphflow prefers native sync and async code paths (LangGraph provides both).
These helpers exist only for the few places where a coroutine must be driven
from synchronous code, e.g. an ``async def`` node inside ``app.run()``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Runs `coro` to completion from synchronous code.

    Uses ``asyncio.run`` when this thread has no running event loop. Otherwise
    (e.g. inside Jupyter) runs it on a fresh loop in a worker thread, carrying
    over context variables so LangGraph's config/stream writer stay available.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    ctx = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()


def is_control_flow(error: BaseException) -> bool:
    """True for LangGraph control-flow signals (interrupt, parent Command) that must propagate."""
    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:  # pragma: no cover
        return False
    return isinstance(error, GraphBubbleUp)


def call_with_timeout(fn: Callable[..., T], timeout: float | None, *args: Any, **kwargs: Any) -> T:
    """Calls a synchronous function, raising ``concurrent.futures.TimeoutError`` after `timeout`.

    Python cannot kill a running thread: on timeout the worker is abandoned and
    finishes in the background, but the caller regains control immediately.
    """
    if not timeout:
        return fn(*args, **kwargs)
    ctx = contextvars.copy_context()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(ctx.run, fn, *args, **kwargs)
        return future.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False)
