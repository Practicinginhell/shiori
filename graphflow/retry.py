"""Retry policies for nodes and tools.

``RetryPolicy`` has the same fields and defaults as LangGraph's
``langgraph.types.RetryPolicy`` (and accepts one), but lives here so the
builder layer does not depend on LangGraph. Following LangGraph's guidance,
the default ``retry_on`` retries only transient failures (connection errors,
timeouts, HTTP 5xx / 429); programming and validation errors fail at once.

Lowered IR stores a policy as plain data (``RetryIR``); a custom ``retry_on``
becomes a registry reference.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from graphflow.errors import SETUP_ERRORS, ConfigurationError, ModelError
from graphflow.ir.registry import Registry
from graphflow.ir.retry import RetryIR

RetryOn = type[BaseException] | Sequence[type[BaseException]] | Callable[[BaseException], bool]

# Errors that retrying cannot fix, whatever the policy says.
_NEVER_RETRY = SETUP_ERRORS

# Before Python 3.11 the asyncio / futures timeouts are not builtins.TimeoutError.
_TIMEOUTS = (TimeoutError, asyncio.TimeoutError, concurrent.futures.TimeoutError)

# The backoff of the legacy ``retries=n`` argument.
_LEGACY_INITIAL_INTERVAL = 0.05


def status_code(exc: BaseException) -> int | None:
    """HTTP status carried by common client errors (httpx, requests, openai, anthropic)."""
    for holder in (exc, getattr(exc, "response", None)):
        code = getattr(holder, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def is_transient_error(exc: BaseException) -> bool:
    """Default ``retry_on``: True for failures that a later attempt may not hit.

    Connection errors and timeouts are transient; HTTP errors are transient for
    429 and 5xx; everything else (ValueError, KeyError, validation errors, ...)
    is not. Unlike LangGraph's default, timeouts are retried.
    """
    if isinstance(exc, _NEVER_RETRY):
        return False
    if isinstance(exc, ModelError):
        return exc.transient
    if isinstance(exc, (ConnectionError, *_TIMEOUTS)):
        return True
    code = status_code(exc)
    if code is not None:
        return code == 429 or 500 <= code < 600
    name = type(exc).__name__
    return name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}


@dataclass(frozen=True)
class RetryPolicy:
    """How a node or tool retries a failed attempt.

    Args:
        max_attempts: Total attempts, including the first.
        initial_interval: Seconds before the first retry.
        backoff_factor: Multiplier applied to the interval after each retry.
        max_interval: Upper bound on the interval.
        jitter: Add up to 100% random extra delay to spread out retries.
        retry_on: Exception type(s), or a predicate ``(exc) -> bool``, that should
            be retried. Defaults to transient errors only (``is_transient_error``).

    Timeouts are retried by default. Python cannot stop a synchronous function
    that timed out, so that attempt keeps running in the background while the
    next one starts: only retry timeouts of work that is safe to repeat.
    """

    max_attempts: int = 3
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    jitter: bool = True
    retry_on: RetryOn | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigurationError("RetryPolicy.max_attempts must be >= 1")
        if self.initial_interval < 0 or self.max_interval < 0 or self.backoff_factor < 1:
            raise ConfigurationError("RetryPolicy intervals must be >= 0 and backoff_factor >= 1")

    @classmethod
    def coerce(cls, policy: Any) -> RetryPolicy:
        """Accepts a RetryPolicy or LangGraph's RetryPolicy (same field names)."""
        if isinstance(policy, cls):
            return policy
        fields = ("max_attempts", "initial_interval", "backoff_factor", "max_interval", "jitter", "retry_on")
        if all(hasattr(policy, f) for f in fields):
            return cls(**{f: getattr(policy, f) for f in fields})
        raise ConfigurationError(f"Expected a RetryPolicy, got {type(policy).__name__}")

    def to_langgraph(self) -> Any:
        """The equivalent ``langgraph.types.RetryPolicy``."""
        from langgraph.types import RetryPolicy as LangGraphRetryPolicy

        return LangGraphRetryPolicy(
            initial_interval=self.initial_interval,
            backoff_factor=self.backoff_factor,
            max_interval=self.max_interval,
            max_attempts=self.max_attempts,
            jitter=self.jitter,
            retry_on=_predicate(self.retry_on),
        )


def backoff_delay(policy: RetryPolicy | RetryIR, attempt: int) -> float:
    """Seconds to wait after failed attempt number `attempt` (0-based)."""
    delay = min(policy.max_interval, policy.initial_interval * policy.backoff_factor ** attempt)
    if policy.jitter:
        delay += random.uniform(0, delay)
    return delay


def _predicate(retry_on: RetryOn | None) -> Callable[[BaseException], bool]:
    if retry_on is None:
        return is_transient_error
    if isinstance(retry_on, type) and issubclass(retry_on, BaseException):
        types: tuple[type[BaseException], ...] = (retry_on,)
    elif isinstance(retry_on, Sequence) and not isinstance(retry_on, str):
        types = tuple(retry_on)
    elif callable(retry_on):
        return retry_on
    else:
        raise ConfigurationError(f"retry_on must be exception type(s) or a predicate, got {retry_on!r}")
    return lambda exc: isinstance(exc, types)


# ---------------------------------------------------------------------- #
# Lowering and resolution
# ---------------------------------------------------------------------- #

def check_retry_args(retries: int, retry: Any, owner: str) -> RetryPolicy | None:
    """Validates the (legacy) `retries` / `retry` pair given to a node or tool."""
    if retries < 0:
        raise ConfigurationError(f"{owner}: retries must be >= 0")
    if retry is None:
        return None
    if retries:
        raise ConfigurationError(f"{owner}: pass either retries or retry, not both")
    return RetryPolicy.coerce(retry)


def lower_retry(policy: RetryPolicy | None, registry: Registry | None, owner: str) -> RetryIR | None:
    """Plain-data form of `policy`; a custom retry_on is registered in `registry`."""
    if policy is None:
        return None
    retry_on = None
    if policy.retry_on is not None:
        predicate = _predicate(policy.retry_on)
        retry_on = registry.register(predicate, "retry_on", owner) if registry is not None else f"retry_on:{owner}"
    return RetryIR(
        max_attempts=policy.max_attempts,
        initial_interval=policy.initial_interval,
        backoff_factor=policy.backoff_factor,
        max_interval=policy.max_interval,
        jitter=policy.jitter,
        retry_on=retry_on,
    )


@dataclass(frozen=True)
class Retrier:
    """A resolved retry policy: how many attempts, how long to wait, what to retry."""

    attempts: int
    delay: Callable[[int], float]
    retryable: Callable[[BaseException], bool]

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        """Whether failed attempt `attempt` (0-based) should be followed by another."""
        return attempt + 1 < self.attempts and not isinstance(exc, _NEVER_RETRY) and self.retryable(exc)


def legacy_retrier(retries: int) -> Retrier:
    """``retries=n``: n extra attempts on any error, with a short exponential backoff."""
    return Retrier(
        attempts=retries + 1,
        delay=lambda attempt: _LEGACY_INITIAL_INTERVAL * 2 ** attempt,
        retryable=lambda exc: True,
    )


def resolve_retrier(retry: RetryIR | RetryPolicy | None, retries: int, registry: Registry | None) -> Retrier:
    """Builds the Retrier for a node/tool from its IR (or builder-stage policy)."""
    if retry is None:
        return legacy_retrier(retries)
    if isinstance(retry, RetryPolicy):
        predicate = _predicate(retry.retry_on)
    elif retry.retry_on is None:
        predicate = is_transient_error
    else:
        if registry is None:
            raise ConfigurationError(f"retry_on reference '{retry.retry_on}' needs a registry")
        predicate = registry.resolve(retry.retry_on)
    return Retrier(attempts=retry.max_attempts, delay=lambda attempt: backoff_delay(retry, attempt), retryable=predicate)
