"""RetryPolicy: exponential backoff with jitter, retrying only transient errors.

Follows LangGraph's error-handling guidance: transient failures (network, rate
limits, timeouts) are retried by the system; programming and validation errors
fail immediately.
"""

import asyncio
import concurrent.futures

import pytest

from graphflow import Application, ExecutionError, RetryPolicy, State, Workflow, tool
from graphflow.errors import StateValidationError, ToolExecutionError
from graphflow.retry import backoff_delay, is_transient_error


class S(State):
    output: str = ""


def run(wf):
    return Application("a").register(wf).run({})


def test_policy_retries_transient_errors_until_success():
    attempts = []

    def flaky(state):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("network blip")
        return {"output": "ok"}

    wf = Workflow("w", state_schema=S).then(flaky, retry=RetryPolicy(max_attempts=3, initial_interval=0.001))
    assert run(wf)["output"] == "ok"
    assert len(attempts) == 3


def test_policy_does_not_retry_programming_errors():
    attempts = []

    def buggy(state):
        attempts.append(1)
        raise KeyError("missing")

    wf = Workflow("w", state_schema=S).then(buggy, retry=RetryPolicy(max_attempts=5, initial_interval=0.001))
    with pytest.raises(ExecutionError, match="failed after 1 attempt"):
        run(wf)
    assert len(attempts) == 1


def test_policy_custom_retry_on_exception_types():
    attempts = []

    class RateLimited(Exception):
        pass

    def limited(state):
        attempts.append(1)
        raise RateLimited()

    policy = RetryPolicy(max_attempts=2, initial_interval=0.001, retry_on=RateLimited)
    with pytest.raises(ExecutionError, match="failed after 2 attempt"):
        run(Workflow("w", state_schema=S).then(limited, retry=policy))
    assert len(attempts) == 2


def test_policy_on_error_handler_runs_after_attempts_are_exhausted():
    attempts = []

    def down(state):
        attempts.append(1)
        raise ConnectionError("down")

    wf = Workflow("w", state_schema=S).then(
        down,
        retry=RetryPolicy(max_attempts=2, initial_interval=0.001),
        on_error=lambda state, error: {"output": f"fallback: {error}"},
    )
    assert run(wf)["output"] == "fallback: down"
    assert len(attempts) == 2


def test_policy_async_run():
    attempts = []

    async def flaky(state):
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("blip")
        return {"output": "ok"}

    app = Application("a").register(
        Workflow("w", state_schema=S).then(flaky, retry=RetryPolicy(max_attempts=2, initial_interval=0.001))
    )
    assert asyncio.run(app.arun({}))["output"] == "ok"


def test_retries_and_retry_are_mutually_exclusive():
    with pytest.raises(Exception, match="retries or retry"):
        Workflow("w").then(lambda s: {}, retries=2, retry=RetryPolicy())


def test_legacy_retries_still_retry_any_error():
    attempts = []

    def flaky(state):
        attempts.append(1)
        if len(attempts) < 2:
            raise ValueError("flaky")
        return {"output": "ok"}

    assert run(Workflow("w", state_schema=S).then(flaky, retries=1))["output"] == "ok"


def test_validation_errors_are_never_retried():
    attempts = []

    def bad(state):
        attempts.append(1)
        raise StateValidationError("bad update")

    with pytest.raises(Exception):
        run(Workflow("w", state_schema=S).then(bad, retry=RetryPolicy(max_attempts=3, retry_on=Exception)))
    assert len(attempts) == 1


def test_transient_error_classification():
    assert is_transient_error(ConnectionError())
    assert is_transient_error(TimeoutError())
    assert is_transient_error(asyncio.TimeoutError())
    assert is_transient_error(concurrent.futures.TimeoutError())
    assert not is_transient_error(ValueError())
    assert not is_transient_error(KeyError())
    assert not is_transient_error(StateValidationError("x"))


def test_backoff_is_exponential_and_capped():
    policy = RetryPolicy(initial_interval=1.0, backoff_factor=2.0, max_interval=5.0, jitter=False)
    assert [backoff_delay(policy, i) for i in range(4)] == [1.0, 2.0, 4.0, 5.0]
    jittered = RetryPolicy(initial_interval=1.0, jitter=True)
    assert all(1.0 <= backoff_delay(jittered, 0) <= 2.0 for _ in range(20))


def test_tool_retry_policy():
    attempts = []

    @tool(retry=RetryPolicy(max_attempts=3, initial_interval=0.001))
    def fetch(url: str) -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("reset")
        return "page"

    assert fetch.execute(url="u") == "page"

    attempts.clear()

    @tool(retry=RetryPolicy(max_attempts=3, initial_interval=0.001))
    def parse(text: str) -> str:
        attempts.append(1)
        raise ValueError("bad input")

    with pytest.raises(ToolExecutionError, match="1 attempt"):
        parse.execute(text="t")
    assert len(attempts) == 1


def test_retry_policy_is_plain_data_in_the_ir():
    wf = Workflow("w").then(lambda s: {}, name="n", retry=RetryPolicy(max_attempts=4, retry_on=ConnectionError))
    node = wf.to_ir().nodes["n"]
    assert node.retry.max_attempts == 4
    assert isinstance(node.retry.retry_on, str)  # a registry reference, not a live object


def test_setup_errors_skip_on_error_and_keep_their_type():
    from graphflow import ConfigurationError

    handled = []

    def misconfigured(state):
        raise ConfigurationError("missing API key")

    wf = Workflow("w", state_schema=S).then(
        misconfigured, retry=RetryPolicy(max_attempts=3, retry_on=Exception),
        on_error=lambda state, error: handled.append(error) or {"output": "fallback"},
    )
    with pytest.raises(ConfigurationError, match="missing API key"):
        run(wf)
    assert handled == []  # a fallback would hide the mistake
