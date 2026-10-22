"""Unit tests for Graphflow Tools system."""

import pytest
from pydantic import BaseModel

from graphflow import tool
from graphflow.core.errors import ToolExecutionError


def test_basic_tool_decorator():
    @tool
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    assert add.name == "add"
    assert "Adds two numbers" in add.description
    result = add(a=3, b=5)
    assert result == 8


def test_tool_pydantic_schema_validation():
    class SearchInput(BaseModel):
        query: str
        limit: int = 5

    @tool(args_schema=SearchInput)
    def search(query: str, limit: int = 5) -> list[str]:
        return [f"result_{i}" for i in range(limit)]

    res = search(query="langgraph", limit=3)
    assert len(res) == 3


@pytest.mark.asyncio
async def test_async_tool_execution():
    @tool
    async def async_fetch(url: str) -> str:
        return f"content from {url}"

    assert async_fetch.is_async is True
    res = await async_fetch.aexecute(url="https://example.com")
    assert res == "content from https://example.com"


def test_tool_retry_behavior():
    attempts = 0

    @tool(retries=2)
    def flaky_tool():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Temporary failure")
        return "success"

    res = flaky_tool()
    assert res == "success"
    assert attempts == 3


def test_tool_failure_exhaustion():
    @tool(retries=1)
    def failing_tool():
        raise RuntimeError("Permanent failure")

    with pytest.raises(ToolExecutionError):
        failing_tool()
