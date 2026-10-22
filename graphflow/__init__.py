"""Graphflow: Production-grade Universal Framework for LangGraph.

Build general-purpose AI applications with clean, high-level declarative APIs
while maintaining LangGraph as the reliable execution engine underneath.
"""

from typing import Any

from graphflow.approvals import ApprovalDecision, ApprovalRequest, ToolDecision
from graphflow.context import FromContext, FromState, get_context
from graphflow.core.agent import Agent
from graphflow.core.application import Application, CompiledApplication
from graphflow.core.state import Field, State, create_state_schema
from graphflow.core.thread import Thread
from graphflow.core.workflow import Workflow
from graphflow.errors import (
    CompilationError,
    ConfigurationError,
    ExecutionError,
    GraphflowError,
    HumanApprovalInterrupted,
    ModelError,
    StateValidationError,
    TimeoutError,
    ToolExecutionError,
    ToolTimeoutError,
)
from graphflow.events import (
    CustomEvent,
    HumanApprovalRequired,
    ModelToken,
    NodeCompleted,
    NodeStarted,
    RunCompleted,
    RunFailed,
    RunInterrupted,
    RunStarted,
    StateUpdated,
    StreamEvent,
    ToolCalled,
    ToolCompleted,
    emit_event,
)
from graphflow.ir.graph import END
from graphflow.middleware import AgentMiddleware, HistoryLimit, ModelRequest, ToolRequest
from graphflow.models.base import BaseModelAdapter, ModelResponse, ToolCall
from graphflow.models.langchain import LangChainModel
from graphflow.models.mock import MockModel
from graphflow.models.openai import OpenAIModel
from graphflow.persistence.base import BaseCheckpointStore
from graphflow.persistence.memory import InMemoryCheckpointStore
from graphflow.retry import RetryPolicy
from graphflow.runtime.checkpoints import Checkpoint
from graphflow.runtime.runner import RunResult
from graphflow.tools.base import Tool, tool

__version__ = "0.2.0"

__all__ = [
    "END",
    "Agent",
    "AgentMiddleware",
    "Application",
    "ApprovalDecision",
    "ApprovalRequest",
    "BaseCheckpointStore",
    "BaseModelAdapter",
    "Checkpoint",
    "CompilationError",
    "CompiledApplication",
    "ConfigurationError",
    "CustomEvent",
    "ExecutionError",
    "Field",
    "FromContext",
    "FromState",
    "GraphflowError",
    "HistoryLimit",
    "HumanApprovalInterrupted",
    "HumanApprovalRequired",
    "InMemoryCheckpointStore",
    "LangChainModel",
    "MockModel",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "ModelToken",
    "NodeCompleted",
    "NodeStarted",
    "OpenAIModel",
    "RetryPolicy",
    "RunCompleted",
    "RunFailed",
    "RunInterrupted",
    "RunResult",
    "RunStarted",
    "SqliteCheckpointStore",
    "State",
    "StateUpdated",
    "StateValidationError",
    "StreamEvent",
    "Thread",
    "TimeoutError",
    "Tool",
    "ToolCall",
    "ToolCalled",
    "ToolCompleted",
    "ToolDecision",
    "ToolExecutionError",
    "ToolRequest",
    "ToolTimeoutError",
    "Workflow",
    "create_state_schema",
    "emit_event",
    "get_context",
    "tool",
]


def __getattr__(name: str) -> Any:
    # Optional dependency (graphflow[sqlite]); imported on first use.
    if name == "SqliteCheckpointStore":
        from graphflow.persistence.sqlite import SqliteCheckpointStore

        return SqliteCheckpointStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
