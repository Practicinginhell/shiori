"""Graphflow IR - Agent and Tool definitions (plain data; live objects are registry refs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphflow.ir.retry import RetryIR


@dataclass
class ToolIR:
    """A tool available to an agent."""
    name: str
    ref: str  # registry ref of the Tool object ("tool:..." or "apptool:<name>" for app-registered tools)
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)  # JSON schema of the arguments
    is_async: bool = False
    timeout: float | None = None
    retries: int = 0
    retry: RetryIR | None = None
    requires_approval: bool = False
    injected: dict[str, str] = field(default_factory=dict)  # argument -> 'context:<key>' or 'state:<key>'


@dataclass
class AgentIR:
    """An agent, compiled into its own subgraph (model -> [approve] -> tools -> model ... -> finalize)."""
    name: str
    instructions: str
    model: str  # registry ref of the model adapter
    model_repr: str = ""
    tools: list[ToolIR] = field(default_factory=list)
    structured_output: str | None = None  # registry ref of the output type
    structured_output_repr: str = ""
    memory_enabled: bool = True
    max_tool_iterations: int = 10
    middleware: list[str] = field(default_factory=list)  # registry refs
    approval_handler: str | None = None  # registry ref
    # tool name -> {"allowed_decisions": [...], "description": str}, or None to never ask
    interrupt_on: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    structured_output_retries: int = 1
