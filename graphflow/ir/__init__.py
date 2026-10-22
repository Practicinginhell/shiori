"""Graphflow Intermediate Representation (IR).

Engine-independent, plain-data description of a workflow. Code and other live
objects are referenced by string keys into a ``Registry``.
"""

from graphflow.ir.agent import AgentIR, ToolIR
from graphflow.ir.graph import (
    END,
    HITLIR,
    ConditionalEdgeIR,
    DynamicFanOutIR,
    EdgeIR,
    GraphIR,
    LoopIR,
    LoweredWorkflow,
    NodeIR,
    ParallelBranchIR,
)
from graphflow.ir.registry import Registry, is_ref
from graphflow.ir.state import FieldIR, ReducerType, StateSchemaIR
from graphflow.ir.validate import validate_graph

__all__ = [
    "END",
    "HITLIR",
    "AgentIR",
    "ConditionalEdgeIR",
    "DynamicFanOutIR",
    "EdgeIR",
    "FieldIR",
    "GraphIR",
    "LoopIR",
    "LoweredWorkflow",
    "NodeIR",
    "ParallelBranchIR",
    "ReducerType",
    "Registry",
    "StateSchemaIR",
    "ToolIR",
    "is_ref",
    "validate_graph",
]
