"""Graphflow Compiler: Translates Graphflow GraphIR into a LangGraph StateGraph.

The IR is plain data; the compiler resolves its references through the
Registry. Compiler-generated nodes and state channels use the ``__gf_`` prefix
and are hidden from stream events and results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from graphflow.approvals import decision_fields
from graphflow.errors import ExecutionError, StateValidationError
from graphflow.graph.agent_graph import AgentExecutor, build_agent_graph
from graphflow.graph.node_wrapper import SubgraphExecutor, create_node_executor, isolation_handler
from graphflow.graph.state_builder import (
    ResolvedField,
    build_langgraph_state_schema,
    build_validator,
    effective_schema,
    resolve_schema,
)
from graphflow.ir.graph import (
    HITLIR,
    ConditionalEdgeIR,
    DynamicFanOutIR,
    GraphIR,
    LoopIR,
    NodeIR,
)
from graphflow.ir.registry import Registry, is_ref
from graphflow.ir.state import StateSchemaIR
from graphflow.ir.validate import validate_graph
from graphflow.retry import resolve_retrier
from graphflow.validation import SchemaValidator

logger = logging.getLogger("graphflow.compiler")

INIT_NODE = "__gf_init__"
HIDDEN_PREFIX = "__gf_"


def is_hidden(name: str) -> bool:
    """True for LangGraph/Graphflow internal node or channel names."""
    return name.startswith("__")


def _loop_guard_id(loop: LoopIR) -> str:
    return f"{HIDDEN_PREFIX}{loop.id}_guard"


def _loop_counter_key(loop: LoopIR) -> str:
    return f"{HIDDEN_PREFIX}{loop.id}_count"


def _resolve(value: Any, registry: Registry | None) -> Any:
    """Resolves a registry reference; callables (builder-stage IR) pass through."""
    if value is None or callable(value):
        return value
    if is_ref(value):
        if registry is None:
            raise ValueError(f"Reference '{value}' needs a registry; compile a lowered workflow with its registry")
        return registry.resolve(value)
    return value


class LangGraphCompiler:
    """Translates Graphflow IR into a runnable LangGraph StateGraph."""

    def compile_to_state_graph(
        self,
        ir: GraphIR,
        registry: Registry | None = None,
        *,
        interrupts_enabled: bool = True,
    ) -> StateGraph:
        """Translates GraphIR into an uncompiled LangGraph StateGraph (Escape Hatch).

        `interrupts_enabled` tells agents whether they may pause for tool approval
        (only possible when the final graph has a checkpointer).
        """
        validate_graph(ir, registry)
        schema = effective_schema(ir.state_schema)
        fields = resolve_schema(ir.state_schema, registry)
        validator = build_validator(ir.state_schema, registry)
        extra_channels = {_loop_counter_key(loop): int for loop in ir.loops}
        builder = StateGraph(build_langgraph_state_schema(ir.state_schema, extra_channels, registry))

        # 1. Nodes
        graph_handler = _resolve(ir.error_handler, registry)
        update_validator = validator if schema.validation == "updates" else None
        for node_id, node_ir in ir.nodes.items():
            target = self._node_target(node_ir, registry, interrupts_enabled)
            handler = self._error_handler(node_ir, schema, graph_handler, registry)
            retrier = resolve_retrier(node_ir.retry, node_ir.retries, registry)
            builder.add_node(
                node_id, create_node_executor(node_ir, schema, handler, target, update_validator, retrier)
            )

        edits_validator = validator if schema.validation != "off" else None
        for hitl_id, hitl_ir in ir.hitl_nodes.items():
            destinations = tuple(t for t in (hitl_ir.resume_target, hitl_ir.reject_target) if t)
            builder.add_node(
                hitl_id, self._create_hitl_node(hitl_ir, schema, edits_validator), destinations=destinations
            )

        # 2. Entry point (optionally through the defaults initializer)
        entry = ir.entry_point or next(iter(ir.nodes))
        if any(rf.has_default for rf in fields.values()):
            builder.add_node(INIT_NODE, self._create_init_node(fields))
            builder.add_edge(START, INIT_NODE)
            builder.add_edge(INIT_NODE, entry)
        else:
            builder.add_edge(START, entry)

        # 3. Static edges
        for edge in ir.edges:
            builder.add_edge(edge.source, edge.target)

        # 4. Parallel branches: fan out from source, join (wait for all) at fan_in
        for par in ir.parallel_branches:
            for branch_node in par.branch_nodes:
                builder.add_edge(par.source, branch_node)
            if par.fan_in:
                builder.add_edge(list(par.branch_nodes), par.fan_in)

        # 5. Dynamic fan-outs (LangGraph Send API)
        for d in ir.dynamic_fan_outs:
            router = self._make_send_router(d, _resolve(d.partition_fn, registry))
            builder.add_conditional_edges(d.source, router, [d.worker_node, d.fan_in])
            builder.add_edge(d.worker_node, d.fan_in)

        # 6. Conditional edges
        for cedge in ir.conditional_edges:
            condition = self._make_condition(cedge, _resolve(cedge.condition, registry))
            builder.add_conditional_edges(cedge.source, condition, cedge.route_map)

        # 7. Loops: body -> guard; guard routes back to body or to the exit
        for loop in ir.loops:
            exit_dest = loop.exit_node or END
            guard = self._create_loop_guard(loop, _resolve(loop.condition, registry))
            builder.add_node(_loop_guard_id(loop), guard, destinations=(loop.body_node, exit_dest))
            builder.add_edge(loop.body_node, _loop_guard_id(loop))

        # 8. Connect loose ends to END
        has_outbound: set[str] = {e.source for e in ir.edges}
        has_outbound |= {c.source for c in ir.conditional_edges}
        for p in ir.parallel_branches:
            has_outbound.add(p.source)
            if p.fan_in:
                has_outbound.update(p.branch_nodes)
        for d in ir.dynamic_fan_outs:
            has_outbound.update((d.source, d.worker_node))
        has_outbound |= {loop.body_node for loop in ir.loops}
        has_outbound |= set(ir.hitl_nodes)

        for n_id in ir.nodes:
            if n_id not in has_outbound:
                builder.add_edge(n_id, END)

        return builder

    def compile(
        self,
        ir: GraphIR,
        checkpointer: Any = None,
        store: Any = None,
        registry: Registry | None = None,
    ) -> Any:
        """Compiles GraphIR into a runnable Pregel CompiledGraph."""
        builder = self.compile_to_state_graph(ir, registry, interrupts_enabled=checkpointer is not None)
        return builder.compile(checkpointer=checkpointer, store=store)

    # ------------------------------------------------------------------ #
    # Node targets and error handling
    # ------------------------------------------------------------------ #

    def _node_target(self, node_ir: NodeIR, registry: Registry | None, interrupts_enabled: bool) -> Any:
        if node_ir.kind == "agent":
            assert node_ir.agent is not None
            if registry is None:
                raise ValueError("Agent nodes need a registry")
            return AgentExecutor(build_agent_graph(node_ir.agent, registry, interrupts_enabled), node_ir.agent)
        if node_ir.kind == "subgraph":
            assert node_ir.subgraph is not None
            child = self.compile_to_state_graph(node_ir.subgraph, registry, interrupts_enabled=interrupts_enabled)
            reducers = {name: rf.reducer for name, rf in resolve_schema(node_ir.subgraph.state_schema, registry).items()}
            return SubgraphExecutor(child.compile(), reducers)
        return _resolve(node_ir.target, registry)

    @staticmethod
    def _error_handler(
        node_ir: NodeIR, schema: StateSchemaIR, graph_handler: Callable[..., Any] | None, registry: Registry | None
    ) -> Callable[..., Any] | None:
        if node_ir.on_error is not None:
            return _resolve(node_ir.on_error, registry)
        if node_ir.isolate_errors and "errors" in schema.fields:
            return isolation_handler(node_ir.name)
        return graph_handler

    # ------------------------------------------------------------------ #
    # Generated nodes and routers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_init_node(fields: dict[str, ResolvedField]) -> Callable[..., Any]:
        """Fills declared defaults for fields the input (or checkpointed thread) doesn't set."""
        defaults = {name: rf for name, rf in fields.items() if rf.has_default}

        def init_defaults(state: dict[str, Any]) -> dict[str, Any]:
            return {name: rf.default_maker() for name, rf in defaults.items() if name not in state}  # type: ignore[misc]

        return init_defaults

    @staticmethod
    def _make_condition(cedge: ConditionalEdgeIR, condition: Callable[..., Any]) -> Callable[..., Any]:
        def route(state: dict[str, Any]) -> Any:
            result = condition(state)
            if isinstance(result, (list, tuple)):
                return [r if isinstance(r, Send) else str(r) for r in result]
            return result if isinstance(result, Send) else str(result)

        route.__name__ = getattr(condition, "__name__", "route")
        return route

    @staticmethod
    def _make_send_router(fanout: DynamicFanOutIR, partition_fn: Callable[..., Any]) -> Callable[..., Any]:
        def send_router(state: dict[str, Any]) -> Any:
            tasks = partition_fn(state)
            if not tasks:
                return fanout.fan_in  # nothing to map over: go straight to the reducer step
            return [Send(fanout.worker_node, task) for task in tasks]

        return send_router

    @staticmethod
    def _create_loop_guard(loop: LoopIR, condition: Callable[..., Any]) -> Callable[..., Any]:
        counter = _loop_counter_key(loop)
        exit_dest = loop.exit_node or END

        def loop_guard(state: dict[str, Any]) -> Command:
            iterations = (state.get(counter) or 0) + 1
            if condition(state):
                if iterations < loop.max_iterations:
                    return Command(update={counter: iterations}, goto=loop.body_node)
                logger.warning(
                    "Loop '%s' stopped after max_iterations=%d with its condition still true.",
                    loop.id,
                    loop.max_iterations,
                )
            return Command(update={counter: 0}, goto=exit_dest)

        return loop_guard

    @staticmethod
    def _create_hitl_node(
        hitl_ir: HITLIR, schema: StateSchemaIR, validator: SchemaValidator | None
    ) -> Callable[..., Any]:
        """Creates an interrupt-based approval node that routes on the human decision."""

        def checked(decision: Any) -> tuple[bool, str, dict[str, Any]]:
            approved, feedback, edits = decision_fields(decision)
            unknown = [k for k in edits if k not in schema.fields]
            if unknown:
                raise StateValidationError(
                    f"Approval edits for '{hitl_ir.node_id}' contain unknown field(s): {', '.join(unknown)}"
                )
            if validator is not None and edits:
                edits = validator.validate(edits, f"approval edits for '{hitl_ir.node_id}'")
            return approved, feedback, edits

        def hitl_node(state: dict[str, Any]) -> Command:
            payload = {
                "prompt": hitl_ir.prompt,
                "description": hitl_ir.description,
                "node_name": hitl_ir.node_id,
                "state": {k: v for k, v in state.items() if not is_hidden(k)},
            }
            # Validation loop: invalid edits ask again with the error. (Raising here would
            # leave the bad answer recorded against the interrupt, failing every later resume.)
            while True:
                try:
                    approved, feedback, edits = checked(interrupt(payload))
                    break
                except StateValidationError as e:
                    payload = {**payload, "error": str(e)}
            if approved:
                return Command(update=edits, goto=hitl_ir.resume_target or END)
            if hitl_ir.reject_target:
                return Command(update=edits, goto=hitl_ir.reject_target)
            raise ExecutionError(f"Human review rejected at '{hitl_ir.node_id}': {feedback}")

        return hitl_node
