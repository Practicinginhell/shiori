"""Graphflow Workflow builder.

Provides a fluent, declarative API to define workflows (sequential, branching,
parallel, loops, human-in-the-loop, dynamic fan-out, sub-workflows) and lower
them into Graphflow GraphIR without leaking LangGraph boilerplate.

The builder tracks a *frontier*: the open ends that the next ``.then()`` will
attach to. After ``branch()`` the frontier is every route target that has no
outgoing edge yet; after ``parallel()`` without a fan-in, the next ``.then()``
node becomes the join that waits for all branches; after ``loop()`` without an
exit node, the next ``.then()`` node becomes the loop exit.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Union

from graphflow.core.agent import Agent
from graphflow.core.state import State, schema_from_any
from graphflow.errors import ConfigurationError
from graphflow.ir.graph import (
    END,
    HITLIR,
    RESERVED_PREFIX,
    ConditionalEdgeIR,
    DynamicFanOutIR,
    EdgeIR,
    ErrorHandler,
    GraphIR,
    LoopIR,
    LoweredWorkflow,
    NodeIR,
    ParallelBranchIR,
)
from graphflow.ir.lower import lower_schema, ref_for
from graphflow.ir.registry import Registry
from graphflow.ir.state import StateSchemaIR
from graphflow.retry import RetryPolicy, check_retry_args, lower_retry

NodeLike = Union[Callable[..., Any], Agent, "Workflow", str]


@dataclass
class _NodeSpec:
    """A node as declared on the builder (holds the live function / Agent / Workflow)."""

    name: str
    func: Any
    retries: int = 0
    retry: RetryPolicy | None = None
    timeout: float | None = None
    on_error: ErrorHandler | None = None
    isolate_errors: bool = False
    raw: bool = False


class Workflow:
    """Fluent builder for workflows and graphs."""

    def __init__(
        self,
        name: str = "workflow",
        state_schema: type[State] | dict[str, Any] | StateSchemaIR | type | None = None,
    ):
        self.name = name
        self.state_schema_ir = schema_from_any(state_schema)

        self._nodes: dict[str, _NodeSpec] = {}
        self._edges: list[EdgeIR] = []
        self._conditional_edges: list[ConditionalEdgeIR] = []
        self._parallel_branches: list[ParallelBranchIR] = []
        self._dynamic_fan_outs: list[DynamicFanOutIR] = []
        self._loops: list[LoopIR] = []
        self._hitl_dict: dict[str, HITLIR] = {}
        self._error_handler: ErrorHandler | None = None

        self._entry_point: str | None = None
        # Open ends that the next .then() attaches to (see module docstring).
        self._frontier: list[str] = []
        self._frontier_from_branch = False
        self._pending_join: ParallelBranchIR | None = None
        self._pending_loop_exit: LoopIR | None = None

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def _get_node_name(self, node: Any, explicit_name: str | None = None) -> str:
        if explicit_name:
            return explicit_name
        if isinstance(node, str):
            return node
        if isinstance(node, (Agent, Workflow)):
            return node.name
        if hasattr(node, "__name__") and node.__name__ != "<lambda>":
            return node.__name__
        return f"node_{len(self._nodes) + 1}"

    def add_node(
        self,
        name_or_func: NodeLike,
        func: Callable[..., Any] | Agent | Workflow | None = None,
        *,
        retries: int = 0,
        timeout: float | None = None,
        on_error: ErrorHandler | None = None,
        retry: RetryPolicy | None = None,
    ) -> Workflow:
        """Add a standalone node without wiring edges.

        Args:
            retries: Extra attempts after any failure (short exponential backoff).
            timeout: Seconds per attempt before the node fails with TimeoutError.
            on_error: ``(state, error) -> update`` called after the last attempt fails;
                its return value becomes the node's update and execution continues.
            retry: A RetryPolicy (backoff, jitter, which errors to retry); by default
                only transient errors are retried. Use instead of `retries`.
        """
        if isinstance(name_or_func, str):
            if func is None:
                raise ConfigurationError(
                    f"add_node('{name_or_func}') requires a function"
                )
            node_name, target = name_or_func, func
        else:
            node_name, target = self._get_node_name(name_or_func), name_or_func

        if node_name.startswith(RESERVED_PREFIX) or node_name == END:
            raise ConfigurationError(
                f"Node name '{node_name}' is reserved (names may not start with '__')"
            )
        if node_name in self._nodes or node_name in self._hitl_dict:
            raise ConfigurationError(
                f"Node '{node_name}' is already registered in workflow '{self.name}'"
            )

        policy = check_retry_args(retries, retry, f"Node '{node_name}'")
        self._nodes[node_name] = _NodeSpec(
            name=node_name,
            func=target,
            retries=retries,
            retry=policy,
            timeout=timeout,
            on_error=on_error,
        )
        return self

    def _ensure_node(
        self, node: NodeLike, name: str | None = None, **node_kwargs: Any
    ) -> str:
        """Registers `node` unless it is a name or already-registered object; returns its id."""
        node_name = self._get_node_name(node, name)
        if isinstance(node, str):
            return node_name  # reference, validated at compile time
        existing = self._nodes.get(node_name)
        if existing is not None:
            if existing.func is not node:
                raise ConfigurationError(
                    f"Node '{node_name}' is already registered in workflow '{self.name}' "
                    "with a different function; pass name=... to disambiguate."
                )
            return node_name
        self.add_node(node_name, node, **node_kwargs)
        return node_name

    def add_raw_node(self, name: str, func: Callable[..., Any]) -> Workflow:
        """Escape Hatch: Add a raw LangGraph node function that directly expects (state, config)."""
        self.add_node(name, func)
        self._nodes[name].raw = True
        return self

    def add_raw_edge(self, source: str, target: str) -> Workflow:
        """Escape Hatch: Add a direct edge between two nodes."""
        self._edges.append(EdgeIR(source=source, target=target))
        return self

    # ------------------------------------------------------------------ #
    # Frontier management
    # ------------------------------------------------------------------ #

    def _has_outgoing(self, node_id: str) -> bool:
        return (
            any(e.source == node_id for e in self._edges)
            or any(c.source == node_id for c in self._conditional_edges)
            or any(
                p.source == node_id or (p.fan_in and node_id in p.branch_nodes)
                for p in self._parallel_branches
            )
            or any(
                d.source == node_id or d.worker_node == node_id
                for d in self._dynamic_fan_outs
            )
            or any(loop.body_node == node_id for loop in self._loops)
            or node_id in self._hitl_dict
        )

    def _connect_frontier_to(self, target: str) -> None:
        if self._entry_point is None:
            self._entry_point = target
        if self._pending_join is not None:
            self._pending_join.fan_in = target
        if self._pending_loop_exit is not None:
            self._pending_loop_exit.exit_node = target
        frontier = self._frontier
        if self._frontier_from_branch:
            frontier = [n for n in frontier if not self._has_outgoing(n)]
        if self._pending_join is None:
            for source in frontier:
                if source != target:
                    self._edges.append(EdgeIR(source=source, target=target))
        self._set_frontier([target])

    def _set_frontier(
        self,
        nodes: list[str],
        *,
        from_branch: bool = False,
        pending_join: ParallelBranchIR | None = None,
        pending_loop_exit: LoopIR | None = None,
    ) -> None:
        self._frontier = nodes
        self._frontier_from_branch = from_branch
        self._pending_join = pending_join
        self._pending_loop_exit = pending_loop_exit

    def _single_source(self, from_node: str | None, operation: str) -> str:
        if from_node:
            return from_node
        if len(self._frontier) == 1 and self._pending_join is None:
            return self._frontier[0]
        if not self._frontier:
            raise ConfigurationError(
                f"Cannot {operation} without a preceding node or explicit from_node"
            )
        raise ConfigurationError(
            f"Cannot {operation} from multiple open branches {self._frontier}; pass from_node=..."
        )

    @property
    def _last_node(self) -> str | None:
        """The single current tail node, if there is exactly one (backward compatibility)."""
        return self._frontier[0] if len(self._frontier) == 1 else None

    # ------------------------------------------------------------------ #
    # Fluent composition API
    # ------------------------------------------------------------------ #

    def then(
        self,
        node: NodeLike,
        name: str | None = None,
        retries: int = 0,
        timeout: float | None = None,
        on_error: ErrorHandler | None = None,
        retry: RetryPolicy | None = None,
    ) -> Workflow:
        """Add a node (function, Agent, sub-Workflow, or existing node name) after the current tail.

        See ``add_node`` for `retries`, `timeout`, `on_error` and `retry`.
        """
        node_name = self._ensure_node(
            node, name, retries=retries, timeout=timeout, on_error=on_error, retry=retry
        )
        self._connect_frontier_to(node_name)
        return self

    def branch(
        self,
        condition: Callable[..., Any],
        routes: dict[Any, NodeLike | None],
        from_node: str | None = None,
    ) -> Workflow:
        """Branch conditionally: ``condition(state)`` returns a key of `routes`.

        A route value of ``None`` or ``graphflow.END`` terminates that path. A
        following ``.then(x)`` joins every route target that has no outgoing edge.
        """
        source = self._single_source(from_node, "branch")

        route_map: dict[str, str] = {}
        targets: list[str] = []
        for outcome_key, target in routes.items():
            if target is None or target == END:
                route_map[str(outcome_key)] = END
                continue
            target_name = self._ensure_node(target)
            route_map[str(outcome_key)] = target_name
            if target_name not in targets:
                targets.append(target_name)

        self._conditional_edges.append(
            ConditionalEdgeIR(source=source, condition=condition, route_map=route_map)
        )
        self._set_frontier(targets, from_branch=True)
        return self

    def parallel(
        self,
        branches: Sequence[NodeLike],
        fan_in: NodeLike | None = None,
        from_node: str | None = None,
        isolate_errors: bool = True,
    ) -> Workflow:
        """Run branch nodes concurrently; `fan_in` runs once after *all* of them finish.

        With ``isolate_errors`` (default), a failing branch records
        ``"<node>: <error>"`` in the state's ``errors`` field (which must be
        declared, e.g. ``errors: list = Field.reducer("append")``) and the other
        branches still complete. Without an ``errors`` field the failure propagates.
        """
        source = self._single_source(from_node, "run parallel branches")

        branch_ids: list[str] = []
        for b in branches:
            b_name = self._ensure_node(b)
            if isolate_errors and b_name in self._nodes:
                self._nodes[b_name].isolate_errors = True
            branch_ids.append(b_name)

        fan_in_id = self._ensure_node(fan_in) if fan_in is not None else ""

        par = ParallelBranchIR(
            id=f"parallel_{len(self._parallel_branches) + 1}",
            source=source,
            branch_nodes=branch_ids,
            fan_in=fan_in_id,
        )
        self._parallel_branches.append(par)
        if fan_in_id:
            self._set_frontier([fan_in_id])
        else:
            self._set_frontier(branch_ids, pending_join=par)
        return self

    def fan_out(
        self,
        partition_fn: Callable[[dict[str, Any]], list[dict[str, Any]]],
        worker_node: NodeLike,
        fan_in: NodeLike,
        from_node: str | None = None,
    ) -> Workflow:
        """Dynamic map-reduce / fan-out (LangGraph Send API translation).

        Args:
            partition_fn: Function taking state and returning a list of task state dicts.
            worker_node: Node executed concurrently for each task (receives the task dict).
            fan_in: Aggregator node executed after all workers complete.
        """
        source = self._single_source(from_node, "fan out")
        worker_name = self._ensure_node(worker_node)
        fan_in_name = self._ensure_node(fan_in)

        self._dynamic_fan_outs.append(
            DynamicFanOutIR(
                id=f"dynamic_fan_out_{len(self._dynamic_fan_outs) + 1}",
                source=source,
                partition_fn=partition_fn,
                worker_node=worker_name,
                fan_in=fan_in_name,
            )
        )
        self._set_frontier([fan_in_name])
        return self

    def loop(
        self,
        body_node: NodeLike,
        while_condition: Callable[[dict[str, Any]], bool],
        exit_node: NodeLike | None = None,
        max_iterations: int = 25,
    ) -> Workflow:
        """Run `body_node`, then repeat it while `while_condition(state)` is True.

        The body runs at most `max_iterations` times. Without `exit_node`, the
        next ``.then()`` node becomes the exit (or the workflow ends).
        """
        body_name = self._ensure_node(body_node)
        exit_name = self._ensure_node(exit_node) if exit_node is not None else ""

        self._connect_frontier_to(body_name)
        loop = LoopIR(
            id=f"loop_{len(self._loops) + 1}",
            body_node=body_name,
            condition=while_condition,
            exit_node=exit_name,
            max_iterations=max_iterations,
        )
        self._loops.append(loop)

        if exit_name:
            self._set_frontier([exit_name])
        else:
            self._set_frontier([], pending_loop_exit=loop)
        return self

    def require_approval(
        self,
        before: NodeLike,
        prompt: str = "Human approval required",
        description: str = "",
        on_reject: NodeLike | None = None,
    ) -> Workflow:
        """Pause for human approval before `before` runs.

        Resume with ``app.resume(thread_id, ApprovalDecision(...))``. Approved ->
        `before` runs (with any ``edits`` applied to state). Rejected -> `on_reject`
        runs if given, otherwise the run fails with ExecutionError.
        """
        target_name = self._ensure_node(before)
        reject_name = self._ensure_node(on_reject) if on_reject is not None else None
        approval_node_id = f"approval_before_{target_name}"
        if approval_node_id in self._hitl_dict:
            raise ConfigurationError(f"'{target_name}' already requires approval")

        self._hitl_dict[approval_node_id] = HITLIR(
            node_id=approval_node_id,
            prompt=prompt,
            description=description,
            resume_target=target_name,
            reject_target=reject_name,
        )
        self._connect_frontier_to(approval_node_id)
        self._set_frontier([target_name])
        return self

    def on_error(self, handler: ErrorHandler) -> Workflow:
        """Registers a workflow-wide fallback for node failures.

        ``handler(state, error)`` is called after a node exhausts its retries
        (unless the node has its own ``on_error``). Its return value is used as
        the failed node's state update and execution continues. Re-raise inside
        the handler to fail the run instead.
        """
        self._error_handler = handler
        return self

    def merge(self, other: Workflow, name: str | None = None) -> Workflow:
        """Returns a new workflow running `self` and then `other`.

        Node names must not collide. The merged workflow uses `self`'s state schema.
        """
        clash = (set(self._nodes) | set(self._hitl_dict)) & (
            set(other._nodes) | set(other._hitl_dict)
        )
        if clash:
            raise ConfigurationError(
                f"Cannot merge workflows: duplicate node(s) {sorted(clash)}"
            )
        if other._entry_point is None:
            raise ConfigurationError(
                f"Cannot merge workflow '{other.name}': it has no entry point"
            )

        merged = self.copy(name or f"{self.name}+{other.name}")
        o = other.copy()
        merged._nodes.update(o._nodes)
        merged._edges.extend(o._edges)
        merged._conditional_edges.extend(o._conditional_edges)
        merged._parallel_branches.extend(o._parallel_branches)
        merged._dynamic_fan_outs.extend(o._dynamic_fan_outs)
        merged._loops.extend(o._loops)
        merged._hitl_dict.update(o._hitl_dict)
        if merged._error_handler is None:
            merged._error_handler = o._error_handler

        merged._connect_frontier_to(o._entry_point)
        # Adopt the tail of `other`, remapping pending IR objects to the copies now in `merged`.
        merged._frontier = list(o._frontier)
        merged._frontier_from_branch = o._frontier_from_branch
        merged._pending_join = o._pending_join
        merged._pending_loop_exit = o._pending_loop_exit
        return merged

    def copy(self, name: str | None = None) -> Workflow:
        """Returns an independent copy of this builder (node functions are shared)."""
        new = Workflow.__new__(Workflow)
        memo: dict[int, Any] = {}
        # Share callables/agents; copy only the IR containers.
        for node in self._nodes.values():
            memo[id(node.func)] = node.func
            memo[id(node.on_error)] = node.on_error
        for c in self._conditional_edges:
            memo[id(c.condition)] = c.condition
        for loop in self._loops:
            memo[id(loop.condition)] = loop.condition
        for d in self._dynamic_fan_outs:
            memo[id(d.partition_fn)] = d.partition_fn
        memo[id(self.state_schema_ir)] = self.state_schema_ir
        memo[id(self._error_handler)] = self._error_handler
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        if name:
            new.name = name
        return new

    # ------------------------------------------------------------------ #
    # Lowering
    # ------------------------------------------------------------------ #

    def lower(self, registry: Registry) -> GraphIR:
        """Lowers this workflow into plain-data GraphIR, registering live objects in `registry`."""
        nodes: dict[str, NodeIR] = {}
        for node_id, spec in self._nodes.items():
            common: dict[str, Any] = {
                "id": node_id,
                "name": node_id,
                "retries": spec.retries,
                "retry": lower_retry(spec.retry, registry, node_id),
                "timeout_seconds": spec.timeout,
                "on_error": ref_for(
                    spec.on_error, registry, "handler", f"{node_id}.on_error"
                ),
                "isolate_errors": spec.isolate_errors,
            }
            if isinstance(spec.func, Workflow):
                nodes[node_id] = NodeIR(
                    kind="subgraph", subgraph=spec.func.lower(registry), **common
                )
            elif isinstance(spec.func, Agent):
                nodes[node_id] = NodeIR(
                    kind="agent", agent=spec.func.lower(registry), **common
                )
            else:
                kind = "raw" if spec.raw else "function"
                nodes[node_id] = NodeIR(
                    kind=kind,
                    target=registry.register(spec.func, "fn", node_id),
                    **common,
                )

        return GraphIR(
            name=self.name,
            nodes=nodes,
            edges=[replace(e) for e in self._edges],
            conditional_edges=[
                replace(
                    c,
                    condition=ref_for(
                        c.condition, registry, "cond", f"{c.source}.route"
                    ),
                    route_map=dict(c.route_map),
                )
                for c in self._conditional_edges
            ],
            parallel_branches=[
                replace(p, branch_nodes=list(p.branch_nodes))
                for p in self._parallel_branches
            ],
            dynamic_fan_outs=[
                replace(
                    d, partition_fn=ref_for(d.partition_fn, registry, "partition", d.id)
                )
                for d in self._dynamic_fan_outs
            ],
            loops=[
                replace(
                    loop, condition=ref_for(loop.condition, registry, "cond", loop.id)
                )
                for loop in self._loops
            ],
            hitl_nodes={k: replace(v) for k, v in self._hitl_dict.items()},
            entry_point=self._entry_point,
            state_schema=lower_schema(self.state_schema_ir, registry),
            error_handler=ref_for(
                self._error_handler, registry, "handler", f"{self.name}.on_error"
            ),
        )

    def to_program(self, registry: Registry | None = None) -> LoweredWorkflow:
        """Lowers this workflow: plain-data IR plus the registry of objects it references."""
        registry = registry if registry is not None else Registry()
        return LoweredWorkflow(ir=self.lower(registry), registry=registry)

    def to_ir(self) -> GraphIR:
        """The plain-data GraphIR of this workflow (JSON-serializable via ``to_dict()``)."""
        return self.to_program().ir

    def to_langgraph(self) -> Any:
        """Escape Hatch: Compiles this workflow directly to a LangGraph StateGraph instance."""
        from graphflow.graph.compiler import LangGraphCompiler

        program = self.to_program()
        return LangGraphCompiler().compile_to_state_graph(program.ir, program.registry)

    def __repr__(self) -> str:
        return f"Workflow(name={self.name!r}, nodes={list(self._nodes)})"
