"""Structural validation of a GraphIR before compilation.

Catches dangling node references, unresolvable registry references and
malformed graphs with a clear message instead of letting them surface as
obscure LangGraph errors (or silently wrong graphs).
"""

from __future__ import annotations

from graphflow.errors import CompilationError
from graphflow.ir.graph import END, GraphIR
from graphflow.ir.registry import Registry, is_ref
from graphflow.ir.state import VALIDATION_MODES


def validate_graph(ir: GraphIR, registry: Registry | None = None) -> None:
    """Raises CompilationError listing every structural problem found in `ir`.

    With a `registry`, also checks that every reference in the IR resolves.
    """
    problems: list[str] = []
    _collect(ir, registry, problems, prefix="")
    if problems:
        details = "\n  - ".join(problems)
        raise CompilationError(f"Workflow '{ir.name}' is invalid:\n  - {details}")


def _collect(ir: GraphIR, registry: Registry | None, problems: list[str], prefix: str) -> None:
    known = ir.all_node_ids()

    def check(ref: str | None, context: str) -> None:
        if ref and ref != END and ref not in known:
            problems.append(f"{prefix}{context} references unknown node '{ref}'")

    def check_ref(ref: object, context: str) -> None:
        if registry is None or ref is None or callable(ref):
            return
        if not is_ref(ref) or ref not in registry:
            problems.append(f"{prefix}{context} has unresolved reference {ref!r}")

    if not ir.nodes and not ir.hitl_nodes:
        problems.append(f"{prefix}it contains no nodes to execute")

    check(ir.entry_point, "Entry point")
    check_ref(ir.error_handler, "Workflow error handler")

    for e in ir.edges:
        check(e.source, f"Edge {e.source} -> {e.target}")
        check(e.target, f"Edge {e.source} -> {e.target}")

    for c in ir.conditional_edges:
        check(c.source, f"Conditional edge from '{c.source}'")
        check_ref(c.condition, f"Conditional edge from '{c.source}'")
        for outcome, target in c.route_map.items():
            check(target, f"Conditional edge from '{c.source}' (route '{outcome}')")

    for p in ir.parallel_branches:
        check(p.source, f"Parallel block '{p.id}' source")
        for b in p.branch_nodes:
            check(b, f"Parallel block '{p.id}' branch")
        check(p.fan_in, f"Parallel block '{p.id}' fan_in")

    for d in ir.dynamic_fan_outs:
        check(d.source, f"Fan-out '{d.id}' source")
        check(d.worker_node, f"Fan-out '{d.id}' worker")
        check(d.fan_in, f"Fan-out '{d.id}' fan_in")
        check_ref(d.partition_fn, f"Fan-out '{d.id}' partition function")

    for loop in ir.loops:
        check(loop.body_node, f"Loop '{loop.id}' body")
        check(loop.exit_node, f"Loop '{loop.id}' exit")
        check_ref(loop.condition, f"Loop '{loop.id}' condition")
        if loop.max_iterations < 1:
            problems.append(f"{prefix}Loop '{loop.id}' max_iterations must be >= 1")

    for h in ir.hitl_nodes.values():
        check(h.resume_target, f"Approval step '{h.node_id}'")
        check(h.reject_target, f"Approval step '{h.node_id}' reject route")

    for n in ir.nodes.values():
        where = f"Node '{n.id}'"
        if n.kind in ("function", "raw"):
            if n.target is None:
                problems.append(f"{prefix}{where} has no function")
            check_ref(n.target, where)
        elif n.kind == "agent":
            if n.agent is None:
                problems.append(f"{prefix}{where} has no agent definition")
            else:
                check_ref(n.agent.model, f"{where} model")
                if registry is not None:
                    problems.extend(
                        f"{prefix}Agent '{n.agent.name}' references unknown tool '{t.name}'"
                        for t in n.agent.tools
                        if t.ref not in registry
                    )
                for ref in n.agent.middleware:
                    check_ref(ref, f"{where} middleware")
                check_ref(n.agent.structured_output, f"{where} structured_output")
                check_ref(n.agent.approval_handler, f"{where} approval_handler")
        elif n.kind == "subgraph":
            if n.subgraph is None:
                problems.append(f"{prefix}{where} has no sub-workflow")
            else:
                _collect(n.subgraph, registry, problems, prefix=f"{prefix}[{n.id}] ")
        check_ref(n.on_error, f"{where} on_error")
        if n.retries < 0:
            problems.append(f"{prefix}{where} retries must be >= 0")
        if n.retry is not None and n.retry.max_attempts < 1:
            problems.append(f"{prefix}{where} retry.max_attempts must be >= 1")
        if n.timeout_seconds is not None and n.timeout_seconds <= 0:
            problems.append(f"{prefix}{where} timeout must be > 0")

    schema = ir.state_schema
    if schema.validation not in VALIDATION_MODES:
        problems.append(f"{prefix}State validation mode must be one of {VALIDATION_MODES}, got {schema.validation!r}")
    for f in schema.fields.values():
        check_ref(f.type_ref, f"Field '{f.name}' type")
        check_ref(f.default_ref, f"Field '{f.name}' default")
        if isinstance(f.default_factory, str):
            check_ref(f.default_factory, f"Field '{f.name}' default_factory")
        if is_ref(f.reducer):
            check_ref(f.reducer, f"Field '{f.name}' reducer")
