"""Graphflow execution tracer.

Records per-run, per-node telemetry. Attach it to an Application to trace every
run automatically::

    tracer = ExecutionTracer()
    app = Application("app", listeners=[tracer])
    app.run({...})
    tracer.last_trace().steps
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from graphflow.events import (
    NodeCompleted,
    NodeStarted,
    RunCompleted,
    RunFailed,
    RunInterrupted,
    RunStarted,
    StreamEvent,
)


@dataclass
class StepTrace:
    """One node execution within a run."""
    node_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    updates: dict[str, Any] | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


@dataclass
class RunTrace:
    """A full workflow run."""
    run_id: str
    workflow_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    status: str = "running"  # running | completed | interrupted | failed
    steps: list[StepTrace] = field(default_factory=list)
    final_state: dict[str, Any] | None = None
    error: str | None = None

    @property
    def total_duration_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "total_duration_seconds": self.total_duration_seconds,
            "error": self.error,
            "steps": [
                {"node": s.node_name, "duration": s.duration_seconds, "error": s.error}
                for s in self.steps
            ],
        }


class ExecutionTracer:
    """Collects run and step traces, either via explicit calls or as an event listener."""

    def __init__(self) -> None:
        self.traces: dict[str, RunTrace] = {}
        self._open_steps: dict[tuple[str, str], StepTrace] = {}

    # Explicit API --------------------------------------------------------

    def start_run(self, run_id: str, workflow_name: str) -> RunTrace:
        trace = RunTrace(run_id=run_id, workflow_name=workflow_name)
        self.traces[run_id] = trace
        return trace

    def start_node(self, run_id: str, node_name: str) -> StepTrace:
        step = StepTrace(node_name=node_name)
        self._trace(run_id).steps.append(step)
        self._open_steps[(run_id, node_name)] = step
        return step

    def end_node(self, run_id: str, node_name: str, updates: dict[str, Any] | None = None) -> StepTrace:
        step = self._open_steps.pop((run_id, node_name), None) or self.start_node(run_id, node_name)
        self._open_steps.pop((run_id, node_name), None)
        step.end_time = time.perf_counter()
        step.updates = updates
        return step

    def record_error(self, run_id: str, node_name: str, error: str) -> StepTrace:
        step = self._open_steps.pop((run_id, node_name), None) or self.start_node(run_id, node_name)
        self._open_steps.pop((run_id, node_name), None)
        step.end_time = time.perf_counter()
        step.error = error
        return step

    def end_run(
        self,
        run_id: str,
        final_state: dict[str, Any] | None = None,
        error: str | None = None,
        status: str | None = None,
    ) -> RunTrace:
        trace = self._trace(run_id)
        trace.end_time = time.perf_counter()
        trace.final_state = final_state
        trace.error = error
        trace.status = status or ("failed" if error else "completed")
        return trace

    def get_trace(self, run_id: str) -> RunTrace | None:
        return self.traces.get(run_id)

    def last_trace(self) -> RunTrace | None:
        return next(reversed(self.traces.values()), None)

    def _trace(self, run_id: str) -> RunTrace:
        trace = self.traces.get(run_id)
        if trace is None:
            trace = self.start_run(run_id, workflow_name="")
        return trace

    # Event listener ------------------------------------------------------

    def on_event(self, event: StreamEvent) -> None:
        run_id = event.run_id
        if isinstance(event, RunStarted):
            self.start_run(run_id, event.workflow_name)
        elif isinstance(event, NodeStarted):
            self.start_node(run_id, event.node_name)
        elif isinstance(event, NodeCompleted):
            self.end_node(run_id, event.node_name, updates=event.output_updates)
        elif isinstance(event, RunCompleted):
            self.end_run(run_id, final_state=event.final_output)
        elif isinstance(event, RunInterrupted):
            self.end_run(run_id, final_state=event.state, status="interrupted")
        elif isinstance(event, RunFailed):
            if event.failed_node:
                self.record_error(run_id, event.failed_node, event.error)
            self.end_run(run_id, error=event.error)

    __call__ = on_event
