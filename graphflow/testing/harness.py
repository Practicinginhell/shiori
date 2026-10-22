"""Testing harness for Graphflow applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphflow.core.application import Application, CompiledApplication
from graphflow.events import NodeStarted, RunCompleted, RunInterrupted, StreamEvent


@dataclass
class TestRunResult:
    """Captured outputs, events, and final state from a test run."""
    __test__ = False  # not a pytest test class

    final_state: dict[str, Any]
    events: list[StreamEvent] = field(default_factory=list)
    interrupted: bool = False

    def get_executed_nodes(self) -> list[str]:
        return [e.node_name for e in self.events if isinstance(e, NodeStarted)]


class TestHarness:
    """Provides utilities to run and assert against workflows using mocks."""
    __test__ = False  # not a pytest test class

    @classmethod
    def test_app(
        cls,
        app: Application | CompiledApplication,
        input_data: dict[str, Any],
        thread_id: str | None = "test-thread",
    ) -> TestRunResult:
        """Runs the application once, capturing every stream event and the final state."""
        events: list[StreamEvent] = list(app.stream(input_data, thread_id=thread_id))
        final: dict[str, Any] = {}
        interrupted = False
        for ev in events:
            if isinstance(ev, RunCompleted):
                final = ev.final_output
            elif isinstance(ev, RunInterrupted):
                final, interrupted = ev.state, True
        return TestRunResult(final_state=final, events=events, interrupted=interrupted)
