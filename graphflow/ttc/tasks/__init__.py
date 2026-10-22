"""Tasks with agent-accessible verifiers."""

from graphflow.ttc.tasks.base import CallableTask, Evaluation, Task
from graphflow.ttc.tasks.polyomino import PolyominoPackingTask

__all__ = ["CallableTask", "Evaluation", "PolyominoPackingTask", "Task"]
