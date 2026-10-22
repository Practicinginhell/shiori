"""Structured logger for Graphflow."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from typing import Any

from graphflow.events import ModelToken, StreamEvent


class StructuredLogger:
    """Structured JSON & human-readable logger for agent runs.

    Can be attached as an Application listener to log every event
    (``ModelToken`` events are skipped unless ``log_tokens=True``).
    """

    def __init__(
        self,
        name: str = "graphflow",
        json_format: bool = False,
        level: int = logging.INFO,
        log_tokens: bool = False,
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.json_format = json_format
        self.log_tokens = log_tokens

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
            self.logger.addHandler(handler)

    def log(self, level: int, event: str, **kwargs: Any) -> None:
        if self.json_format:
            self.logger.log(level, json.dumps({"event": event, **kwargs}, default=str))
        else:
            details = " ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.log(level, f"{event} {details}".strip())

    def info(self, event: str, **kwargs: Any) -> None:
        self.log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self.log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self.log(logging.ERROR, event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self.log(logging.DEBUG, event, **kwargs)

    def on_event(self, event: StreamEvent) -> None:
        if isinstance(event, ModelToken) and not self.log_tokens:
            return
        fields = {
            k: v for k, v in dataclasses.asdict(event).items()
            if k not in ("timestamp", "full_state", "input_state", "initial_input", "final_output", "state")
        }
        level = logging.ERROR if type(event).__name__ == "RunFailed" else logging.INFO
        self.log(level, type(event).__name__, **fields)

    __call__ = on_event
