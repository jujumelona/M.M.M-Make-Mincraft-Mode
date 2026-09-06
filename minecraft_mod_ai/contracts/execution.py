"""Execution-stage contracts.

Keep execution hand-off shapes explicit so stage code and small-model agents
share one import path instead of inventing structurally similar dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ExecutionStatus(str, Enum):
    """Terminal or intermediate execution result state."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Canonical receipt emitted by an execution step."""

    status: ExecutionStatus
    stage: str
    message: str = ""
    artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["ExecutionReceipt", "ExecutionStatus"]
