"""Canonical contract types used across the MMM pipeline.

Small-model agents should import contracts from this package instead of
redeclaring ad-hoc dict shapes in stage modules.
"""

from .execution import ExecutionReceipt, ExecutionStatus
from .planning import PlannerClaim, PlannerDecision, PlannerInput
from .target import TargetContract

__all__ = [
    "ExecutionReceipt",
    "ExecutionStatus",
    "PlannerClaim",
    "PlannerDecision",
    "PlannerInput",
    "TargetContract",
]
