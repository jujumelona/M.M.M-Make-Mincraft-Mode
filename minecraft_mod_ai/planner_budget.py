from __future__ import annotations

"""Shared global planner call budget and stage execution tracking."""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class PlannerBudget:
    """Track and enforce global model invocation limits across planning stages."""

    max_calls: int | None = None
    call_count: int = 0
    stage_counts: Counter[str] = field(default_factory=Counter)

    def consume(self, stage: str, count: int = 1) -> None:
        """Record model calls for a stage and enforce the global ceiling."""
        if count < 0:
            raise ValueError(f"PLANNER_BUDGET: count must be non-negative, got {count}")
        if self.max_calls is not None and self.call_count + count > self.max_calls:
            raise ValueError(
                f"PLANNER_BUDGET_EXCEEDED: stage={stage!r} "
                f"requested={count} current_total={self.call_count} max_calls={self.max_calls}"
            )
        self.call_count += count
        self.stage_counts[stage] += count

    def remaining(self) -> int | None:
        """Return remaining calls when an explicit structural ceiling is configured."""
        if self.max_calls is None:
            return None
        return max(0, self.max_calls - self.call_count)

    def summary(self) -> dict[str, int | None]:
        """Return total calls and per-stage breakdown."""
        return {
            "total_calls": self.call_count,
            "max_calls": self.max_calls,
            "remaining": self.remaining(),
            **dict(self.stage_counts),
        }
