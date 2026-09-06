"""Planner hand-off contracts.

These types are deliberately small and serializable. Planner stages should
exchange these contracts instead of loosely shaped dictionaries so producers
and consumers agree on field names and optionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PlannerInput:
    request: str
    context: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PlannerClaim:
    claim: str
    evidence_ids: tuple[str, ...] = ()
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    action: str
    rationale: str
    claims: tuple[PlannerClaim, ...] = ()
    metadata: Mapping[str, Any] | None = None


__all__ = ["PlannerClaim", "PlannerDecision", "PlannerInput"]
