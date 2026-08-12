from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def install(agentic_module: Any) -> None:
    """Keep adaptive planning search while gating expensive repair breadth.

    Auto planning delegates to the core risk-aware policy so independent candidates
    can use the native llama-server slots when the request actually warrants search.
    Explicit ``on`` and ``off`` remain hard overrides. Repair search stays failure-
    gated: it widens only after the same verifier signature survives a prior repair.
    """

    current_plan_count = agentic_module._planner_candidate_count
    if not getattr(current_plan_count, "_mmm_failure_gated_search", False):

        def planner_candidate_count(request: Any, stage: str) -> int:
            mode = agentic_module._mode()
            if mode == "on":
                return agentic_module._env_int(
                    "MMM_PLAN_SEARCH_WIDTH",
                    2,
                    maximum=3,
                )
            if mode == "off":
                return 1
            return max(1, min(3, int(current_plan_count(request, stage))))

        planner_candidate_count._mmm_failure_gated_search = True  # type: ignore[attr-defined]
        planner_candidate_count.__wrapped__ = current_plan_count  # type: ignore[attr-defined]
        agentic_module._planner_candidate_count = planner_candidate_count

    current_repair_count = agentic_module._repair_candidate_count
    if getattr(current_repair_count, "_mmm_failure_gated_search", False):
        return

    def repair_candidate_count(
        self: Any,
        evidence: Mapping[str, Any],
        memory: Sequence[Mapping[str, Any]],
    ) -> int:
        mode = agentic_module._mode()
        if mode == "off":
            return 1
        width = agentic_module._env_int(
            "MMM_REPAIR_SEARCH_WIDTH",
            2,
            maximum=3,
        )
        if mode == "on":
            return width

        # A strong verified memory hit is already the cheapest high-confidence route.
        if memory and float(memory[0].get("similarity", 0.0)) >= 0.72:
            return 1

        signature = self._signature(dict(evidence))
        counts = getattr(self, "_mmm_signature_counts", None)
        if not isinstance(counts, Counter):
            counts = Counter()
            self._mmm_signature_counts = counts
        counts[signature] += 1

        # First encounter gets one focused repair. Only if that same verifier state
        # survives and comes back do we pay for independent alternative candidates.
        return min(3, width) if counts[signature] >= 2 else 1

    repair_candidate_count._mmm_failure_gated_search = True  # type: ignore[attr-defined]
    repair_candidate_count.__wrapped__ = current_repair_count  # type: ignore[attr-defined]
    agentic_module._repair_candidate_count = repair_candidate_count


__all__ = ["install"]
