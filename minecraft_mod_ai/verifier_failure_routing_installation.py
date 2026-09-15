from __future__ import annotations

"""Install verifier routing invariants for the host-owned tool loop.

This keeps environment/toolchain failures out of source-repair RECOVER while
ensuring a genuine source-repair phase can actually mutate source.
"""

import json
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .verifier_failure_policy import FailureKind, classify_verifier_failure


def _payload_text(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(payload)


def install() -> None:
    from . import progress_aware_tool_loop as loop

    current_outcome = loop._verification_outcome
    if not getattr(current_outcome, "_mmm_typed_failure_routing", False):
        @wraps(current_outcome)
        def typed_verification_outcome(tool_name: str, payload: Mapping[str, Any]) -> str:
            outcome = current_outcome(tool_name, payload)
            if outcome != "FAIL":
                return outcome
            kind = classify_verifier_failure(_payload_text(payload))
            if kind is FailureKind.TOOLCHAIN_ENVIRONMENT:
                # A missing/unsupported Java/JDT target is verifier infrastructure,
                # never evidence that generated source is wrong.
                return "UNAVAILABLE"
            if kind is FailureKind.TRANSIENT_INFRASTRUCTURE:
                return "UNAVAILABLE"
            # SOURCE_DEFECT stays FAIL. UNKNOWN deliberately stays FAIL here so it
            # cannot be silently promoted to PASS; source mutation gating below is
            # what prevents an impossible repair surface.
            return outcome

        typed_verification_outcome._mmm_typed_failure_routing = True  # type: ignore[attr-defined]
        typed_verification_outcome.__wrapped__ = current_outcome  # type: ignore[attr-defined]
        loop._verification_outcome = typed_verification_outcome

    current_filter = loop._filter_tools_for_phase
    if not getattr(current_filter, "_mmm_recover_mutation_surface", False):
        @wraps(current_filter)
        def filter_tools_with_recover_mutation(
            exposed_tools: Sequence[Mapping[str, Any]],
            phase: Any,
            role: str,
            **kwargs: Any,
        ) -> tuple[Mapping[str, Any], ...]:
            selected = list(current_filter(exposed_tools, phase, role, **kwargs))
            if phase != loop.LoopPhase.RECOVER or role not in {"coder", "coder_safe"}:
                return tuple(selected)

            selected_names = {loop._tool_name(schema) for schema in selected}
            # RECOVER previously required a material source correction while
            # filtering every mutation tool out. Re-add only mutation tools that
            # were already reviewed and exposed to this request.
            for schema in exposed_tools:
                name = loop._tool_name(schema)
                if name in loop._MUTATION_ACT_TOOLS and name not in selected_names:
                    selected.append(schema)
                    selected_names.add(name)
            return tuple(selected)

        filter_tools_with_recover_mutation._mmm_recover_mutation_surface = True  # type: ignore[attr-defined]
        filter_tools_with_recover_mutation.__wrapped__ = current_filter  # type: ignore[attr-defined]
        loop._filter_tools_for_phase = filter_tools_with_recover_mutation


__all__ = ["install"]
