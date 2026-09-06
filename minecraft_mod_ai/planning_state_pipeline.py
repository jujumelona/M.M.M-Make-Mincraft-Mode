from __future__ import annotations

"""Single prompt-first planning state machine.

No requirement catalog, implementation plan, or retrieval query exists before the
preceding state is available. This is the production ordering contract used by the
planner entry point.
"""

from collections.abc import Mapping
from typing import Any

from .planning_state_contract import build_initial_planning_state, validate_planning_state
from .planning_state_implementation import compile_detailed_implementation_plans
from .planning_state_research import collect_planning_state_research
from .planning_state_resolution import compile_researched_requirements


def prepare_planning_state(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve prompt meaning, reference scope, implementation evidence, and plan detail."""

    state = build_initial_planning_state(router, prompt)

    # Pass 1 resolves reference semantics/external facts/scope/repository facts that are
    # necessary before the system is allowed to decide what the mod must contain.
    state = collect_planning_state_research(
        router,
        prompt,
        state,
        trace_metadata=trace_metadata,
    )

    # The second bounded template converts only authored + grounded resolved knowledge
    # into player-visible requirements and automatically opens implementation research.
    state = compile_researched_requirements(router, prompt, state)

    # Pass 2 searches actual reusable implementations, source/API behavior and support
    # artifacts for each requirement. Prompt vocabulary is no longer the sole query source.
    state = collect_planning_state_research(
        router,
        prompt,
        state,
        trace_metadata=trace_metadata,
    )

    # The final template is code-facing but evidence-bound. Semantic-only tasks are not
    # considered ready and therefore never reach the coder.
    state = compile_detailed_implementation_plans(router, prompt, state)
    validate_planning_state(state, prompt=prompt)
    if state.get("plan_ready") is not True:
        raise ValueError("PLANNING_STATE_NOT_READY: planning state did not reach code-ready coverage")
    return state


__all__ = ["prepare_planning_state"]
