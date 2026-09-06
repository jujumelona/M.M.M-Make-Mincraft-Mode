from __future__ import annotations

"""Single prompt-first planning state machine.

No requirement catalog, implementation plan, or retrieval query exists before the
preceding state is available. This is the production ordering contract used by the
planner entry point.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
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
    existing_state: Mapping[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Resolve prompt meaning, reference scope, implementation evidence, and plan detail."""

    state = deepcopy(dict(existing_state)) if existing_state is not None else build_initial_planning_state(router, prompt)
    validate_planning_state(state, prompt=prompt)
    if state.get("plan_ready") is True:
        return state
    if checkpoint is not None:
        checkpoint(deepcopy(state))
    requirements_exist = any(item.get("decision_type") == "requirement" for item in state["decisions"])

    if not requirements_exist:
        state = collect_planning_state_research(router, prompt, state, trace_metadata=trace_metadata)
        if checkpoint is not None:
            checkpoint(deepcopy(state))
        state = compile_researched_requirements(router, prompt, state)
        if checkpoint is not None:
            checkpoint(deepcopy(state))

    # Pass 2 searches actual reusable implementations, source/API behavior and support
    # artifacts for each requirement. Prompt vocabulary is no longer the sole query source.
    state = collect_planning_state_research(
        router,
        prompt,
        state,
        trace_metadata=trace_metadata,
    )

    if checkpoint is not None:
        checkpoint(deepcopy(state))

    # The final template is code-facing but evidence-bound. Semantic-only tasks are not
    # considered ready and therefore never reach the coder.
    state = compile_detailed_implementation_plans(router, prompt, state)
    validate_planning_state(state, prompt=prompt)
    if state.get("plan_ready") is not True:
        raise ValueError("PLANNING_STATE_NOT_READY: planning state did not reach code-ready coverage")
    if checkpoint is not None:
        checkpoint(deepcopy(state))
    return state


__all__ = ["prepare_planning_state"]
