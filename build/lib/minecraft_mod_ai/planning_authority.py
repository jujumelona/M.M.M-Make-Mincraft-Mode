from __future__ import annotations

"""Authoritative request catalog boundary for prompt-first grounded planning.

The old deterministic clause/capability/query compiler was removed because it converted
raw prompt wording into implementation/search decisions before research.  Authority now
comes from the plan-ready planning-state SSOT and its grounded detailed plan.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

from .planning_state_contract import validate_planning_state
from .planning_state_handoff import build_request_catalog_from_planning_state
from .planning_state_pipeline import prepare_planning_state
from .root_cause_trace import emit_root_cause, trace_scope

_ACTIVE_REQUEST_CATALOG: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar(
    "mmm_active_authoritative_request_catalog",
    default=None,
)
_ACTIVE_PLANNING_STATE: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar(
    "mmm_active_planning_state",
    default=None,
)


def build_authoritative_request_catalog(
    prompt: str,
    router: Any | None = None,
    *,
    planning_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a researched, provenance-checked request catalog.

    A caller may supply the already prepared state to avoid repeating research.  Without
    one, a real model/router is mandatory and the complete prompt-first state machine is
    run.  There is deliberately no deterministic raw-prompt fallback.
    """

    authored = str(prompt or "")
    if not authored.strip():
        raise ValueError("PLANNING_AUTHORITY_PROMPT: prompt must not be empty")
    with trace_scope("planner"):
        emit_root_cause(
            "pipeline_boundary_start",
            stage="planning",
            operation="build_authoritative_request_catalog",
            gate="planner",
            result="START",
            details={
                "prompt": authored,
                "authority": "prompt_first_grounded_state_machine",
                "planning_state_supplied": planning_state is not None,
            },
        )
        if planning_state is None:
            if router is None:
                raise ValueError(
                    "PLANNING_AUTHORITY_STATE_REQUIRED: no raw-prompt deterministic fallback exists; "
                    "supply a model router or a plan-ready planning_state"
                )
            state = prepare_planning_state(router, authored)
        else:
            state = deepcopy(dict(planning_state))
        validate_planning_state(state, prompt=authored)
        if state.get("plan_ready") is not True:
            raise ValueError(
                "PLANNING_AUTHORITY_NOT_READY: request catalog cannot be emitted before grounded plan coverage"
            )
        catalog = build_request_catalog_from_planning_state(authored, state)
        emit_root_cause(
            "pipeline_boundary_result",
            stage="planning",
            operation="build_authoritative_request_catalog",
            gate="planner",
            result="PASS",
            details={
                "catalog": catalog,
                "planning_state_sha256": state.get("state_sha256"),
            },
        )
        return catalog


def active_authoritative_request_catalog(prompt: str) -> dict[str, Any] | None:
    active = _ACTIVE_REQUEST_CATALOG.get()
    if active is None or active[0] != prompt:
        return None
    return deepcopy(active[1])


def active_planning_state(prompt: str) -> dict[str, Any] | None:
    active = _ACTIVE_PLANNING_STATE.get()
    if active is None or active[0] != prompt:
        return None
    return deepcopy(active[1])


@contextmanager
def authoritative_request_scope(
    prompt: str,
    catalog: Mapping[str, Any],
    *,
    planning_state: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    catalog_token = _ACTIVE_REQUEST_CATALOG.set((prompt, deepcopy(dict(catalog))))
    state_token = None
    if planning_state is not None:
        validate_planning_state(planning_state, prompt=prompt)
        state_token = _ACTIVE_PLANNING_STATE.set((prompt, deepcopy(dict(planning_state))))
    try:
        yield
    finally:
        if state_token is not None:
            _ACTIVE_PLANNING_STATE.reset(state_token)
        _ACTIVE_REQUEST_CATALOG.reset(catalog_token)


__all__ = [
    "active_authoritative_request_catalog",
    "active_planning_state",
    "authoritative_request_scope",
    "build_authoritative_request_catalog",
]
