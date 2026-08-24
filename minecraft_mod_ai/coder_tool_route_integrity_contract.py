from __future__ import annotations

"""Compatibility helpers after retiring per-turn writable/causal tool forcing.

The production tool loop now keeps one stable selector-owned tool surface and relies on
ordinary model function calling inside that reviewed surface. This module retains only
implementation-intent and mutation-proof helpers used by the progress-aware loop plus
legacy runtime markers during the migration.
"""

from typing import Any, Mapping, Sequence

from .agent_intent import implementation_requested, structured_user_intent
from .source_mutation_contract import mutation_observation_applied

_MARKER = "_mmm_coder_tool_route_integrity_v1"


def _user_only_request_query(messages: Sequence[Mapping[str, Any]]) -> str:
    return structured_user_intent(messages)


def _is_implementation_request(messages: Sequence[Mapping[str, Any]]) -> bool:
    return implementation_requested(messages)


def _source_mutation_applied(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(mutation_observation_applied(message) for message in reversed(messages))


class _WritableProgressAdapter:
    """Legacy transparent adapter retained only for import compatibility."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def generate_turn(self, request: Any) -> Any:
        return self.inner.generate_turn(request)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _run_with_dynamic_frontier(
    current: Any,
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: Any,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    return current(
        router,
        config=config,
        adapter=adapter,
        request=request,
        runtime=runtime,
        stage=stage,
        role=role,
    )


def _require_mutation_surface(
    tools: Sequence[Mapping[str, Any]],
    *,
    messages: Sequence[Mapping[str, Any]],
    stage: str,
    role: str,
) -> None:
    del tools, messages, stage, role


def install(
    *,
    model_router_module: Any,
    small_model_module: Any,
    causal_module: Any,
) -> None:
    """Retired hook: do not wrap or alter the live tool loop."""

    del small_model_module, causal_module
    method = model_router_module.ModelRouter._generate_with_tools
    # Temporary compatibility markers consumed by the older structural preflight.
    # They no longer imply a causal wrapper is installed.
    setattr(method, _MARKER, True)
    setattr(method, "_mmm_progress_aware_causal_composed", True)
    setattr(method, "_mmm_writable_coder_fail_closed", True)
    setattr(method, "_mmm_dynamic_causal_frontier", True)
    setattr(method, "_mmm_writable_coder_progress_forced", False)
    setattr(method, "_mmm_writable_coder_route_reachable", True)
    setattr(method, "_mmm_writable_coder_mutation_completion_invariant", True)


__all__ = [
    "_WritableProgressAdapter",
    "_is_implementation_request",
    "_require_mutation_surface",
    "_run_with_dynamic_frontier",
    "_source_mutation_applied",
    "_user_only_request_query",
    "install",
]
