from __future__ import annotations

"""Freeze the user request contract before any model-owned planning can run.

This module deliberately sits outside model generation. Requirement identity, source
spans, and hashes are derived from the raw user prompt once and then injected back into
the design returned by ``GameDesignPlanner``. Downstream evidence planning therefore
cannot promote model-invented modules or prose into mandatory user requirements.
"""

from functools import wraps
from typing import Any

from . import evidence_first_planning as _evidence
from .game_design import GameDesignPlanner

_INSTALLED = False


def build_authoritative_request_catalog(prompt: str) -> dict[str, Any]:
    """Build a prompt-only immutable request catalog.

    Passing an empty design is intentional: it prevents model-produced modules,
    features, acceptance prose, or reuse hints from participating in requirement
    identity. The existing evidence compiler still validates all prompt spans/hashes.
    """

    return _evidence.build_request_catalog(prompt, {})


def install_evidence_request_guard() -> None:
    """Install the pre-model request freeze exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    original_plan = GameDesignPlanner.plan
    if getattr(original_plan, "__mmm_request_contract_guard__", False):
        _INSTALLED = True
        return

    @wraps(original_plan)
    def guarded_plan(self: GameDesignPlanner, prompt: str, *args: Any, **kwargs: Any):
        # This must happen before ``original_plan`` because that call may invoke a
        # model. No model output is available or consulted at this boundary.
        request_catalog = build_authoritative_request_catalog(prompt)
        result = original_plan(self, prompt, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            return result
        design, proposal = result
        if not isinstance(design, dict):
            return result
        frozen_design = dict(design)
        frozen_design["_evidence_request_catalog"] = request_catalog
        return frozen_design, proposal

    guarded_plan.__mmm_request_contract_guard__ = True  # type: ignore[attr-defined]
    GameDesignPlanner.plan = guarded_plan  # type: ignore[method-assign]
    _INSTALLED = True


__all__ = ["build_authoritative_request_catalog", "install_evidence_request_guard"]
