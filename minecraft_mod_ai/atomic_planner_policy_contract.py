from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

from .runtime_contract_wrappers import has_contract_marker


def _reviewer_capable(router: Any) -> bool:
    registry = getattr(router, "registry", None)
    profile = getattr(router, "profile", None)
    role = getattr(registry, "role", None)
    if registry is None or not isinstance(profile, str) or not callable(role):
        return False
    try:
        role(profile, "coder_safe")
    except Exception:
        return False
    return True


def install(atomic_module: Any, complete_planner_module: Any) -> None:
    """Require semantic coverage review only when the router declares that role.

    Production ModelRouter profiles must expose coder_safe. Lightweight injected
    planner protocols used for offline analysis/tests may return a proposal carrying
    unresolved IR, but such a proposal remains non-executable for binary production.
    """

    cls = complete_planner_module.CompleteGameDesignPlanner
    current = cls.plan
    if has_contract_marker(current, "_mmm_atomic_planner_policy"):
        return
    base = getattr(current, "__wrapped__", current)

    @wraps(base)
    def plan(self: Any, *args: Any, **kwargs: Any):
        proposal = base(self, *args, **kwargs)
        ir = atomic_module.compile_ir(proposal)
        capable = _reviewer_capable(self.router)
        if ir["unresolved_atom_ids"] and capable:
            ir = atomic_module.semantic_review(self.router, proposal, ir)
        if ir["unresolved_atom_ids"] and capable:
            missing = [
                atom["text"]
                for atom in ir["atoms"]
                if atom["atom_id"] in set(ir["unresolved_atom_ids"])
            ]
            raise complete_planner_module.SpecValidationError(
                "Planner left authoritative request atoms uncovered after bounded review: "
                + " | ".join(missing[:6])
            )
        game_design = dict(proposal.game_design)
        game_design["_atomic_requirement_ir"] = ir
        return replace(
            proposal,
            game_design=game_design,
            approval_hash="",
        ).with_hash()

    plan._mmm_atomic_requirement_ir = True
    plan._mmm_atomic_planner_policy = True
    cls.plan = plan
