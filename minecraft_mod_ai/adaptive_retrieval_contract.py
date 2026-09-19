from __future__ import annotations

from typing import Any


def install(model_router_module: Any) -> None:
    """Install repository grounding and adaptive execution hardening."""

    if not (
        bool(getattr(model_router_module.ModelRouter._generate_with_tools, "_mmm_progress_aware_tool_loop_owner", False))
        or bool(getattr(model_router_module.ModelRouter._generate_with_tools, "_mmm_dynamic_causal_frontier", False))
    ):
        raise RuntimeError(
            "ModelRouter must directly own the canonical production tool loop."
        )

    from .agent_tool_allowlist_hardening import harden_agent_tool_allowlist
    from .hybrid_route_hardening import harden_code_search_routes
    from .inference_time_scaling import harden_runtime
    from .runtime_composer_hardening import harden_runtime_composer_identity

    harden_runtime()
    harden_code_search_routes()
    harden_runtime_composer_identity()
    harden_agent_tool_allowlist()
    _expose_composed_repair_contracts()


def _inherit_boolean_contract_markers(current: Any) -> None:
    wrapped = getattr(current, "__wrapped__", None)
    while callable(wrapped):
        for name, value in vars(wrapped).items():
            if name.startswith("_mmm_") and value is True:
                setattr(current, name, True)
        wrapped = getattr(wrapped, "__wrapped__", None)


def _expose_composed_repair_contracts() -> None:
    from .repair_engine import RepairEngine

    _inherit_boolean_contract_markers(RepairEngine._signature)
