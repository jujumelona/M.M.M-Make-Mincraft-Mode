from __future__ import annotations

"""One final reconciliation layer for approval-bound execution and public API stability."""

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from .fabric_immutable_rebind_contract import install as install_fabric_rebind
    from .immutable_platform_execution_contract import install as install_immutable_platform
    from .planner_graph_integrity_contract import install as install_planner_graph_integrity
    from .runtime_regression_reconciliation import install as install_regression_reconciliation
    from .runtime_wrapper_integrity import verify_installed_wrappers

    # PlanIR authority/provenance is folded into its pre-existing runtime owner so the
    # mutation surface does not grow just to harden provenance.
    install_immutable_platform()
    install_fabric_rebind()
    install_planner_graph_integrity()
    install_regression_reconciliation()
    verify_installed_wrappers()

    _INSTALLED = True


__all__ = ["install"]
