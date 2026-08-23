from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .runtime_contract_wrappers import contract_wraps, has_contract_marker

_REPAIR_ROUTER_MARKER = "_mmm_host_evidence_repair_router"


def _single_stream_active() -> bool:
    """Return whether the native runtime currently exposes at most one decode lane.

    Candidate breadth is owned by ``agentic_search_efficiency_contract``. This helper
    remains as a compatibility/query primitive for callers that need to inspect the
    active native topology without installing another count policy.
    """

    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "").strip()
    if not raw:
        return True
    try:
        return int(raw) <= 1
    except ValueError:
        return True


class _HostEvidenceRepairRouter:
    """Keep repair tools optional when the host already indexed fresh evidence."""

    def __init__(self, router: Any) -> None:
        self._router = router

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def bind_agent_workspace(
        self,
        workspace_root: str | Path,
        *,
        require_fresh_evidence: bool = False,
    ) -> "_HostEvidenceRepairRouter":
        del require_fresh_evidence
        self._router.bind_agent_workspace(
            workspace_root,
            require_fresh_evidence=False,
        )
        return self

    def generate_text(self, role: str, messages: Any, **kwargs: Any) -> str:
        return self._router.generate_text(role, messages, **kwargs)


def _host_evidence_repair_router(router: Any) -> Any:
    if isinstance(router, _HostEvidenceRepairRouter):
        return router
    return _HostEvidenceRepairRouter(router)


def _install_repair_policy(agentic_module: Any, repair_module: Any) -> None:
    """Install only host-evidence routing; candidate breadth has one separate owner."""

    del agentic_module
    repair_cls = repair_module.RepairEngine
    current_init = repair_cls.__init__
    if has_contract_marker(current_init, _REPAIR_ROUTER_MARKER):
        return

    @contract_wraps(current_init)
    def init_with_host_evidence(self: Any, *args: Any, **kwargs: Any) -> None:
        current_init(self, *args, **kwargs)
        self.router = _host_evidence_repair_router(self.router)

    setattr(init_with_host_evidence, _REPAIR_ROUTER_MARKER, True)
    repair_cls.__init__ = init_with_host_evidence


def install(agentic_module: Any, repair_module: Any | None = None) -> None:
    """Reuse host repair evidence without installing a second candidate-width policy."""

    if repair_module is not None:
        _install_repair_policy(agentic_module, repair_module)


__all__ = [
    "_host_evidence_repair_router",
    "_single_stream_active",
    "install",
]
