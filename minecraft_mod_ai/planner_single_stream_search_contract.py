from __future__ import annotations

import os
from functools import wraps
from pathlib import Path
from typing import Any

_REPAIR_MARKER = "_mmm_single_stream_repair_search"
_REPAIR_ROUTER_MARKER = "_mmm_host_evidence_repair_router"


def _single_stream_active() -> bool:
    """Return whether repair search should assume one native decode lane."""
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
    current_count = agentic_module._repair_candidate_count
    if not getattr(current_count, _REPAIR_MARKER, False):

        @wraps(current_count)
        def repair_candidate_count(self: Any, evidence: Any, memory: Any) -> int:
            mode = agentic_module._mode()
            if mode == "on":
                return current_count(self, evidence, memory)
            if mode == "auto" and _single_stream_active():
                return 1
            return current_count(self, evidence, memory)

        setattr(repair_candidate_count, _REPAIR_MARKER, True)
        agentic_module._repair_candidate_count = repair_candidate_count

    repair_cls = repair_module.RepairEngine
    current_init = repair_cls.__init__
    if getattr(current_init, _REPAIR_ROUTER_MARKER, False):
        return

    @wraps(current_init)
    def init_with_host_evidence(self: Any, *args: Any, **kwargs: Any) -> None:
        current_init(self, *args, **kwargs)
        self.router = _host_evidence_repair_router(self.router)

    setattr(init_with_host_evidence, _REPAIR_ROUTER_MARKER, True)
    repair_cls.__init__ = init_with_host_evidence


def install(agentic_module: Any, repair_module: Any | None = None) -> None:
    """Apply single-stream policy only to repair; Planner is host-owned directly."""
    if repair_module is not None:
        _install_repair_policy(agentic_module, repair_module)


__all__ = [
    "_host_evidence_repair_router",
    "_single_stream_active",
    "install",
]
