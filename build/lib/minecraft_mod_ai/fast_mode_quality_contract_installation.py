from __future__ import annotations

"""Install the quality-preserving Fast Mode contract for coder grounding.

Fast Mode may reduce orchestration overhead, but it must never shrink the model request,
output allowance, or exact project-source grounding supplied to the coder.  The
canonical generator historically imposed a 4 KiB source-context cap only when Fast
Mode was enabled; this installer forces Fast and normal mode through the same live
request-capacity calculation.
"""

from functools import wraps
from types import ModuleType
from typing import Any

_MARKER = "_mmm_fast_mode_quality_contract_v1"


def install(target_module: ModuleType | None = None) -> None:
    """Make coder grounding capacity invariant to the Fast Mode flag."""

    if target_module is None:
        from . import custom_module_generator as target_module

    current = target_module._coder_project_context_budget
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def quality_preserving_coder_project_context_budget(
        router: Any,
        policy: Any,
        *,
        fast_mode: bool,
    ) -> int:
        # Fast Mode is an orchestration policy.  It is not permission to lower the
        # coder's evidence/context budget.  Delegate to the canonical normal-mode
        # calculation so live request capacity and host policy remain authoritative.
        del fast_mode
        return current(router, policy, fast_mode=False)

    setattr(quality_preserving_coder_project_context_budget, _MARKER, True)
    target_module._coder_project_context_budget = quality_preserving_coder_project_context_budget


__all__ = ["install"]
