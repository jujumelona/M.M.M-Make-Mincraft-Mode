from __future__ import annotations

from functools import wraps
from typing import Any


class _SafeReviewerRouter:
    """Pin atomic semantic review to the independent safe model role."""

    def __init__(self, router: Any) -> None:
        self._router = router

    def generate_tool_decision(
        self,
        _ignored_role: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Atomic review is a structured decision, not free-form JSON text. Keep the
        # independent coder_safe role while preserving the native tool-call path.
        return self._router.generate_tool_decision(
            "coder_safe",
            *args,
            **kwargs,
        )


def install(atomic_module: Any) -> None:
    original = atomic_module.semantic_review
    if getattr(original, "_mmm_independent_reviewer_role", False):
        return

    @wraps(original)
    def reviewed(router: Any, proposal: Any, ir: dict[str, Any]):
        return original(
            _SafeReviewerRouter(router),
            proposal,
            ir,
        )

    reviewed._mmm_independent_reviewer_role = True
    atomic_module.semantic_review = reviewed
