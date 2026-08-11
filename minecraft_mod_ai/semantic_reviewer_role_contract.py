from __future__ import annotations

from functools import wraps
from typing import Any


class _SafeReviewerRouter:
    def __init__(self, router: Any) -> None:
        self._router = router

    def generate_text(self, _ignored_role: str, *args: Any, **kwargs: Any) -> str:
        # model-registry-v2 requires coder_safe. If the profile is malformed or
        # unavailable, fail closed rather than silently falling back to the planner.
        return self._router.generate_text(
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
