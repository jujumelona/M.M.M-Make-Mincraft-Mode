from __future__ import annotations

import inspect
from typing import Any

from .structured_output import StructuredOutputValidationError


class StructuredPlannerRouter:
    """Delegate structured planner generation through a bounded tool-free path.

    Research, ecosystem discovery and project evidence are collected before the
    production-outline/page phase. Re-entering the agent tool loop while serializing a
    bounded JSON contract repeats retrieval and adds extra model round-trips. Structured
    planner JSON is therefore tool-free and, if host validation rejects one complete
    response, regenerated exactly once before the validation error is allowed to escape.
    Ordinary game design, research and coder turns keep their normal tool policy.
    """

    def __init__(self, router: Any) -> None:
        self._router = router
        self._accepts_enable_tools = self._supports_keyword(
            getattr(router, "generate_text", None),
            "enable_tools",
        )

    @staticmethod
    def _supports_keyword(function: Any, name: str) -> bool:
        if not callable(function):
            return False
        try:
            parameters = inspect.signature(function).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or parameter.name == name
            for parameter in parameters
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def generate_text(self, role: str, messages: Any, **kwargs: Any) -> str:
        structured_planner = role == "planner" and kwargs.get("response_format") == "json"
        if self._accepts_enable_tools and structured_planner:
            kwargs["enable_tools"] = False

        try:
            return self._router.generate_text(role, messages, **kwargs)
        except StructuredOutputValidationError:
            if not structured_planner:
                raise
            # Retry the entire structured generation once. Do not mutate or locally
            # repair the malformed response, and do not recurse through this proxy.
            return self._router.generate_text(role, messages, **kwargs)


def structured_planner_router(router: Any) -> Any:
    if isinstance(router, StructuredPlannerRouter):
        return router
    return StructuredPlannerRouter(router)


__all__ = ["StructuredPlannerRouter", "structured_planner_router"]
