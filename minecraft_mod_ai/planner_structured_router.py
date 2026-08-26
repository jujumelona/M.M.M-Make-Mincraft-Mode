from __future__ import annotations

import inspect
from typing import Any


class StructuredPlannerRouter:
    """Delegate to one router while making structured planner decode tool-free.

    Research, ecosystem discovery and project evidence are collected before the
    production-outline/page phase. Re-entering the agent tool loop while serializing a
    bounded JSON contract repeats retrieval and adds extra model round-trips. This
    proxy changes only calls made through the structured planner path; native structured
    validation/retry is owned by the llama structured-decode policy so this layer never
    adds another retry. Ordinary game design, research and coder turns keep their normal
    tool policy.
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
        if (
            self._accepts_enable_tools
            and role == "planner"
            and kwargs.get("response_format") == "json"
        ):
            kwargs["enable_tools"] = False
        return self._router.generate_text(role, messages, **kwargs)


def structured_planner_router(router: Any) -> Any:
    if isinstance(router, StructuredPlannerRouter):
        return router
    return StructuredPlannerRouter(router)


__all__ = ["StructuredPlannerRouter", "structured_planner_router"]
