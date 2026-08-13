from __future__ import annotations

from typing import Any


class StructuredPlannerRouter:
    """Delegate to one router while making structured planner decode tool-free.

    Research, ecosystem discovery and project evidence are collected before the
    production-outline/page phase. Re-entering the agent tool loop while serializing a
    bounded JSON contract repeats retrieval and adds extra model round-trips. This
    proxy changes only calls made through the structured planner path; ordinary game
    design, research and coder turns keep their normal tool policy.
    """

    def __init__(self, router: Any) -> None:
        self._router = router

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def generate_text(self, role: str, messages: Any, **kwargs: Any) -> str:
        if role == "planner" and kwargs.get("response_format") == "json":
            kwargs["enable_tools"] = False
        return self._router.generate_text(role, messages, **kwargs)


def structured_planner_router(router: Any) -> Any:
    if isinstance(router, StructuredPlannerRouter):
        return router
    return StructuredPlannerRouter(router)


__all__ = ["StructuredPlannerRouter", "structured_planner_router"]
