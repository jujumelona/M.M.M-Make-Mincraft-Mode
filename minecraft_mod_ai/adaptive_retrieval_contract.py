from __future__ import annotations

from typing import Any

_INSTALL_MARKER = "__mmm_progress_aware_retrieval_v1__"


def install(model_router_module: Any) -> None:
    """Install progress-aware retrieval without moving router ownership.

    The router module remains the canonical owner of ModelRouter and all of its
    globals. This contract replaces only the retrieve/act/observe method so
    existing runtime wrappers and test monkeypatches keep their normal module
    semantics.
    """
    router_cls = model_router_module.ModelRouter
    current = router_cls._generate_with_tools
    if getattr(current, _INSTALL_MARKER, False):
        return

    def _generate_with_tools(
        self: Any,
        *,
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        from .progress_aware_tool_loop import generate_with_tools

        return generate_with_tools(
            self,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    setattr(_generate_with_tools, _INSTALL_MARKER, True)
    setattr(_generate_with_tools, "__wrapped__", current)
    router_cls._generate_with_tools = _generate_with_tools
