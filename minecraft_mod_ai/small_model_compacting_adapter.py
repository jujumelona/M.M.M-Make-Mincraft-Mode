from __future__ import annotations

from functools import wraps
from typing import Any

from .small_model_context_compaction import compact_messages


_MARKER = "_mmm_lossless_context_compaction"


class CompactingAdapter:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def generate_turn(self, request: Any) -> Any:
        messages = compact_messages(request.messages)
        if messages == tuple(request.messages):
            return self.inner.generate_turn(request)
        from .model_adapters import GenerationRequest

        return self.inner.generate_turn(
            GenerationRequest(
                messages=messages,
                media_paths=request.media_paths,
                response_format=request.response_format,
                response_schema=request.response_schema,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
            )
        )


def install(model_router_module: Any) -> None:
    """Bind lossless tool-context compaction at the model-router owner boundary."""
    current = model_router_module.ModelRouter._generate_with_tools
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate_with_compaction(self, *, adapter, request, runtime, stage, role):
        return current(
            self,
            adapter=CompactingAdapter(adapter),
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    setattr(generate_with_compaction, _MARKER, True)
    model_router_module.ModelRouter._generate_with_tools = generate_with_compaction


__all__ = ["CompactingAdapter", "install"]
