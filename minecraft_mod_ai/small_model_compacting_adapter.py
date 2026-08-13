from __future__ import annotations

from typing import Any

from .small_model_context_compaction import compact_messages


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


__all__ = ["CompactingAdapter"]
