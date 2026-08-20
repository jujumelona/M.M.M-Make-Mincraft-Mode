from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

from .model_context_budget import fit_messages_to_context
from .small_model_context_compaction import compact_messages


_MARKER = "_mmm_lossless_context_compaction"


class CompactingAdapter:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def generate_turn(self, request: Any) -> Any:
        # Exchange compaction removes older history when possible. Context fitting then
        # handles the first assistant/tool exchange too, where there is no older round
        # for the historical compactor to replace.
        messages = compact_messages(request.messages)
        messages = fit_messages_to_context(
            messages,
            config=getattr(self.inner, "config", None),
            tools=getattr(request, "tools", ()) or (),
        )
        if messages == tuple(request.messages):
            return self.inner.generate_turn(request)

        # Clone the frozen GenerationRequest instead of reconstructing it field by
        # field. This preserves task/prompt/metadata and any future request fields
        # added by another runtime contract while changing only the compacted history.
        return self.inner.generate_turn(replace(request, messages=messages))


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
