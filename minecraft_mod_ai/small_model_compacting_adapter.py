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


def _is_live_compaction_wrapper(value: Any) -> bool:
    """Identify the actual wrapper implementation, not inherited marker metadata.

    ``functools.wraps`` copies a wrapped callable's ``__dict__`` by default. A later
    wrapper can therefore inherit ``_mmm_lossless_context_compaction=True`` even when
    it bypasses the compaction callable entirely. The code object cannot be copied by
    ``wraps`` and is the authoritative owner check for this runtime boundary.
    """

    code = getattr(value, "__code__", None)
    if code is None:
        return False
    filename = str(getattr(code, "co_filename", "")).replace("\\", "/")
    return (
        filename.endswith("/small_model_compacting_adapter.py")
        and str(getattr(code, "co_name", "")) == "generate_with_compaction"
    )


def install(model_router_module: Any) -> None:
    """Bind lossless tool-context compaction at the live model-router boundary."""
    current = model_router_module.ModelRouter._generate_with_tools
    if _is_live_compaction_wrapper(current):
        return

    @wraps(current)
    def generate_with_compaction(
        self,
        *,
        config,
        adapter,
        request,
        runtime,
        stage,
        role,
    ):
        return current(
            self,
            config=config,
            adapter=CompactingAdapter(adapter),
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    setattr(generate_with_compaction, _MARKER, True)
    model_router_module.ModelRouter._generate_with_tools = generate_with_compaction


__all__ = ["CompactingAdapter", "_is_live_compaction_wrapper", "install"]
