from __future__ import annotations

"""Target-scoped research bound to an already validated platform receipt.

This module deliberately has no fallback artifact. A research failure propagates to the
planning stage that requested it, so incomplete evidence cannot become executable state.
"""

from collections.abc import Mapping
from typing import Any, Callable

from .platform_catalog import PlatformAdapter
from .spec import SpecValidationError

TargetResearchFn = Callable[[PlatformAdapter], Mapping[str, Any]]


def target_research_callback(research_brief: Mapping[str, Any]) -> TargetResearchFn:
    """Return a fail-closed evidence function for validated adapter receipts."""

    frozen_brief = dict(research_brief)

    def retrieve(adapter: PlatformAdapter) -> Mapping[str, Any]:
        from . import central_research, retrieval
        from .agentic_research_fusion import retrieve_target_agentic_evidence

        adapter.validate()
        brief = {
            **frozen_brief,
            "_mmm_platform_target": {
                "adapter_id": adapter.adapter_id,
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
            },
        }
        value = retrieve_target_agentic_evidence(
            brief,
            central_module=central_research,
            retrieve=retrieval.retrieve_official_evidence,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            mappings=adapter.yarn_mappings,
        )
        if not isinstance(value, Mapping):
            raise SpecValidationError(
                "Target research returned a non-object evidence receipt."
            )
        payload = dict(value)
        if payload.get("status") == "unavailable":
            raise SpecValidationError(
                "Target research explicitly reported unavailable evidence."
            )
        if not str(payload.get("schema_version") or "").strip():
            raise SpecValidationError(
                "Target research returned evidence without schema_version."
            )
        return payload

    return retrieve


__all__ = ["TargetResearchFn", "target_research_callback"]
