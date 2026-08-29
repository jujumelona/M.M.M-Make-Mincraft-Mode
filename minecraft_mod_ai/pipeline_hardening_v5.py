from __future__ import annotations

"""Invalidate research checkpoints created before lossless page grounding."""

from collections.abc import Mapping
from typing import Any

from .pipeline_hardening import _replace_bound_references

_INSTALLED = False
_POLICY_VERSION = "lossless-page-grounding-provenance-v1"


def install_pipeline_hardening_v5() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import agentic_pre_design_rag as rag

    original = rag._domain_checkpoint_key
    if not getattr(original, "_mmm_lossless_checkpoint_key", False):
        def checkpoint_key(
            router: Any,
            *,
            prompt: str,
            domain: Mapping[str, Any],
            document: Mapping[str, Any],
        ) -> str:
            legacy = original(
                router,
                prompt=prompt,
                domain=domain,
                document=document,
            )
            return rag._sha256(
                {
                    "legacy_domain_key": legacy,
                    "research_policy": _POLICY_VERSION,
                }
            ).removeprefix("sha256:")

        checkpoint_key._mmm_lossless_checkpoint_key = True  # type: ignore[attr-defined]
        rag._domain_checkpoint_key = checkpoint_key
        _replace_bound_references(original, checkpoint_key)

    _INSTALLED = True


__all__ = ["install_pipeline_hardening_v5"]
