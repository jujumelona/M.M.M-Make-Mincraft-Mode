from __future__ import annotations

"""Preserve host-issued mutation target identity across localization evidence.

PlanIR mutation authority and coder target-existence reconciliation are composed by
runtime wrappers.  Their contract must not depend on a particular ``evidence_source``
label: those labels describe how existence was learned, not whether a host-owned path
is authoritative.  This final reconciliation pins any target that is present in the
host-owned exact writable set before repository retrieval can merge localization data.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

_MARKER = "_mmm_mutation_authority_pin_v1"


def _union_paths(loop_module: Any, *groups: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for group in groups:
        for raw in group:
            path = loop_module._canonical_mutation_path(raw)
            if path and path not in result:
                result.append(path)
    return tuple(result)


def install(loop_module: Any | None = None) -> None:
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module

    if getattr(loop_module, _MARKER, False):
        return

    original_extract = loop_module._extract_mutation_context_from_payload

    def extract(payload: Any):
        context = original_extract(payload)
        if context is None:
            return None

        parser = getattr(loop_module, "_planir_owned_anchor_sets", None)
        if not callable(parser):
            return context

        writable, creatable = parser(payload)
        target = loop_module._canonical_mutation_path(
            getattr(context, "target_path", None)
        )
        if not target or target not in set(writable):
            return context

        # Host-owned task identity is authoritative independently of whether the
        # destination already exists. Retrieval may add source/body evidence, but it
        # must never replace this target with a related repository hit.
        if not hasattr(context, "target_pinned"):
            raise RuntimeError(
                "HOST_MUTATION_AUTHORITY_MISMATCH: PlanIR authority context is not "
                "installed before mutation pin reconciliation."
            )

        return replace(
            context,
            target_pinned=True,
            writable_paths=_union_paths(
                loop_module,
                tuple(getattr(context, "writable_paths", ()) or ()),
                writable,
            ),
            creatable_paths=_union_paths(
                loop_module,
                tuple(getattr(context, "creatable_paths", ()) or ()),
                creatable,
            ),
        )

    loop_module._extract_mutation_context_from_payload = extract
    setattr(loop_module, _MARKER, True)


def assert_installed(loop_module: Any | None = None) -> None:
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module
    if not getattr(loop_module, _MARKER, False):
        raise RuntimeError("mutation authority pin reconciliation is not installed")


__all__ = ["assert_installed", "install"]
