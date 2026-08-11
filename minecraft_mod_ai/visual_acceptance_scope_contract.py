from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any, Mapping


def _index_from_ref(value: str) -> int | None:
    if not value.startswith("acceptance:"):
        return None
    tail = value.removeprefix("acceptance:")
    if not tail.isdigit():
        return None
    return int(tail)


def _visual_refs(proposal: Any) -> tuple[str, ...]:
    design = getattr(proposal, "game_design", {})
    ir = design.get("_atomic_requirement_ir") if isinstance(design, Mapping) else None
    refs: set[str] = set()
    if isinstance(ir, Mapping):
        for atom in ir.get("atoms", []):
            if (
                isinstance(atom, Mapping)
                and "visual_3d" in atom.get("evidence_dimensions", [])
            ):
                refs.update(
                    str(value)
                    for value in atom.get("acceptance_refs", [])
                    if isinstance(value, str)
                )
    tests = tuple(str(value) for value in getattr(proposal, "acceptance_tests", ()))
    for index, test in enumerate(tests):
        if test.startswith("[visual_3d]"):
            refs.add(f"acceptance:{index:08d}")
    return tuple(sorted(refs))


def install(orchestrator_module: Any) -> None:
    """Never ask the visual critic to certify non-visual acceptance tests."""

    original = orchestrator_module.visual_review
    if getattr(original, "_mmm_visual_acceptance_scope", False):
        return

    @wraps(original)
    def scoped_visual_review(
        router: Any,
        proposal: Any,
        screenshots: tuple[str, ...],
    ) -> dict[str, Any]:
        refs = _visual_refs(proposal)
        tests = tuple(str(value) for value in proposal.acceptance_tests)
        selected: list[str] = []
        selected_refs: list[str] = []
        for ref in refs:
            index = _index_from_ref(ref)
            if index is None or index < 0 or index >= len(tests):
                continue
            selected.append(tests[index])
            selected_refs.append(ref)
        if not selected:
            # A visual dimension without a scoped observable check is unresolved;
            # do not make the critic infer or invent one.
            return {
                "schema_version": "mmm/visual-review-v2",
                "status": "FAIL",
                "error": "No visual-scoped acceptance test is bound to the proposal.",
                "screenshots": list(screenshots),
                "acceptance_test_results": [],
                "atomic_acceptance_refs": [],
            }
        scoped = replace(
            proposal,
            acceptance_tests=tuple(selected),
            approval_hash="",
        )
        receipt = original(router, scoped, screenshots)
        if not isinstance(receipt, dict):
            return receipt
        return {
            **receipt,
            "atomic_acceptance_refs": selected_refs,
        }

    scoped_visual_review._mmm_visual_acceptance_scope = True
    orchestrator_module.visual_review = scoped_visual_review
