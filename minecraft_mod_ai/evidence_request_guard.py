from __future__ import annotations

"""Freeze the authored request before model-owned design planning.

Raw request text, source spans, hashes, and mandatory scope remain host-owned. A
semantic router may interpret each authored span into canonical gameplay capabilities,
but it cannot invent source text or erase a span. Multi-capability spans are expanded
into independent requirement records before reuse/gap/task planning so every semantic
root receives its own proof and implementation path.
"""

from functools import wraps
from typing import Any

from . import evidence_first_planning as _evidence
from .game_design import GameDesignPlanner

_INSTALLED = False
_ORIGINAL_BUILD_REQUEST_CATALOG = _evidence.build_request_catalog


def _split_multi_root_requirements(catalog: dict[str, Any]) -> dict[str, Any]:
    raw_requirements = catalog.get("requirements")
    if not isinstance(raw_requirements, list):
        return catalog

    expanded: list[dict[str, Any]] = []
    changed = False
    for raw in raw_requirements:
        if not isinstance(raw, dict):
            expanded.append(raw)
            continue
        roots = tuple(
            dict.fromkeys(
                str(value).strip().removeprefix("capability:")
                for value in raw.get("gameplay_capabilities", ())
                if str(value).strip()
            )
        )
        if len(roots) <= 1:
            expanded.append(dict(raw))
            continue

        changed = True
        statement = str(raw.get("statement") or "").strip()
        base_requirement_id = str(raw.get("requirement_id") or "requirement")
        for root_index, root in enumerate(roots):
            requirement_id = _evidence._stable_id(
                "req",
                root,
                {
                    "semantic_parent": base_requirement_id,
                    "root_index": root_index,
                },
            )
            item = dict(raw)
            item["requirement_id"] = requirement_id
            item["capability"] = root
            item["provides"] = [_evidence._canonical_capability(root)]
            item["gameplay_capabilities"] = [root]
            item["artifact_task_ids"] = [
                _evidence._stable_id(
                    "task",
                    root,
                    {"requirement_id": requirement_id, "layer": "artifact"},
                )
            ]
            item["acceptance"] = [
                f"Verify the observable outcome for {root}: {statement}"
            ]
            expanded.append(item)

    if not changed:
        return catalog

    normalized = dict(catalog)
    normalized["requirements"] = expanded
    normalized["catalog_sha256"] = ""
    normalized["catalog_sha256"] = _evidence._hash_without(
        normalized, "catalog_sha256"
    )
    return normalized


def _build_request_catalog_with_semantic_root_expansion(
    prompt: str,
    game_design: Any,
    router: Any | None = None,
) -> dict[str, Any]:
    catalog = _ORIGINAL_BUILD_REQUEST_CATALOG(
        prompt,
        game_design,
        router=router,
    )
    return _split_multi_root_requirements(catalog)


def build_authoritative_request_catalog(
    prompt: str,
    router: Any | None = None,
) -> dict[str, Any]:
    """Build an immutable prompt-span catalog with semantic-model interpretation.

    Passing an empty design is intentional: model-produced design modules, features,
    acceptance prose, and reuse hints cannot participate in requirement identity.
    When a router is available it owns only semantic interpretation of the already
    frozen authored spans.
    """

    catalog = _ORIGINAL_BUILD_REQUEST_CATALOG(prompt, {}, router=router)
    return _split_multi_root_requirements(catalog)


def install_evidence_request_guard() -> None:
    """Install request freezing and semantic-root expansion exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    if not getattr(
        _evidence.build_request_catalog,
        "__mmm_semantic_root_expansion__",
        False,
    ):
        _build_request_catalog_with_semantic_root_expansion.__mmm_semantic_root_expansion__ = True  # type: ignore[attr-defined]
        _evidence.build_request_catalog = _build_request_catalog_with_semantic_root_expansion  # type: ignore[assignment]

    original_plan = GameDesignPlanner.plan
    if getattr(original_plan, "__mmm_request_contract_guard__", False):
        _INSTALLED = True
        return

    @wraps(original_plan)
    def guarded_plan(self: GameDesignPlanner, prompt: str, *args: Any, **kwargs: Any):
        if not prompt.strip():
            return original_plan(self, prompt, *args, **kwargs)
        request_catalog = build_authoritative_request_catalog(
            prompt,
            router=self.router,
        )
        result = original_plan(self, prompt, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            return result
        design, proposal = result
        if not isinstance(design, dict):
            return result
        frozen_design = dict(design)
        frozen_design["_evidence_request_catalog"] = request_catalog
        return frozen_design, proposal

    guarded_plan.__mmm_request_contract_guard__ = True  # type: ignore[attr-defined]
    GameDesignPlanner.plan = guarded_plan  # type: ignore[method-assign]
    _INSTALLED = True


__all__ = ["build_authoritative_request_catalog", "install_evidence_request_guard"]
