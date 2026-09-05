from __future__ import annotations

"""Freeze the authored request before model-owned design planning.

Raw request text, source spans, hashes, and mandatory scope remain host-owned. A
semantic router may interpret each authored span into canonical gameplay capabilities,
but it cannot invent source text or erase a span. Multi-capability spans are expanded
into independent requirement records before reuse/gap/task planning so every semantic
root receives its own proof and implementation path.
"""

import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from functools import wraps
from typing import Any

from . import evidence_first_planning as _evidence
from .game_design import GameDesignPlanner
from .requirement_acceptance import requirement_acceptance

_INSTALLED = False
_ORIGINAL_BUILD_REQUEST_CATALOG = _evidence.build_request_catalog
_ACTIVE_REQUEST_CATALOG: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar(
    "mmm_active_authoritative_request_catalog",
    default=None,
)


def active_authoritative_request_catalog(prompt: str) -> dict[str, Any] | None:
    """Return the frozen catalog, with a host-only safe fallback if no guard is active."""

    active = _ACTIVE_REQUEST_CATALOG.get()
    if active is not None and active[0] == prompt:
        return dict(active[1])
    return build_authoritative_request_catalog(prompt, router=None)


class _StrictSemanticRouterProxy:
    """Observe production semantic failures even if a downstream caller catches them."""

    def __init__(self, router: Any) -> None:
        self._router = router
        self.failure_reason = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        try:
            raw = self._router.generate_text(*args, **kwargs)
        except Exception as exc:
            self.failure_reason = f"semantic router call failed: {type(exc).__name__}: {exc}"
            raise

        if kwargs.get("response_format") != "json":
            return raw

        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            self.failure_reason = f"semantic router returned invalid JSON: {type(exc).__name__}: {exc}"
            return raw

        if not isinstance(payload, Mapping):
            self.failure_reason = "semantic router JSON root is not an object"
            return raw

        candidates = payload.get("gameplay_capability_candidates")
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes, bytearray)
        ):
            self.failure_reason = "semantic router omitted gameplay_capability_candidates"
            return raw
        roots = tuple(str(item).strip() for item in candidates if str(item).strip())
        if not roots:
            self.failure_reason = "semantic router returned no gameplay capability roots"
        if bool(payload.get("unresolved", False)):
            self.failure_reason = "semantic router marked the authored requirement unresolved"
        return raw


def _original_catalog_strict(
    prompt: str,
    game_design: Any,
    *,
    router: Any | None,
) -> dict[str, Any]:
    if router is None:
        return _ORIGINAL_BUILD_REQUEST_CATALOG(prompt, game_design, router=None)
    proxy = _StrictSemanticRouterProxy(router)
    catalog = _ORIGINAL_BUILD_REQUEST_CATALOG(prompt, game_design, router=proxy)
    if proxy.failure_reason:
        raise _evidence.EvidencePlanError(
            "production semantic interpretation failed closed: " + proxy.failure_reason
        )
    return catalog


def _normalize_public_acceptance_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Re-establish the public/internal acceptance boundary for every requirement."""

    raw_requirements = catalog.get("requirements")
    if not isinstance(raw_requirements, list):
        return dict(catalog)

    requirements: list[Any] = []
    changed = False
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            requirements.append(raw)
            continue
        item = dict(raw)
        raw_acceptance = item.get("acceptance")
        candidates = (
            raw_acceptance
            if isinstance(raw_acceptance, Sequence)
            and not isinstance(raw_acceptance, (str, bytes, bytearray))
            else ()
        )
        capability = str(item.get("capability") or "requested behavior").strip()
        public_acceptance = list(requirement_acceptance(capability, candidates))
        if item.get("acceptance") != public_acceptance:
            item["acceptance"] = public_acceptance
            changed = True
        requirements.append(item)

    if not changed:
        return dict(catalog)

    normalized = dict(catalog)
    normalized["requirements"] = requirements
    normalized["catalog_sha256"] = ""
    normalized["catalog_sha256"] = _evidence._hash_without(
        normalized, "catalog_sha256"
    )
    return normalized


def _split_multi_root_requirements(catalog: dict[str, Any]) -> dict[str, Any]:
    catalog = _normalize_public_acceptance_catalog(catalog)
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
                f"Verify the observable player-facing behavior for capability {root}."
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
    catalog = _original_catalog_strict(
        prompt,
        game_design,
        router=router,
    )
    return _split_multi_root_requirements(catalog)


def build_authoritative_request_catalog(
    prompt: str,
    router: Any | None = None,
) -> dict[str, Any]:
    """Build one immutable host-owned request catalog."""

    if router is None:
        catalog = _original_catalog_strict(prompt, {}, router=None)
    else:
        from .semantic_batching_contract import build_bounded_requirement_catalog

        catalog = build_bounded_requirement_catalog(prompt, router=router)
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

    request_catalog_builder = build_authoritative_request_catalog

    @wraps(original_plan)
    def guarded_plan(self: GameDesignPlanner, prompt: str, *args: Any, **kwargs: Any):
        if not prompt.strip():
            return original_plan(self, prompt, *args, **kwargs)
        request_catalog = request_catalog_builder(
            prompt,
            router=self.router,
        )
        token = _ACTIVE_REQUEST_CATALOG.set((prompt, dict(request_catalog)))
        try:
            result = original_plan(self, prompt, *args, **kwargs)
        finally:
            _ACTIVE_REQUEST_CATALOG.reset(token)
        if not isinstance(result, tuple) or len(result) != 2:
            return result
        design, proposal = result
        if not isinstance(design, dict):
            return result
        observed = design.get("_evidence_request_catalog")
        if not isinstance(observed, Mapping) or dict(observed) != dict(request_catalog):
            raise _evidence.EvidencePlanError(
                "PLAN_ORDER_VIOLATION: platform/reuse planning did not consume the exact "
                "authoritative request catalog."
            )
        from .reuse_planner import validate_pre_retrieval_plan

        frozen_plan = design.get("_pre_retrieval_plan")
        if not isinstance(frozen_plan, Mapping):
            raise _evidence.EvidencePlanError(
                "PLAN_ORDER_VIOLATION: semantic work was not frozen before retrieval."
            )
        validate_pre_retrieval_plan(frozen_plan, prompt=prompt, design=design)
        return design, proposal

    guarded_plan.__mmm_request_contract_guard__ = True  # type: ignore[attr-defined]
    GameDesignPlanner.plan = guarded_plan  # type: ignore[method-assign]
    _INSTALLED = True


__all__ = [
    "active_authoritative_request_catalog",
    "build_authoritative_request_catalog",
    "install_evidence_request_guard",
]
