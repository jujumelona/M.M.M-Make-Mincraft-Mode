from __future__ import annotations

"""Production serialization adapter for the canonical requirement acceptance contract.

All public-acceptance semantics are owned by ``acceptance_contracts``. This module only
adapts that canonical contract to production-specific exception types, catalog shapes and
quality coverage. It must not define a second acceptance policy.
"""

from collections.abc import Mapping
from functools import wraps
from typing import Any

from . import production_contract as _production
from .acceptance_contracts import (
    CANONICAL_ACCEPTANCE_OWNER,
    AcceptanceContractError,
    approved_requirements as _canonical_approved_requirements,
    canonical_public_acceptance as _canonical_acceptance_values,
    is_public_acceptance as _canonical_is_public_acceptance,
    project_requirement_public_acceptance,
    validate_public_acceptance,
)

_INSTALLED = False


def _strict_public_acceptance(value: Any) -> bool:
    """Compatibility adapter; the rule itself is owned by acceptance_contracts."""

    return _canonical_is_public_acceptance(value)


def _canonical_public_acceptance(values: Any) -> list[str]:
    """Compatibility adapter returning the canonical ordered public checks."""

    return list(_canonical_acceptance_values(values))


def _production_public_acceptance(statement: str) -> None:
    """Expose the canonical rule through ProductionContractError."""

    validate_public_acceptance(
        statement,
        error_type=_production.ProductionContractError,
    )


_production_public_acceptance._mmm_contextual_legacy_boundary = True
_production_public_acceptance._mmm_acceptance_contract_owner = CANONICAL_ACCEPTANCE_OWNER


def _install_planner_public_acceptance_guard() -> None:
    """Bind evidence planning directly to the canonical acceptance predicate."""

    from . import evidence_first_planning as _evidence

    _evidence._is_public_acceptance = _canonical_is_public_acceptance


def _validated_evidence_plan(
    evidence_plan: Mapping[str, Any] | None,
    *,
    requested_prompt: str,
) -> Mapping[str, Any] | None:
    """Validate the frozen contract without rewriting authored acceptance or task IDs."""

    if isinstance(evidence_plan, Mapping):
        from .evidence_first_planning import validate_evidence_first_plan

        validate_evidence_first_plan(evidence_plan, prompt=requested_prompt)
    return evidence_plan


def _filter_evidence_input_acceptance(
    acceptance_tests: Any,
    evidence_plan: Mapping[str, Any] | None,
) -> Any:
    """Drop non-authoritative internal acceptance before evidence-mode compilation."""

    if not isinstance(evidence_plan, Mapping):
        return acceptance_tests
    if isinstance(acceptance_tests, (str, bytes, bytearray)):
        return acceptance_tests
    try:
        values = tuple(acceptance_tests)
    except TypeError:
        return acceptance_tests
    return tuple(
        value
        for value in values
        if not isinstance(value, str) or _canonical_is_public_acceptance(value)
    )


def _approved_requirements(
    evidence_plan: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    """Compatibility adapter to the canonical requirement authority extractor."""

    return _canonical_approved_requirements(evidence_plan)


def _approved_acceptance(requirement: Mapping[str, Any]) -> str:
    """Project every approved public check without imposing a count limit."""

    try:
        return project_requirement_public_acceptance(requirement)
    except AcceptanceContractError as exc:
        raise _production.ProductionContractError(str(exc)) from exc


def _requirement_context(
    evidence_plan: Mapping[str, Any], requirement_id: str
) -> tuple[str, set[str], set[str]]:
    approved = _approved_requirements(evidence_plan).get(requirement_id, {})
    span = approved.get("source_span") if isinstance(approved, Mapping) else {}
    text = " ".join(
        str(value or "")
        for value in (
            approved.get("capability") if isinstance(approved, Mapping) else "",
            approved.get("semantic_statement") if isinstance(approved, Mapping) else "",
            span.get("text") if isinstance(span, Mapping) else "",
            " ".join(approved.get("acceptance", []))
            if isinstance(approved, Mapping)
            and isinstance(approved.get("acceptance"), list)
            else "",
        )
    )
    predicates: set[str] = set()
    artifact_kinds: set[str] = set()
    tasks = evidence_plan.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            refs = task.get("requirement_refs")
            if not isinstance(refs, list) or requirement_id not in {
                str(value) for value in refs
            }:
                continue
            values = task.get("conditional_predicates")
            if isinstance(values, list):
                predicates.update(str(value) for value in values)
            artifacts = task.get("artifact_obligations")
            if isinstance(artifacts, list):
                artifact_kinds.update(
                    str(item.get("kind"))
                    for item in artifacts
                    if isinstance(item, Mapping) and str(item.get("kind") or "")
                )
    return text, predicates, artifact_kinds


def _conditional_dimensions(
    evidence_plan: Mapping[str, Any],
    requirement_id: str,
    active_ids: set[str],
) -> list[str]:
    text, predicates, artifact_kinds = _requirement_context(
        evidence_plan, requirement_id
    )
    selected: list[str] = []
    for dimension_id in _production._CONDITIONAL_ORDER:
        if dimension_id not in active_ids:
            continue
        triggered = _production._text_triggers_dimension(text, dimension_id)
        if dimension_id == "visual_3d":
            triggered = (
                triggered
                or "needs_client_render" in predicates
                or bool(
                    artifact_kinds
                    & {
                        "client_visual_or_ui_resource",
                        "data_or_client_resource",
                    }
                )
            )
        elif dimension_id == "state_save_migration":
            triggered = triggered or "needs_persistence" in predicates
        elif dimension_id == "multiplayer":
            triggered = triggered or "needs_network" in predicates
        if triggered:
            selected.append(dimension_id)
    return selected


def _rewrite_compilation(
    compilation: Any,
    *,
    modules: Any,
    assets: Any,
    evidence_plan: Mapping[str, Any] | None,
) -> Any:
    if not isinstance(evidence_plan, Mapping):
        return compilation
    contract = dict(compilation.contract)
    approved = _approved_requirements(evidence_plan)
    if not approved:
        raise _production.ProductionContractError(
            "evidence-mode production compilation has no approved requirement authority"
        )

    catalog = [dict(item) for item in contract.get("acceptance_catalog", [])]
    approved_statements = {
        req_id: _approved_acceptance(req) for req_id, req in approved.items()
    }
    # Input acceptance is supplementary in evidence mode. Remove both the exact
    # requirement-scoped projection and every individual canonical check so the same
    # contract cannot appear twice under different origins.
    authoritative_public = set(approved_statements.values())
    for requirement in approved.values():
        authoritative_public.update(
            _canonical_acceptance_values(requirement.get("acceptance"))
        )

    removed_refs: set[str] = set()
    rewritten_catalog: list[dict[str, Any]] = []
    seen_public: set[str] = set()
    for item in catalog:
        origin = str(item.get("origin") or "")
        ref = str(item.get("acceptance_ref") or "")
        if origin == "requirement":
            req_id = ref.removeprefix("acceptance:")
            if req_id not in approved_statements:
                raise _production.ProductionContractError(
                    f"production acceptance invented an unknown requirement identity: {req_id}"
                )
            item["statement"] = approved_statements[req_id]
        elif (
            origin == "input"
            and str(item.get("statement") or "") in authoritative_public
        ):
            removed_refs.add(ref)
            continue

        if item.get("visibility") == "public":
            statement = str(item.get("statement") or "")
            try:
                _production_public_acceptance(statement)
            except _production.ProductionContractError as exc:
                raise _production.ProductionContractError(
                    f"public acceptance leaked an internal task invariant: {ref}"
                ) from exc
            if statement in seen_public:
                raise _production.ProductionContractError(
                    "duplicate public acceptance statement would destroy requirement "
                    f"traceability: {ref}"
                )
            seen_public.add(statement)
        rewritten_catalog.append(item)
    contract["acceptance_catalog"] = rewritten_catalog

    active_ids = {
        str(item.get("dimension_id"))
        for item in contract.get("quality_dimension_catalog", [])
        if isinstance(item, Mapping)
    }
    groups: list[dict[str, Any]] = []
    for raw in contract.get("coverage_groups", []):
        if not isinstance(raw, Mapping):
            continue
        group = dict(raw)
        req_id = str(group.get("requirement_ref") or "")
        if req_id not in approved:
            raise _production.ProductionContractError(
                f"coverage group references a non-authoritative requirement: {req_id}"
            )
        refs = [
            str(value)
            for value in group.get("acceptance_refs", [])
            if str(value) not in removed_refs
        ]
        canonical_ref = f"acceptance:{req_id}"
        if canonical_ref not in refs:
            refs.insert(0, canonical_ref)
        group["acceptance_refs"] = list(dict.fromkeys(refs))

        dimensions = [
            value for value in _production._BASELINE_DIMENSIONS if value in active_ids
        ]
        dimensions.extend(
            value
            for value in _conditional_dimensions(evidence_plan, req_id, active_ids)
            if value not in dimensions
        )
        group["quality_dimension_refs"] = [f"quality:{value}" for value in dimensions]
        group["evidence_route_refs"] = [f"evidence:{value}" for value in dimensions]
        groups.append(group)
    contract["coverage_groups"] = groups

    public_tuple = tuple(
        str(item["statement"])
        for item in rewritten_catalog
        if item.get("visibility") == "public"
    )
    stats = dict(contract.get("catalog_stats") or {})
    stats["acceptance_tests"] = len(rewritten_catalog)
    stats["coverage_groups"] = len(groups)
    contract["catalog_stats"] = stats
    contract["contract_sha256"] = ""
    contract["contract_sha256"] = _production._hash_without_field(
        contract, "contract_sha256"
    )
    _production.validate_production_contract(
        contract,
        modules,
        public_tuple,
        assets,
        evidence_plan,
    )
    return _production.ProductionContractCompilation(
        contract=contract,
        acceptance_tests=public_tuple,
    )


def install_production_boundary_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    _install_planner_public_acceptance_guard()
    if getattr(
        _production._validate_public_acceptance,
        "_mmm_acceptance_contract_owner",
        "",
    ) != CANONICAL_ACCEPTANCE_OWNER:
        _production._validate_public_acceptance = _production_public_acceptance

    original = _production.compile_production_contract
    if not getattr(original, "_mmm_authority_acceptance_projection", False):

        @wraps(original)
        def compile_contract(
            requested_prompt: str,
            game_design: Mapping[str, Any],
            research_brief: Any = None,
            modules=(),
            assets=(),
            acceptance_tests=(),
            evidence_plan: Mapping[str, Any] | None = None,
        ):
            effective_plan = _validated_evidence_plan(
                evidence_plan,
                requested_prompt=requested_prompt,
            )
            effective_acceptance = _filter_evidence_input_acceptance(
                acceptance_tests,
                effective_plan,
            )
            compilation = original(
                requested_prompt,
                game_design,
                research_brief,
                modules,
                assets,
                effective_acceptance,
                effective_plan,
            )
            return _rewrite_compilation(
                compilation,
                modules=modules,
                assets=assets,
                evidence_plan=effective_plan,
            )

        compile_contract._mmm_authority_acceptance_projection = True
        compile_contract._mmm_acceptance_contract_owner = CANONICAL_ACCEPTANCE_OWNER
        _production.compile_production_contract = compile_contract
    _INSTALLED = True


__all__ = ["install_production_boundary_contract"]
