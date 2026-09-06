from __future__ import annotations

"""Production serialization boundary for stable requirement identity and public acceptance.

Evidence-mode production contracts must project the approved requirement graph rather than
rephrasing it. User-facing acceptance is copied from the requirement authority; task IDs,
anchor integrity and implementation invariants remain internal. Conditional quality coverage
is bound to the requirement that activated it instead of every requested requirement.
"""

from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
from typing import Any

from . import production_contract as _production

_INSTALLED = False
_STRICT_PUBLIC_ACCEPTANCE = _production._validate_public_acceptance
_ALLOW_VERIFIED_LEGACY_ACCEPTANCE: ContextVar[bool] = ContextVar(
    "mmm_allow_verified_legacy_acceptance", default=False
)


def _strict_public_acceptance(value: Any) -> bool:
    """Return whether ``value`` satisfies the production public-acceptance boundary."""
    if not isinstance(value, str):
        return False
    try:
        _production._validate_public_acceptance(value.strip())
    except _production.ProductionContractError:
        return False
    return True


def _canonical_public_acceptance(values: Any) -> list[str]:
    """Normalize already-authoritative acceptance without reviving retired inference."""

    if not isinstance(values, list):
        return []
    result: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or not _strict_public_acceptance(text) or text in result:
            continue
        result.append(text)
    return result


def _contextual_public_acceptance(statement: str) -> None:
    """Relax only the already-verified legacy projection seed in this context."""
    if _ALLOW_VERIFIED_LEGACY_ACCEPTANCE.get():
        if not isinstance(statement, str) or not statement.strip():
            raise _production.ProductionContractError(
                "legacy acceptance must still be a non-empty string"
            )
        return
    _STRICT_PUBLIC_ACCEPTANCE(statement)


_contextual_public_acceptance._mmm_contextual_legacy_boundary = True


def _install_planner_public_acceptance_guard() -> None:
    """Make evidence planning use the same strict public boundary as production.

    The evidence planner has additional testability checks that remain authoritative.
    This guard only tightens its result with the production leak detector so a plan
    accepted upstream cannot fail later solely because task/integrity language crossed
    the public boundary.
    """
    from . import evidence_first_planning as _evidence

    original = _evidence._is_public_acceptance
    if getattr(original, "_mmm_production_public_acceptance_guard", False):
        return

    @wraps(original)
    def is_public_acceptance(value: Any) -> bool:
        if not original(value):
            return False
        normalize = getattr(_evidence, "_normalize_public_acceptance", None)
        candidate = (
            normalize(value) if callable(normalize) else str(value or "").strip()
        )
        return _strict_public_acceptance(candidate)

    is_public_acceptance._mmm_production_public_acceptance_guard = True
    _evidence._is_public_acceptance = is_public_acceptance


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
    """Drop non-authoritative internal acceptance text before evidence-mode compilation.

    In evidence mode the canonical public contract belongs to
    ``request_catalog.requirements[*].acceptance``. Free-form input tests are only
    supplementary, so internal task/integrity prose must never be allowed to abort the
    compiler before the canonical requirement authority is projected. Outside evidence
    mode the original strict fail-closed behavior is preserved unchanged.
    """
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
        if not isinstance(value, str) or _strict_public_acceptance(value)
    )


def _approved_requirements(
    evidence_plan: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(evidence_plan, Mapping):
        return {}
    request = evidence_plan.get("request_catalog")
    values = request.get("requirements") if isinstance(request, Mapping) else None
    if not isinstance(values, list):
        return {}
    return {
        str(item.get("requirement_id")): item
        for item in values
        if isinstance(item, Mapping) and str(item.get("requirement_id") or "")
    }


def _approved_acceptance(requirement: Mapping[str, Any]) -> str:
    """Project one strict public check after the original evidence hash was verified."""
    acceptance = _canonical_public_acceptance(requirement.get("acceptance"))
    if len(acceptance) == 1:
        return acceptance[0]
    if len(acceptance) > 1:
        raise _production.ProductionContractError(
            f"approved requirement {requirement.get('requirement_id')} exposes multiple public acceptance contracts"
        )
    observable = requirement.get("observable_behavior")
    if isinstance(observable, Mapping):
        given = str(observable.get("given") or "").strip()
        when = str(observable.get("when") or "").strip()
        then = str(observable.get("then") or "").strip()
        if given and when and then:
            candidate = f"Given {given}, when {when}, then {then}."
            if _strict_public_acceptance(candidate):
                return candidate
    capability = str(requirement.get("capability") or "").strip()
    if capability:
        candidate = "Verify the observable player-facing behavior for capability " + capability + "."
        if _strict_public_acceptance(candidate):
            return candidate
    span = requirement.get("source_span")
    source_text = str(span.get("text") or "").strip() if isinstance(span, Mapping) else ""
    if source_text:
        candidate = "Demonstrate the observable requested behavior: " + source_text
        if _strict_public_acceptance(candidate):
            return candidate
    raise _production.ProductionContractError(
        f"approved requirement {requirement.get('requirement_id')} has no safe public acceptance projection"
    )


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
    approved_statement_set = set(approved_statements.values())

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
            and str(item.get("statement") or "") in approved_statement_set
        ):
            removed_refs.add(ref)
            continue
        if item.get("visibility") == "public":
            statement = str(item.get("statement") or "")
            try:
                _production._validate_public_acceptance(statement)
            except _production.ProductionContractError as exc:
                raise _production.ProductionContractError(
                    f"public acceptance leaked an internal task invariant: {ref}"
                ) from exc
            if statement in seen_public:
                raise _production.ProductionContractError(
                    f"duplicate public acceptance statement would destroy requirement traceability: {ref}"
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
    if not getattr(
        _production._validate_public_acceptance,
        "_mmm_contextual_legacy_boundary",
        False,
    ):
        _production._validate_public_acceptance = _contextual_public_acceptance

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
            legacy_token = _ALLOW_VERIFIED_LEGACY_ACCEPTANCE.set(
                isinstance(effective_plan, Mapping)
            )
            try:
                compilation = original(
                    requested_prompt,
                    game_design,
                    research_brief,
                    modules,
                    assets,
                    effective_acceptance,
                    effective_plan,
                )
            finally:
                _ALLOW_VERIFIED_LEGACY_ACCEPTANCE.reset(legacy_token)
            return _rewrite_compilation(
                compilation,
                modules=modules,
                assets=assets,
                evidence_plan=effective_plan,
            )

        compile_contract._mmm_authority_acceptance_projection = True
        _production.compile_production_contract = compile_contract
    _INSTALLED = True


__all__ = ["install_production_boundary_contract"]
