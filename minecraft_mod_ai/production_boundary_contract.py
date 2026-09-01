from __future__ import annotations

"""Production serialization boundary for stable requirement identity and public acceptance.

Evidence-mode production contracts must project the approved requirement graph rather than
rephrasing it. User-facing acceptance is copied from the requirement authority; task IDs,
anchor integrity and implementation invariants remain internal. Conditional quality coverage
is bound to the requirement that activated it instead of every requested requirement.
"""

from collections.abc import Mapping
from functools import wraps
from typing import Any

from . import production_contract as _production

_INSTALLED = False


def _strict_public_acceptance(value: Any) -> bool:
    """Return whether ``value`` satisfies the production public-acceptance boundary."""
    if not isinstance(value, str):
        return False
    try:
        _production._validate_public_acceptance(value.strip())
    except _production.ProductionContractError:
        return False
    return True


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
        candidate = normalize(value) if callable(normalize) else str(value or "").strip()
        return _strict_public_acceptance(candidate)

    is_public_acceptance._mmm_production_public_acceptance_guard = True
    _evidence._is_public_acceptance = is_public_acceptance


def _migrate_verified_evidence_public_acceptance(
    evidence_plan: Mapping[str, Any] | None,
    *,
    requested_prompt: str,
) -> Mapping[str, Any] | None:
    """Migrate legacy public acceptance only after verifying the original plan."""
    if not isinstance(evidence_plan, Mapping):
        return evidence_plan

    from . import evidence_first_planning as _evidence

    _evidence.validate_evidence_first_plan(evidence_plan, prompt=requested_prompt)
    request = evidence_plan.get("request_catalog")
    if not isinstance(request, Mapping):
        return evidence_plan
    raw_requirements = request.get("requirements")
    if not isinstance(raw_requirements, list):
        return evidence_plan

    canonical_by_ref: dict[str, list[str]] = {}
    migrated_requirements: list[Any] = []
    request_changed = False
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            migrated_requirements.append(raw)
            continue
        requirement = dict(raw)
        current_raw = requirement.get("acceptance")
        current = list(current_raw) if isinstance(current_raw, list) else []
        canonical = list(
            _evidence._requirement_acceptance(
                str(requirement.get("capability") or ""),
                current,
            )
        )
        canonical = [value for value in canonical if _strict_public_acceptance(value)]
        if not canonical:
            canonical = [
                "Verify the observable player-facing behavior for the approved requirement."
            ]
        requirement_ref = str(requirement.get("requirement_id") or "")
        if requirement_ref:
            canonical_by_ref[requirement_ref] = canonical
        if canonical != current:
            requirement["acceptance"] = canonical
            request_changed = True
        migrated_requirements.append(requirement)

    migrated_request: Mapping[str, Any] = request
    migrated_gaps: Any = evidence_plan.get("gap_catalog")
    migrated_tasks: Any = evidence_plan.get("tasks")
    if request_changed:
        request_copy = dict(request)
        request_copy["requirements"] = migrated_requirements
        request_copy["catalog_sha256"] = ""
        request_copy["catalog_sha256"] = _evidence._hash_without(
            request_copy,
            "catalog_sha256",
        )
        migrated_request = request_copy

        raw_gaps = evidence_plan.get("gap_catalog")
        if not isinstance(raw_gaps, list):
            raise _evidence.EvidencePlanError("Gap catalog must be a list.")
        gap_values: list[Any] = []
        for raw in raw_gaps:
            if not isinstance(raw, Mapping):
                gap_values.append(raw)
                continue
            gap = dict(raw)
            requirement_ref = str(gap.get("requirement_ref") or "")
            canonical = canonical_by_ref.get(requirement_ref)
            if canonical is not None:
                gap["acceptance"] = list(canonical)
                gap["gap_sha256"] = ""
                gap["gap_sha256"] = _evidence._hash_without(gap, "gap_sha256")
            gap_values.append(gap)
        migrated_gaps = gap_values

        rebuilt_tasks = _evidence._compile_tasks(
            migrated_gaps,
            evidence_plan.get("reuse_decisions") or (),
            evidence_plan.get("target_decision") or {},
            evidence_plan.get("branch_predicates") or {},
            evidence_plan.get("ownership_context") or {},
        )
        order = _evidence._topological(rebuilt_tasks)
        tasks_by_id = {str(task["task_id"]): task for task in rebuilt_tasks}
        migrated_tasks = [tasks_by_id[task_id] for task_id in order]

    raw_bindings = evidence_plan.get("acceptance_release_bindings")
    migrated_bindings: Any = raw_bindings
    bindings_changed = False
    if isinstance(raw_bindings, list):
        binding_values: list[Any] = []
        for raw in raw_bindings:
            if not isinstance(raw, Mapping):
                binding_values.append(raw)
                continue
            binding = dict(raw)
            requirement_ref = str(binding.get("requirement_ref") or "")
            canonical = canonical_by_ref.get(requirement_ref)
            current_raw = binding.get("acceptance")
            current = list(current_raw) if isinstance(current_raw, list) else []
            if canonical is not None and canonical != current:
                binding["acceptance"] = list(canonical)
                bindings_changed = True
            binding_values.append(binding)
        if bindings_changed:
            migrated_bindings = binding_values

    if not request_changed and not bindings_changed:
        return evidence_plan

    migrated_plan = dict(evidence_plan)
    if request_changed:
        migrated_plan["request_catalog"] = migrated_request
        migrated_plan["gap_catalog"] = migrated_gaps
        migrated_plan["tasks"] = migrated_tasks
    if bindings_changed:
        migrated_plan["acceptance_release_bindings"] = migrated_bindings
    boundary = migrated_plan.get("acceptance_boundary")
    if request_changed and isinstance(boundary, Mapping):
        boundary_copy = dict(boundary)
        boundary_copy["public_acceptance"] = [
            {
                "requirement_ref": requirement.get("requirement_id"),
                "capability": requirement.get("capability"),
                "acceptance": list(requirement.get("acceptance") or ()),
            }
            for requirement in migrated_requirements
            if isinstance(requirement, Mapping)
        ]
        migrated_plan["acceptance_boundary"] = boundary_copy
    migrated_plan["plan_sha256"] = ""
    migrated_plan["plan_sha256"] = _evidence._hash_without(
        migrated_plan,
        "plan_sha256",
    )
    _evidence.validate_evidence_first_plan(migrated_plan, prompt=requested_prompt)
    return migrated_plan


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


def _approved_requirements(evidence_plan: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
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
    values = requirement.get("acceptance")
    acceptance = [str(value).strip() for value in values if str(value).strip()] if isinstance(values, list) else []
    if len(acceptance) != 1:
        raise _production.ProductionContractError(
            f"approved requirement {requirement.get('requirement_id')} must expose exactly one canonical public acceptance contract"
        )
    statement = acceptance[0]
    try:
        _production._validate_public_acceptance(statement)
    except _production.ProductionContractError as exc:
        raise _production.ProductionContractError(
            f"approved public acceptance contains internal task/integrity language: {requirement.get('requirement_id')}"
        ) from exc
    return statement


def _requirement_context(evidence_plan: Mapping[str, Any], requirement_id: str) -> tuple[str, set[str], set[str]]:
    approved = _approved_requirements(evidence_plan).get(requirement_id, {})
    span = approved.get("source_span") if isinstance(approved, Mapping) else {}
    text = " ".join(
        str(value or "")
        for value in (
            approved.get("capability") if isinstance(approved, Mapping) else "",
            approved.get("semantic_statement") if isinstance(approved, Mapping) else "",
            span.get("text") if isinstance(span, Mapping) else "",
            " ".join(approved.get("acceptance", [])) if isinstance(approved, Mapping) and isinstance(approved.get("acceptance"), list) else "",
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
            if not isinstance(refs, list) or requirement_id not in {str(value) for value in refs}:
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
    text, predicates, artifact_kinds = _requirement_context(evidence_plan, requirement_id)
    selected: list[str] = []
    for dimension_id in _production._CONDITIONAL_ORDER:
        if dimension_id not in active_ids:
            continue
        triggered = _production._text_triggers_dimension(text, dimension_id)
        if dimension_id == "visual_3d":
            triggered = triggered or "needs_client_render" in predicates or bool(
                artifact_kinds
                & {
                    "client_visual_or_ui_resource",
                    "data_or_client_resource",
                }
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
    approved_statements = {req_id: _approved_acceptance(req) for req_id, req in approved.items()}
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
        elif origin == "input" and str(item.get("statement") or "") in approved_statement_set:
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
            effective_plan = _migrate_verified_evidence_public_acceptance(
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
        _production.compile_production_contract = compile_contract
    _INSTALLED = True


__all__ = ["install_production_boundary_contract"]
