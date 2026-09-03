from __future__ import annotations

"""CodePlan-style implementation graph, artifact plan and acceptance boundary.

Requirement causality is compiled into the same consumes/provides dataflow used by the
implementation graph. There is one DAG-edge authority: ``_bind_consumes_dependencies``.
Task-local integrity checks are separated from user-facing acceptance. Architecture-derived
resources are explicit artifact obligations rather than an empty assets list or an implicit
Java-only plan.
"""

import math
import re
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from functools import wraps
from typing import Any

from . import evidence_first_planning as _planning
from . import target_grounding_contract as _target_contract

_INSTALLED = False
_REQUIREMENT_DEPS: ContextVar[dict[str, tuple[str, ...]]] = ContextVar(
    "mmm_requirement_dependency_context", default={}
)
_REQUIREMENT_PROVIDES: ContextVar[dict[str, tuple[str, ...]]] = ContextVar(
    "mmm_requirement_provides_context", default={}
)

_FLOW_RE = re.compile(
    r"\bgiven\b(?P<given>.*?)\bwhen\b(?P<when>.*?)\bthen\b(?P<then>.*?)(?:$|\n)",
    re.IGNORECASE | re.DOTALL,
)
_CAUSAL_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
        "does", "for", "from", "has", "have", "in", "into", "is", "it", "its", "of",
        "on", "or", "player", "requested", "respective", "section", "sections", "the",
        "their", "then", "to", "when", "with",
    }
)


def _causal_token_variants(value: Any) -> set[str]:
    """Return conservative lexical variants for requirement-state matching.

    This is deliberately domain-neutral.  It normalizes simple English inflection used by
    generated Given/When/Then acceptance without inventing gameplay semantics.  Ambiguous
    semantic relations remain unbound instead of being guessed.
    """

    result: set[str] = set()
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(value or "").casefold())
    for raw in raw_tokens:
        compound = raw.strip("_")
        # Capabilities and provides are commonly canonicalized as snake_case while
        # acceptance prose uses ordinary words.  Keep the compound for exact matching,
        # but also normalize each semantic component so ``module_construction`` can
        # conservatively match an authored Given-state such as ``constructed module``.
        tokens = (compound, *(part for part in compound.split("_") if part != compound))
        for token in tokens:
            if len(token) < 3 or token in _CAUSAL_STOPWORDS:
                continue
            variants = {token}
            if token.endswith("ies") and len(token) > 4:
                variants.add(token[:-3] + "y")
            elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                variants.add(token[:-1])
            if token.endswith("ing") and len(token) > 5:
                variants.add(token[:-3])
                variants.add(token[:-3] + "e")
            if token.endswith("ed") and len(token) > 4:
                variants.add(token[:-2])
                variants.add(token[:-1])
            if token.endswith("tion") and len(token) > 6:
                variants.add(token[:-3])
            result.update(item for item in variants if len(item) >= 3)
    return result


def _acceptance_flow(requirement: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Extract prerequisite and produced-state tokens from public acceptance evidence."""

    given: set[str] = set()
    produced: set[str] = set()
    for acceptance in _planning._strings(requirement.get("acceptance")):
        match = _FLOW_RE.search(acceptance.replace(";", " "))
        if match is None:
            continue
        given.update(_causal_token_variants(match.group("given")))
        produced.update(_causal_token_variants(match.group("then")))
    produced.update(_causal_token_variants(requirement.get("capability")))
    produced.update(
        token
        for provide in _planning._strings(requirement.get("provides"))
        for token in _causal_token_variants(provide)
    )
    return given, produced


def _derive_requirement_causality(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a conservative authored-order prerequisite DAG from observable states.

    CodePlan-style edit chains need requirement causality before file-level work is compiled.
    The base catalog previously emitted independent requirements, so every generic custom task
    consumed only ``target:frozen``.  Here a child Given-state may consume an earlier
    requirement only when the earlier requirement's observable Then/provide vocabulary matches
    informative prerequisite vocabulary.  Corpus-common words are down-weighted by document
    frequency, so generic terms cannot create dense false dependencies.
    """

    raw_requirements = catalog.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        return dict(catalog)
    requirements = [dict(item) for item in raw_requirements if isinstance(item, Mapping)]
    if len(requirements) != len(raw_requirements):
        return dict(catalog)

    flows = [_acceptance_flow(requirement) for requirement in requirements]
    produced_frequency: dict[str, int] = {}
    for _given, produced in flows:
        for token in produced:
            produced_frequency[token] = produced_frequency.get(token, 0) + 1
    informative_limit = max(2, int(math.ceil(len(requirements) * 0.35)))

    for child_index, requirement in enumerate(requirements):
        existing = tuple(_planning._strings(requirement.get("depends_on")))
        if existing:
            requirement["depends_on"] = list(existing)
            continue
        needed, _produced = flows[child_index]
        if not needed or child_index == 0:
            requirement["depends_on"] = []
            requirement["dependency_reasons"] = {}
            continue

        candidates: list[tuple[int, int, str, tuple[str, ...]]] = []
        for parent_index in range(child_index):
            parent_id = str(requirements[parent_index].get("requirement_id") or "").strip()
            if not parent_id:
                continue
            parent_produced = flows[parent_index][1]
            overlap = needed & parent_produced
            informative = tuple(
                sorted(
                    token
                    for token in overlap
                    if produced_frequency.get(token, 0) <= informative_limit
                )
            )
            if not informative:
                continue
            # Prefer more matched state tokens, then the closest earlier producer.  We keep
            # every tied semantic producer only when it contributes a distinct needed token.
            candidates.append((len(informative), parent_index, parent_id, informative))

        selected: list[tuple[str, tuple[str, ...]]] = []
        covered_tokens: set[str] = set()
        for _score, _index, parent_id, matched in sorted(
            candidates, key=lambda item: (-item[0], -item[1], item[2])
        ):
            novel = tuple(token for token in matched if token not in covered_tokens)
            if not novel:
                continue
            selected.append((parent_id, novel))
            covered_tokens.update(novel)

        requirement["depends_on"] = [parent_id for parent_id, _matched in selected]
        requirement["dependency_reasons"] = {
            parent_id: {
                "kind": "acceptance_precondition_dataflow",
                "matched_terms": list(matched),
            }
            for parent_id, matched in selected
        }

    result = dict(catalog)
    result["requirements"] = requirements
    result["catalog_sha256"] = ""
    result["catalog_sha256"] = _planning._hash_without(result, "catalog_sha256")
    return result


def _capture_requirement_graph(catalog: Mapping[str, Any]) -> None:
    """Capture authoritative requirement causality and its semantic dataflow tokens."""

    requirements = catalog.get("requirements")
    graph: dict[str, tuple[str, ...]] = {}
    provides: dict[str, tuple[str, ...]] = {}
    if isinstance(requirements, list):
        known = {
            str(item.get("requirement_id") or "").strip()
            for item in requirements
            if isinstance(item, Mapping) and str(item.get("requirement_id") or "").strip()
        }
        for raw in requirements:
            if not isinstance(raw, Mapping):
                continue
            req_id = str(raw.get("requirement_id") or "").strip()
            if not req_id:
                continue
            values = raw.get("depends_on")
            deps = (
                tuple(
                    str(value).strip()
                    for value in values
                    if str(value).strip()
                )
                if isinstance(values, list)
                else ()
            )
            deps = tuple(dict.fromkeys(deps))
            unknown = tuple(item for item in deps if item not in known)
            if unknown:
                raise _planning.EvidencePlanError(
                    f"Requirement {req_id} references unknown causal dependencies: {list(unknown)}"
                )
            if req_id in deps:
                raise _planning.EvidencePlanError(
                    f"Requirement {req_id} may not depend on itself."
                )
            graph[req_id] = deps
            semantic_provides = tuple(_planning._strings(raw.get("provides")))
            if not semantic_provides:
                capability = str(raw.get("capability") or "").strip()
                if capability:
                    semantic_provides = (_planning._canonical_capability(capability),)
            provides[req_id] = tuple(dict.fromkeys(semantic_provides))
    _REQUIREMENT_DEPS.set(graph)
    _REQUIREMENT_PROVIDES.set(provides)


def _normalize_ownership(
    game_design: Mapping[str, Any], ownership: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(ownership)
    raw_module = str(value.get("module_id") or ":").strip()
    value["gradle_project_path"] = (
        raw_module if raw_module == ":" or raw_module.startswith(":") else ""
    )
    value["module_id"] = _target_contract._logical_module_id(raw_module, {})
    raw_topology = value.get("topology_module_ids")
    if isinstance(raw_topology, list):
        value["topology_module_ids"] = list(
            dict.fromkeys(
                _target_contract._logical_module_id(str(item), {})
                for item in raw_topology
                if str(item).strip()
            )
        )
    return value


def _artifact_obligations(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    task_id = str(task.get("task_id") or "")
    requirement_refs = list(_planning._strings(task.get("requirement_refs")))
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    kind_map = {
        "symbol": "source_code",
        "resource": "data_or_client_resource",
        "registry_id": "registry_entry",
        "test": "verification_artifact",
        "build_config": "build_configuration",
        "loader_module": "loader_module_binding",
    }
    for raw in (
        task.get("owned_anchors", [])
        if isinstance(task.get("owned_anchors"), list)
        else []
    ):
        if not isinstance(raw, Mapping):
            continue
        anchor_kind = str(raw.get("kind") or "").strip()
        locator = str(raw.get("locator") or "").strip()
        artifact_kind = kind_map.get(anchor_kind, "implementation_artifact")
        key = (artifact_kind, locator)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "artifact_id": _planning._stable_id(
                    "artifact", artifact_kind, {"task": task_id, "locator": locator}
                ),
                "kind": artifact_kind,
                "locator": locator,
                "requirement_refs": requirement_refs,
                "task_ref": task_id,
                "status": "REQUIRED",
                "provenance_role": "implementation_obligation",
            }
        )

    predicates = set(_planning._strings(task.get("conditional_predicates")))
    extra: list[tuple[str, str]] = []
    if "needs_datagen" in predicates:
        extra.append(("generated_data_resource", "datagen output and reference closure"))
    if "needs_client_render" in predicates:
        extra.append(
            ("client_visual_or_ui_resource", "client model/texture/UI resource contract")
        )
    if "needs_worldgen" in predicates:
        extra.append(
            ("worldgen_data", "world-generation configured/placed/binding data")
        )
    if "needs_persistence" in predicates:
        extra.append(
            ("persistence_schema", "serialized state schema and compatibility contract")
        )
    if "needs_network" in predicates:
        extra.append(("network_protocol", "payload/codec/validation contract"))
    for artifact_kind, purpose in extra:
        key = (artifact_kind, purpose)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "artifact_id": _planning._stable_id(
                    "artifact", artifact_kind, {"task": task_id, "purpose": purpose}
                ),
                "kind": artifact_kind,
                "locator": "unresolved:" + artifact_kind,
                "requirement_refs": requirement_refs,
                "task_ref": task_id,
                "status": "REQUIRED_UNRESOLVED",
                "provenance_role": "implementation_obligation",
                "purpose": purpose,
            }
        )
    return result


def _task_roots(tasks: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    by_id = {str(task.get("task_id") or ""): task for task in tasks}
    for task in tasks:
        for ref in _planning._strings(task.get("requirement_refs")):
            groups.setdefault(ref, []).append(str(task.get("task_id") or ""))
    roots: dict[str, list[str]] = {}
    for req, ids in groups.items():
        id_set = set(ids)
        roots[req] = [
            task_id
            for task_id in ids
            if not any(
                dep in id_set
                for dep in _planning._strings(by_id[task_id].get("depends_on"))
            )
        ]
    return roots


def _project_requirement_dataflow(
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Project requirement causality into consumes, then let one binder own DAG edges.

    A requirement with implementation tasks exposes its approved semantic ``provides``
    through those tasks. A retained requirement has no task and therefore its approved
    provides are host-root inputs. Child requirement roots consume the parent's approved
    provides. The ordinary dataflow binder then creates exactly the dependency edges that
    the validator independently reconstructs; no second edge compiler exists.
    """

    result = [dict(task) for task in tasks]
    req_deps = dict(_REQUIREMENT_DEPS.get())
    req_provides = dict(_REQUIREMENT_PROVIDES.get())
    roots = _task_roots(result)
    task_requirements = {
        ref
        for task in result
        for ref in _planning._strings(task.get("requirement_refs"))
    }
    root_provides = {"target:frozen"}
    for req_id, provided in req_provides.items():
        if req_id not in task_requirements:
            root_provides.update(provided)

    by_id = {str(task.get("task_id") or ""): task for task in result}
    for req_id, parents in req_deps.items():
        for root_id in roots.get(req_id, ()):
            task = by_id[root_id]
            consumes = list(_planning._strings(task.get("consumes")))
            for parent_req in parents:
                parent_tokens = req_provides.get(parent_req, ())
                if not parent_tokens:
                    raise _planning.EvidencePlanError(
                        f"Requirement {req_id} depends on {parent_req} without an approved provide token."
                    )
                for token in parent_tokens:
                    if token not in consumes:
                        consumes.append(token)
            task["consumes"] = consumes
            task["task_sha256"] = ""
            task["task_sha256"] = _planning._hash_without(task, "task_sha256")

    bound = [
        dict(task)
        for task in _planning._bind_consumes_dependencies(
            result,
            root_provides=root_provides,
        )
    ]
    provider_requirement = {
        str(task.get("task_id") or ""): next(
            iter(_planning._strings(task.get("requirement_refs"))), ""
        )
        for task in bound
    }
    for task in bound:
        req_refs = tuple(_planning._strings(task.get("requirement_refs")))
        req_id = req_refs[0] if req_refs else ""
        parent_requirements = set(req_deps.get(req_id, ()))
        reasons: dict[str, dict[str, str]] = {}
        for dep in _planning._strings(task.get("depends_on")):
            parent_req = provider_requirement.get(dep, "")
            if parent_req in parent_requirements:
                reasons[dep] = {
                    "kind": "requirement_dataflow",
                    "requirement_ref": parent_req,
                }
            else:
                reasons[dep] = {"kind": "implementation_dataflow"}
        task["dependency_reasons"] = reasons
        task["task_sha256"] = ""
        task["task_sha256"] = _planning._hash_without(task, "task_sha256")
    _planning._topological(bound)
    return tuple(bound)


def _postprocess_tasks(
    tasks: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    gap_by_req = {
        str(gap.get("requirement_ref") or ""): gap
        for gap in gaps
        if isinstance(gap, Mapping)
    }
    result: list[dict[str, Any]] = []
    for raw in tasks:
        task = dict(raw)
        task_id = str(task.get("task_id") or "")
        req_refs = list(_planning._strings(task.get("requirement_refs")))
        req = req_refs[0] if req_refs else ""
        legacy_acceptance = list(_planning._strings(task.get("acceptance")))
        public = [
            item for item in legacy_acceptance if _planning._is_public_acceptance(item)
        ]
        internal = [item for item in legacy_acceptance if item not in public]
        if not internal:
            internal = [
                f"{task_id}: declared provides, owned anchors, hashes, and required gates are internally consistent"
            ]
        task["internal_invariants"] = internal
        task["public_acceptance"] = public
        task["acceptance"] = internal
        gap = gap_by_req.get(req, {})
        gap_acceptance = list(_planning._strings(gap.get("acceptance")))
        if (
            str(task.get("semantic_outcome") or "").startswith(
                "Implement one independently verifiable outcome for "
            )
            and gap_acceptance
        ):
            task["semantic_outcome"] = (
                f"Satisfy requirement {req} for {gap.get('capability')}: {gap_acceptance[0]}"
            )
        task["artifact_obligations"] = _artifact_obligations(task)
        task["impact_domains"] = list(
            dict.fromkeys(
                [
                    "source"
                    if item.get("kind") == "source_code"
                    else "resources"
                    if "resource" in str(item.get("kind"))
                    or "worldgen" in str(item.get("kind"))
                    else "state"
                    if item.get("kind") == "persistence_schema"
                    else "network"
                    if item.get("kind") == "network_protocol"
                    else "build"
                    if "build" in str(item.get("kind"))
                    or "loader" in str(item.get("kind"))
                    else "verification"
                    if item.get("kind") == "verification_artifact"
                    else "registry"
                    for item in task["artifact_obligations"]
                ]
            )
        )
        task["task_sha256"] = ""
        task["task_sha256"] = _planning._hash_without(task, "task_sha256")
        result.append(task)

    return _project_requirement_dataflow(result)


def _artifact_plan(
    plan: Mapping[str, Any], game_design: Mapping[str, Any]
) -> dict[str, Any]:
    required: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in (
        plan.get("tasks", []) if isinstance(plan.get("tasks"), list) else []
    ):
        if not isinstance(task, Mapping):
            continue
        for artifact in (
            task.get("artifact_obligations", [])
            if isinstance(task.get("artifact_obligations"), list)
            else []
        ):
            if not isinstance(artifact, Mapping):
                continue
            artifact_id = str(artifact.get("artifact_id") or "")
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            required.append(dict(artifact))

    supplied_assets = (
        [dict(item) for item in game_design.get("assets", []) if isinstance(item, Mapping)]
        if isinstance(game_design.get("assets"), list)
        else []
    )
    resource_required = [
        item
        for item in required
        if item.get("kind")
        in {
            "data_or_client_resource",
            "generated_data_resource",
            "client_visual_or_ui_resource",
            "worldgen_data",
        }
    ]
    return {
        "schema_version": "mmm/artifact-plan-v1",
        "required_artifacts": required,
        "supplied_asset_briefs": supplied_assets,
        "asset_requirement_status": (
            "REQUIRED_AND_SUPPLIED"
            if resource_required and supplied_assets
            else "REQUIRED_UNRESOLVED"
            if resource_required
            else "NOT_REQUIRED_BY_ARCHITECTURE"
        ),
        "zero_asset_justification": (
            "No architecture-derived client/data visual resource obligation exists."
            if not resource_required and not supplied_assets
            else ""
        ),
    }


def _design_resolution(plan: Mapping[str, Any]) -> dict[str, Any]:
    alternatives: list[dict[str, Any]] = []
    obligations: list[dict[str, Any]] = []
    seen_alt: set[tuple[str, str]] = set()
    for task in (
        plan.get("tasks", []) if isinstance(plan.get("tasks"), list) else []
    ):
        if not isinstance(task, Mapping):
            continue
        refs = list(_planning._strings(task.get("requirement_refs")))
        predicates = set(_planning._strings(task.get("conditional_predicates")))
        outcome = str(task.get("semantic_outcome") or "")
        if "needs_client_render" in predicates and (
            "screen" in outcome.casefold() or "menu" in outcome.casefold()
        ):
            key = (refs[0] if refs else "", "menu_screen_contract")
            if key not in seen_alt:
                seen_alt.add(key)
                alternatives.append(
                    {
                        "decision_id": _planning._stable_id(
                            "design", "menu_screen_contract", key
                        ),
                        "provenance_role": "selected_design_alternative",
                        "requirement_refs": refs,
                        "selection": "menu_screen_contract",
                        "reason": "client-render architecture branch requires an interactive presentation contract",
                    }
                )
        for artifact in (
            task.get("artifact_obligations", [])
            if isinstance(task.get("artifact_obligations"), list)
            else []
        ):
            if isinstance(artifact, Mapping):
                obligations.append(
                    {
                        "obligation_id": artifact.get("artifact_id"),
                        "provenance_role": "implementation_obligation",
                        "requirement_refs": list(
                            artifact.get("requirement_refs") or refs
                        ),
                        "kind": artifact.get("kind"),
                        "reason": "required by the selected task architecture",
                    }
                )
    return {
        "schema_version": "mmm/design-resolution-v1",
        "selected_design_alternatives": alternatives,
        "implementation_obligations": obligations,
        "policy": (
            "Authored requirements remain authoritative gameplay goals. Design alternatives and "
            "implementation obligations are separate provenance classes and never become authored mandates."
        ),
    }


def _acceptance_boundary(plan: Mapping[str, Any]) -> dict[str, Any]:
    public = []
    request = plan.get("request_catalog")
    requirements = (
        request.get("requirements", []) if isinstance(request, Mapping) else []
    )
    for req in requirements if isinstance(requirements, list) else []:
        if not isinstance(req, Mapping):
            continue
        public.append(
            {
                "requirement_ref": req.get("requirement_id"),
                "capability": req.get("capability"),
                "acceptance": list(_planning._strings(req.get("acceptance"))),
            }
        )
    internal = []
    for task in (
        plan.get("tasks", []) if isinstance(plan.get("tasks"), list) else []
    ):
        if not isinstance(task, Mapping):
            continue
        internal.append(
            {
                "task_ref": task.get("task_id"),
                "checks": list(_planning._strings(task.get("internal_invariants"))),
            }
        )
    return {
        "schema_version": "mmm/acceptance-boundary-v1",
        "public_acceptance": public,
        "internal_invariants": internal,
    }


def install_task_artifact_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    current_catalog = _planning.build_request_catalog
    if not getattr(current_catalog, "_mmm_capture_requirement_causality", False):

        @wraps(current_catalog)
        def build_request_catalog(*args: Any, **kwargs: Any):
            catalog = current_catalog(*args, **kwargs)
            if isinstance(catalog, Mapping):
                catalog = _derive_requirement_causality(catalog)
                _capture_requirement_graph(catalog)
            return catalog

        build_request_catalog._mmm_capture_requirement_causality = True
        _planning.build_request_catalog = build_request_catalog

    current_ownership = _planning._ownership_context
    if not getattr(current_ownership, "_mmm_logical_module_identity", False):

        @wraps(current_ownership)
        def ownership(game_design: Mapping[str, Any]):
            return _normalize_ownership(game_design, current_ownership(game_design))

        ownership._mmm_logical_module_identity = True
        _planning._ownership_context = ownership

    current_tasks = _planning._compile_tasks
    if not getattr(current_tasks, "_mmm_codeplan_task_graph", False):

        @wraps(current_tasks)
        def compile_tasks(gaps, reuse, target, branches, ownership):
            return _postprocess_tasks(
                current_tasks(gaps, reuse, target, branches, ownership), gaps
            )

        compile_tasks._mmm_codeplan_task_graph = True
        _planning._compile_tasks = compile_tasks

    current_validate = _planning.validate_evidence_first_plan
    if not getattr(current_validate, "_mmm_requirement_context_validation", False):

        @wraps(current_validate)
        def validate(plan: Mapping[str, Any], *, prompt: str | None = None):
            request = plan.get("request_catalog")
            previous_deps = _REQUIREMENT_DEPS.get()
            previous_provides = _REQUIREMENT_PROVIDES.get()
            captured = False
            if isinstance(request, Mapping):
                _capture_requirement_graph(request)
                captured = True
            try:
                return current_validate(plan, prompt=prompt)
            finally:
                if captured:
                    _REQUIREMENT_DEPS.set(previous_deps)
                    _REQUIREMENT_PROVIDES.set(previous_provides)

        validate._mmm_requirement_context_validation = True
        _planning.validate_evidence_first_plan = validate

    current_compile = _planning.compile_evidence_first_plan
    if not getattr(current_compile, "_mmm_artifact_acceptance_boundary", False):

        @wraps(current_compile)
        def compile_plan(
            prompt: str, game_design: Mapping[str, Any], **kwargs: Any
        ):
            plan = dict(current_compile(prompt, game_design, **kwargs))
            plan["artifact_plan"] = _artifact_plan(plan, game_design)
            plan["design_resolution"] = _design_resolution(plan)
            plan["acceptance_boundary"] = _acceptance_boundary(plan)
            plan["plan_sha256"] = ""
            plan["plan_sha256"] = _planning._hash_without(plan, "plan_sha256")
            _planning.validate_evidence_first_plan(plan, prompt=prompt)
            return plan

        compile_plan._mmm_artifact_acceptance_boundary = True
        _planning.compile_evidence_first_plan = compile_plan

    _INSTALLED = True


__all__ = ["install_task_artifact_contract"]
