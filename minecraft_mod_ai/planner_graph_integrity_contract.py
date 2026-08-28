from __future__ import annotations

"""Final planner-scope integrity contract.

The authored request is a graph, not a fixed-size list of model-produced modules.  This
contract keeps semantic extraction unbounded by arbitrary item counts, makes every
semantic clause its own decomposition scope, supplements model output with every
host-recognized authored capability, recursively materializes required ontology
capabilities, and binds those cross-system requirements into the implementation DAG.

The model is allowed to interpret language.  It never owns graph cardinality, source
provenance, dependency closure, task identities, or JSON planner pages.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from . import evidence_first_planning as _evidence
from . import evidence_request_guard as _guard
from . import semantic_requirement_authority as _semantic
from .canonical_capability_ontology import (
    atomic_capability_definitions,
    resolve_capabilities_from_phrase_structured,
)

_INSTALLED = False


def _clean_line_field(value: Any) -> str:
    return " ".join(str(value or "").replace("\t", " ").split())


def _plain_semantic_call(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
    *,
    repair_diagnostics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Ask for leaf requirements as plain text, with no JSON array or item ceiling."""

    system = (
        "Decompose the supplied Minecraft-mod clause into every independent, smallest "
        "player-visible behavior that must be implemented. Never stop at an arbitrary "
        "number of requirements and never group unrelated mechanics merely because they "
        "occur in one sentence. Preserve the clause index and use one meaningful lower-case "
        "dotted capability id per leaf behavior. Do not choose APIs, classes, files, loader "
        "details, persistence/network implementations, or other technical design unless the "
        "user explicitly requested that behavior.\n\n"
        "Return plain text only, one requirement per physical line, using actual TAB "
        "characters between exactly these eight fields:\n"
        "REQ<TAB>clause_index<TAB>capability_id<TAB>source_anchor<TAB>semantic_statement"
        "<TAB>given<TAB>when<TAB>then\n"
        "Do not return JSON, Markdown, headings, bullets, counts, or commentary. Replace any "
        "tabs/newlines inside a field with spaces. Emit as many REQ lines as the clause needs."
    )
    user_lines = [
        f"CLAUSE\t{int(clause['clause_index'])}\t{_clean_line_field(clause['text'])}"
        for clause in clauses
    ]
    if repair_diagnostics:
        user_lines.append("REPAIR_NOTES")
        for diagnostic in repair_diagnostics:
            user_lines.append(
                "DIAGNOSTIC\t"
                + _clean_line_field(diagnostic.get("error_code"))
                + "\t"
                + _clean_line_field(diagnostic.get("expected_contract"))
            )
    raw = router.generate_text(
        "planner",
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_lines)},
        ],
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    if not isinstance(raw, str):
        raw = str(raw)

    requirements: list[dict[str, Any]] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        while line.startswith(("- ", "* ")):
            line = line[2:].lstrip()
        if line.startswith("REQ\t"):
            parts = line.split("\t", 7)
        elif line.startswith("REQ|"):
            # Tolerate one common small-model formatting drift without introducing a
            # structured-repair round trip. maxsplit preserves pipes inside the last field.
            parts = line.split("|", 7)
        else:
            continue
        if len(parts) != 8:
            continue
        try:
            clause_index = int(parts[1].strip())
        except ValueError:
            continue
        fields = [_clean_line_field(item) for item in parts[2:]]
        capability_id, source_anchor, statement, given, when, then = fields
        if not all((capability_id, source_anchor, statement, given, when, then)):
            continue
        requirements.append(
            {
                "source_clause_index": clause_index,
                "capability_id": capability_id,
                "source_anchor": source_anchor,
                "semantic_statement": statement,
                "given": given,
                "when": when,
                "then": then,
            }
        )
    return {"requirements": requirements}


def _host_grounding_for_resolution_node(
    clause: Mapping[str, Any],
    source_anchor: str,
) -> dict[str, Any]:
    """Ground an ontology-recognized authored token without model-owned offsets."""

    text = str(clause["text"])
    anchor = str(source_anchor or "").strip()
    first = text.find(anchor) if anchor else -1
    if first >= 0:
        absolute_start = int(clause["char_start"]) + first
        quote = text[first : first + len(anchor)]
        return {
            "source_quote": quote,
            "source_start": absolute_start,
            "source_end": absolute_start + len(quote),
            "grounding_method": "host_ontology_exact",
            "grounding_similarity": 1.0,
            "model_anchor": anchor,
        }
    return {
        "source_quote": text,
        "source_start": int(clause["char_start"]),
        "source_end": int(clause["char_end"]),
        "grounding_method": "host_clause_scope",
        "grounding_similarity": 1.0,
        "model_anchor": anchor or text,
    }


def _supplement_host_recognized_nodes(
    nodes: Sequence[Mapping[str, Any]],
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prevent the model from silently omitting an authored capability the host knows."""

    definitions = atomic_capability_definitions()
    output = [dict(item) for item in nodes]
    seen = {
        (int(item["source_clause_index"]), str(item["capability_id"]).casefold())
        for item in output
    }
    for clause in clauses:
        clause_index = int(clause["clause_index"])
        resolution = resolve_capabilities_from_phrase_structured(str(clause["text"]))
        for node in resolution.nodes:
            capability = str(node.capability_id).strip().casefold()
            if (
                not capability
                or capability.startswith(("unresolved:", "provisional:"))
                or node.origin not in {"explicit", "archetype_inferred"}
                or (clause_index, capability) in seen
            ):
                continue
            grounding = _host_grounding_for_resolution_node(clause, node.source_span)
            definition = definitions.get(capability)
            description = (
                definition.description
                if definition is not None
                else f"Implement the authored {capability} behavior."
            )
            output.append(
                {
                    "capability_id": capability,
                    "provenance_role": "explicit",
                    "source_clause_index": clause_index,
                    **grounding,
                    "semantic_statement": description,
                    "derived_from": [],
                    "depends_on": [],
                    "derivation_reason": "host-recognized authored capability omitted by semantic model",
                    "observable_behavior": {
                        "given": f"the authored request includes {capability}",
                        "when": "the corresponding gameplay behavior is exercised",
                        "then": f"the requested {capability} behavior is observable and complete",
                    },
                }
            )
            seen.add((clause_index, capability))
    return output


def _unbounded_generate_approved_nodes(
    prompt: str,
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Decompose each authored clause independently; cardinality is model-output unbounded."""

    del prompt
    accepted: list[dict[str, Any]] = []
    for clause in clauses:
        scope = (clause,)
        try:
            payload = _plain_semantic_call(router, scope)
        except Exception as exc:
            diagnostics = [
                _semantic._diagnostic(
                    "REQ_MODEL_RESPONSE",
                    "$",
                    f"{type(exc).__name__}: {exc}",
                    "plain-text leaf requirement lines for this authored clause",
                    f"clause:{int(clause['clause_index'])}",
                )
            ]
            first_nodes: list[dict[str, Any]] = []
            invalid = {int(clause["clause_index"])}
        else:
            first_nodes, invalid, diagnostics = _semantic._evaluate_batch(payload, scope)
        if invalid:
            try:
                repair_payload = _plain_semantic_call(
                    router,
                    scope,
                    repair_diagnostics=diagnostics,
                )
            except Exception as exc:
                raise _evidence.EvidencePlanError(
                    "semantic leaf decomposition repair failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            repair_nodes, remaining, repair_diagnostics = _semantic._evaluate_batch(
                repair_payload,
                scope,
            )
            if remaining:
                raise _evidence.EvidencePlanError(
                    "semantic leaf decomposition could not cover an authored clause: "
                    + _semantic._canonical(
                        {
                            "clause_index": int(clause["clause_index"]),
                            "diagnostics": repair_diagnostics,
                        }
                    )
                )
            accepted.extend(repair_nodes)
        else:
            accepted.extend(first_nodes)

    accepted = _supplement_host_recognized_nodes(accepted, clauses)
    # Deduplicate exact semantic leaves but never collapse distinct behaviors merely
    # because they came from the same source clause.
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for item in accepted:
        key = (
            int(item["source_clause_index"]),
            str(item["capability_id"]).casefold(),
            str(item["semantic_statement"]).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(dict(item))
    return _semantic._assign_local_ids(deduplicated)


def _would_create_requirement_cycle(
    dependencies: Mapping[str, Sequence[str]],
    consumer: str,
    provider: str,
) -> bool:
    if consumer == provider:
        return True
    stack = [provider]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == consumer:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(str(item) for item in dependencies.get(current, ()))
    return False


def _expand_catalog_dependency_closure(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize ontology-required systems recursively until a fixed point."""

    raw_requirements = catalog.get("requirements")
    if not isinstance(raw_requirements, list):
        return dict(catalog)
    definitions = atomic_capability_definitions()
    requirements = [dict(item) for item in raw_requirements if isinstance(item, Mapping)]
    if not requirements:
        return dict(catalog)

    by_id = {str(item["requirement_id"]): item for item in requirements}
    first_by_capability: dict[str, str] = {}
    for item in requirements:
        capability = str(item.get("capability") or "").strip().casefold()
        if capability:
            first_by_capability.setdefault(capability, str(item["requirement_id"]))

    dependencies: dict[str, list[str]] = {
        requirement_id: [str(value) for value in item.get("depends_on", ()) if str(value)]
        for requirement_id, item in by_id.items()
    }
    queue = [str(item["requirement_id"]) for item in requirements]
    cursor = 0
    while cursor < len(queue):
        requirement_id = queue[cursor]
        cursor += 1
        requirement = by_id[requirement_id]
        capability = str(requirement.get("capability") or "").strip().casefold()
        definition = definitions.get(capability)
        if definition is None:
            continue
        for dependency_capability in definition.default_dependencies:
            dep = str(dependency_capability).strip().casefold()
            if not dep or dep == capability:
                continue
            dependency_id = first_by_capability.get(dep)
            if dependency_id is None:
                source_span = dict(requirement.get("source_span") or {})
                dependency_id = _evidence._stable_id(
                    "req",
                    dep,
                    {
                        "prompt_sha256": catalog.get("prompt_sha256"),
                        "ontology_dependency": dep,
                    },
                )
                dep_definition = definitions.get(dep)
                semantic_statement = (
                    dep_definition.description
                    if dep_definition is not None
                    else f"Provide required capability {dep}."
                )
                behavior = {
                    "given": f"a requested behavior requires {dep}",
                    "when": "the parent gameplay behavior is executed",
                    "then": f"{dep} is available and the parent behavior completes correctly",
                }
                derived = {
                    "requirement_id": dependency_id,
                    "capability": dep,
                    "statement": str(requirement.get("statement") or dep),
                    "semantic_statement": semantic_statement,
                    "mandatory": True,
                    "provenance_role": "logically_derived",
                    "source_span": source_span,
                    "derived_from": [requirement_id],
                    "depends_on": [],
                    "provides": [_evidence._canonical_capability(dep)],
                    "gameplay_capabilities": [dep],
                    "implementation_capabilities": [],
                    "artifact_task_ids": [],
                    "semantic_status": "RESOLVED",
                    "unresolved_spans": [],
                    "acceptance": [
                        f"Given {behavior['given']}; when {behavior['when']}; then {behavior['then']}."
                    ],
                    "observable_behavior": behavior,
                    "derivation_reason": f"ontology dependency required by {capability}",
                }
                requirements.append(derived)
                by_id[dependency_id] = derived
                first_by_capability[dep] = dependency_id
                dependencies[dependency_id] = []
                queue.append(dependency_id)
            else:
                derived = by_id[dependency_id]
                if derived.get("provenance_role") == "logically_derived":
                    parents = [str(value) for value in derived.get("derived_from", ()) if str(value)]
                    if requirement_id not in parents:
                        parents.append(requirement_id)
                        derived["derived_from"] = parents

            if dependency_id in dependencies.setdefault(requirement_id, []):
                continue
            if _would_create_requirement_cycle(
                dependencies,
                requirement_id,
                dependency_id,
            ):
                # Mutual semantic requirements still both exist.  A back-edge is omitted
                # only to keep the executable plan a DAG; neither capability is dropped.
                continue
            dependencies[requirement_id].append(dependency_id)
            requirement["depends_on"] = list(dependencies[requirement_id])

    graph_edges: list[dict[str, str]] = []
    for item in requirements:
        requirement_id = str(item["requirement_id"])
        item["depends_on"] = list(dict.fromkeys(dependencies.get(requirement_id, ())))
        for dependency_id in item["depends_on"]:
            graph_edges.append(
                {
                    "from": requirement_id,
                    "to": dependency_id,
                    "kind": "depends_on",
                }
            )

    expanded = dict(catalog)
    expanded["requirements"] = requirements
    expanded["requirement_graph"] = {
        "node_ids": [str(item["requirement_id"]) for item in requirements],
        "edges": graph_edges,
    }
    audit = dict(expanded.get("semantic_audit") or {})
    audit.update(
        {
            "graph_cardinality_policy": "unbounded_by_semantic_item_count",
            "decomposition_scope": "per_authored_clause_to_leaf_behaviors",
            "dependency_expansion": "recursive_ontology_closure_until_fixed_point",
            "approved_requirement_count": len(requirements),
        }
    )
    expanded["semantic_audit"] = audit
    expanded["catalog_sha256"] = ""
    expanded["catalog_sha256"] = _evidence._hash_without(expanded, "catalog_sha256")
    return expanded


def _build_catalog_with_dependency_closure(
    prompt: str,
    nodes: Sequence[Mapping[str, Any]],
    clauses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original = _build_catalog_with_dependency_closure.__wrapped__
    catalog = original(prompt, nodes, clauses)
    return _expand_catalog_dependency_closure(catalog)


def _task_capability(gap: Mapping[str, Any]) -> str:
    return str(gap.get("capability") or "").strip().casefold().removeprefix("capability:")


def _would_create_capability_cycle(
    edges: Mapping[str, Sequence[str]],
    consumer: str,
    provider: str,
) -> bool:
    if consumer == provider:
        return True
    stack = [provider]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == consumer:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(str(item) for item in edges.get(current, ()))
    return False


def _compile_tasks_with_cross_system_dependencies(
    gaps: Sequence[Mapping[str, Any]],
    reuse: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    ownership: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    original = _compile_tasks_with_cross_system_dependencies.__wrapped__
    tasks = original(gaps, reuse, target, branches, ownership)
    if not tasks or not gaps:
        return tasks

    definitions = atomic_capability_definitions()
    gap_by_capability: dict[str, Mapping[str, Any]] = {}
    first_task_by_gap: dict[str, str] = {}
    required_provide_by_capability: dict[str, str] = {}
    for gap in gaps:
        capability = _task_capability(gap)
        if not capability or capability in gap_by_capability:
            continue
        gap_by_capability[capability] = gap
        missing = [str(item) for item in gap.get("missing_provides", ()) if str(item)]
        if missing:
            required_provide_by_capability[capability] = missing[0]
    for task in tasks:
        for gap_ref in task.get("gap_refs", ()):
            first_task_by_gap.setdefault(str(gap_ref), str(task["task_id"]))

    accepted_edges: dict[str, list[str]] = {capability: [] for capability in gap_by_capability}
    mutable = [dict(task) for task in tasks]
    task_by_id = {str(task["task_id"]): task for task in mutable}

    for capability, gap in gap_by_capability.items():
        definition = definitions.get(capability)
        if definition is None:
            continue
        consumer_task_id = first_task_by_gap.get(str(gap.get("gap_id") or ""))
        if not consumer_task_id:
            continue
        consumer_task = task_by_id[consumer_task_id]
        for dep in definition.default_dependencies:
            dependency = str(dep).strip().casefold()
            if dependency not in gap_by_capability:
                # The dependency may already be verified project state; only missing
                # capabilities need an implementation edge.
                continue
            if _would_create_capability_cycle(accepted_edges, capability, dependency):
                continue
            required_provide = required_provide_by_capability.get(dependency)
            if not required_provide:
                continue
            consumes = [str(item) for item in consumer_task.get("consumes", ()) if str(item)]
            if required_provide not in consumes:
                consumes.append(required_provide)
                consumer_task["consumes"] = consumes
            accepted_edges.setdefault(capability, []).append(dependency)

    # Rebind from consumes after adding cross-system edges. This recomputes direct
    # task dependencies and task hashes and preserves the existing one-provider rule.
    return _evidence._bind_consumes_dependencies(
        mutable,
        root_provides={"target:frozen"},
    )


def _host_stub_all_candidates(
    clause: str,
    source_start: int,
    source_end: int,
    prompt: str,
) -> Any:
    """Legacy/fallback semantic path: keep every explicit host-recognized root."""

    resolution = resolve_capabilities_from_phrase_structured(clause)
    candidates = tuple(
        dict.fromkeys(
            node.capability_id
            for node in resolution.nodes
            if node.origin in {"explicit", "archetype_inferred"}
            and not node.capability_id.startswith(("unresolved:", "provisional:"))
        )
    )
    return _evidence.SemanticRequirementIR(
        source_start=source_start,
        source_end=source_end,
        source_sha256=_evidence._sha(prompt[source_start:source_end]),
        intent=clause,
        gameplay_capability_candidates=candidates,
        confidence=0.85 if candidates else 0.0,
        unresolved=False,
    )


def _host_invoke_semantic_model(
    clause: str,
    source_start: int,
    source_end: int,
    prompt: str,
    router: Any | None,
) -> Any:
    # Production semantic interpretation is owned by semantic_requirement_authority.
    # This legacy evidence-first fallback is deterministic and never emits planner JSON.
    del router
    return _host_stub_all_candidates(clause, source_start, source_end, prompt)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    # Remove the historical one-candidate truncation and JSON semantic fallback.
    _evidence._stub_semantic_model = _host_stub_all_candidates
    _evidence._invoke_semantic_model = _host_invoke_semantic_model

    # Active production semantic path: plain-text leaf decomposition, one authored
    # clause per scope, no array maxItems and no fixed semantic item count.
    _semantic._call_semantic_model = _plain_semantic_call
    _semantic._generate_approved_nodes = _unbounded_generate_approved_nodes

    original_build_catalog = _semantic._build_catalog
    if not getattr(original_build_catalog, "__mmm_unbounded_dependency_graph__", False):
        _build_catalog_with_dependency_closure.__wrapped__ = original_build_catalog  # type: ignore[attr-defined]
        _build_catalog_with_dependency_closure.__mmm_unbounded_dependency_graph__ = True  # type: ignore[attr-defined]
        _semantic._build_catalog = _build_catalog_with_dependency_closure

    original_compile_tasks = _evidence._compile_tasks
    if not getattr(original_compile_tasks, "__mmm_cross_system_dependencies__", False):
        _compile_tasks_with_cross_system_dependencies.__wrapped__ = original_compile_tasks  # type: ignore[attr-defined]
        _compile_tasks_with_cross_system_dependencies.__mmm_cross_system_dependencies__ = True  # type: ignore[attr-defined]
        _evidence._compile_tasks = _compile_tasks_with_cross_system_dependencies

    # semantic_requirement_authority is installed immediately before this contract;
    # keep the request guard pointed at its now-strengthened catalog builder.
    _guard.build_authoritative_request_catalog = _semantic.build_approved_requirement_catalog
    _INSTALLED = True


__all__ = ["install"]
