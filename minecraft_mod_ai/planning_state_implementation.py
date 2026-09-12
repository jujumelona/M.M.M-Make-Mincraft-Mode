from __future__ import annotations

from .fixed_template_generation import generate_fixed_template_text

"""Host-owned detailed-plan compilation from grounded research.

Each requirement is decomposed into schema-constrained engineering worksheet sections.
The host owns requirement selection, section dependencies, evidence identifiers,
validation, scheduling, and plan assembly. The model never chooses its own response
shape, and detailed planning never depends on free-form section boundaries or
reasoning-label parsing.
"""

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from copy import deepcopy
from typing import Any

from .model_concurrency import (
    planning_work_unit_timeout_seconds,
    router_native_model_parallelism,
    run_with_model_execution_deadline,
)
from .planner_operation import planner_operation
from .planning_detail_contract import validate_detailed_plan_grounding
from .planning_detail_template import (
    WORKSHEET_SECTIONS,
    normalize_required_sections,
    validate_worksheet,
    worksheet_section_prompt,
)
from .planning_state_contract import validate_planning_state
from .root_cause_trace import emit_root_cause
from .worksheet_atomic_chunker import (
    merge_worksheet_section_chunks,
    pack_section_concerns,
    worksheet_chunk_prompt,
    worksheet_chunk_schema,
)

SECTION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "behavior_contract": (),
    "state_model": ("behavior_contract",),
    "integration": ("behavior_contract",),
    "resources_and_ui": ("behavior_contract", "integration"),
    "algorithm": ("behavior_contract", "state_model"),
    "authority_and_network": ("behavior_contract", "state_model", "integration"),
    "persistence": ("state_model", "integration"),
    "reuse_assessment": ("integration",),
    "failure_and_limits": (
        "behavior_contract",
        "state_model",
        "algorithm",
        "integration",
    ),
    "verification": (
        "behavior_contract",
        "algorithm",
        "integration",
        "failure_and_limits",
    ),
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _requirement_decisions(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in state.get("decisions", [])
        if isinstance(item, Mapping) and item.get("decision_type") == "requirement"
    ]


def _implementation_evidence(
    state: Mapping[str, Any], requirement_ref: str
) -> list[Mapping[str, Any]]:
    research_ids = {
        str(item.get("research_id") or "")
        for item in state.get("research_queue", [])
        if isinstance(item, Mapping)
        and str(item.get("requirement_ref") or "") == requirement_ref
        and item.get("status") == "complete"
    }
    rows: list[Mapping[str, Any]] = []
    for item in state.get("evidence", []):
        if (
            isinstance(item, Mapping)
            and str(item.get("research_ref") or "") in research_ids
            and item.get("sufficient") is True
        ):
            trace = item.get("candidate_trace")
            if isinstance(trace, Mapping):
                from .planning_candidate_evidence import requirement_for
                from .planning_semantic_research import validate_semantic_review
                review = validate_semantic_review(
                    requirement_for(state, {"requirement_ref": requirement_ref}),
                    state.get("task_candidate_pool") or {}, trace.get("semantic_review") or {},
                )
                if not review["complete"]:
                    raise ValueError(f"DETAILED_PLAN_EVIDENCE: stale or incomplete semantic review for {requirement_ref}")
                proofs = review["accepted_proofs"]
                rows.append({
                    "research_ref": _text(item.get("research_ref")),
                    "claims": [{"claim": proof["excerpt"], "relevance": proof["reason"],
                                "obligation": proof["obligation"], "evidence_refs": [proof["source_id"]],
                                "content_sha256": proof["content_sha256"]} for proof in proofs],
                    "evidence_refs": list(dict.fromkeys(proof["source_id"] for proof in proofs)),
                    "sufficient": True, "source": "host_verified_semantic_research",
                    "pool_sha256": review["pool_sha256"],
                    "mod_discovery": {"status": "evidence_selected", "complete": True,
                                      "candidates": [{"source_id": proof["source_id"],
                                                      "name": proof["source_title"],
                                                      "url": proof["source_url"],
                                                      "compatibility": "not_verified",
                                                      "reuse_authority": "verification_required"}
                                                     for proof in proofs]},
                })
                continue
            rows.append(
                {
                    "research_ref": _text(item.get("research_ref")),
                    "claims": deepcopy(item.get("claims") or []),
                    "evidence_refs": [
                        _text(ref) for ref in item.get("evidence_refs", []) if _text(ref)
                    ],
                    "sufficient": True,
                    "source": _text(item.get("source")),
                    "mod_discovery": deepcopy(item.get("mod_discovery") or {}),
                }
            )
    return rows


def _allowed_refs(evidence: list[Mapping[str, Any]]) -> set[str]:
    return {
        _text(ref)
        for item in evidence
        for ref in item.get("evidence_refs", [])
        if _text(ref)
    }


def _requirement_grounding(
    state: Mapping[str, Any], requirement_ref: str
) -> tuple[list[Mapping[str, Any]], set[str]]:
    evidence = _implementation_evidence(state, requirement_ref)
    allowed = _allowed_refs(evidence)
    if not evidence or not allowed:
        raise ValueError(
            f"DETAILED_PLAN_EVIDENCE: {requirement_ref} has no sufficient grounded implementation evidence"
        )
    return evidence, allowed


def _preflight_detailed_planning(
    state: Mapping[str, Any], requirements: list[Mapping[str, Any]]
) -> None:
    seen: set[str] = set()
    for requirement in requirements:
        requirement_ref = _text(requirement.get("requirement_id"))
        if not requirement_ref or requirement_ref in seen:
            raise ValueError(
                "DETAILED_PLAN_REQUIREMENTS: requirement IDs must be non-empty and unique"
            )
        seen.add(requirement_ref)
        _requirement_grounding(state, requirement_ref)


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _evidence_context(evidence: list[Mapping[str, Any]]) -> str:
    from .planning_mod_discovery import discovery_context

    rows: list[str] = []
    for item in evidence:
        research_ref = _text(item.get("research_ref"))
        refs = ", ".join(
            _text(ref) for ref in item.get("evidence_refs", []) if _text(ref)
        )
        claims = item.get("claims") or []
        claim_text = " | ".join(_text(claim) for claim in claims if _text(claim))
        source = _text(item.get("source"))
        rows.append(
            f"- research_ref={research_ref}; evidence_refs=[{refs}]; "
            f"source={source or 'unspecified'}; claims={claim_text or 'no claim prose'}"
        )
        discovery = item.get("mod_discovery")
        if isinstance(discovery, Mapping) and discovery:
            rows.append("Catalog discovery: " + discovery_context(discovery))
    return "\n".join(rows)


def _section_dependencies(
    section: str, selected_sections: tuple[str, ...]
) -> tuple[str, ...]:
    selected = set(selected_sections)
    return tuple(
        dependency
        for dependency in SECTION_DEPENDENCIES[section]
        if dependency in selected
    )


def _section_dependency_context(
    section: str,
    selected_sections: tuple[str, ...],
    completed: Mapping[str, Mapping[str, Any]],
) -> str:
    dependencies = _section_dependencies(section, selected_sections)
    if not dependencies:
        return "- none; this section has no worksheet prerequisites"

    # Preserve JSON field boundaries and every prerequisite rule. Prompt transport
    # owns context budgeting; slicing a serialized contract silently loses semantics.
    return json.dumps(
        {dependency: completed[dependency] for dependency in dependencies},
        ensure_ascii=False, separators=(",", ":"),
    )


def _section_messages(
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    section: str,
    evidence: list[Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    statement = _text(requirement.get("statement"))
    acceptance = requirement.get("acceptance")
    acceptance_rows = (
        [_text(item) for item in acceptance if _text(item)]
        if isinstance(acceptance, list)
        else []
    )
    acceptance_text = "\n".join(f"- {row}" for row in acceptance_rows) or "- none supplied"
    prerequisite_context = _section_dependency_context(
        section, selected_sections, completed
    )
    return [
        {
            "role": "system",
            "content": (
                "Complete exactly one host-selected engineering worksheet section for one requirement. "
                "Return only the JSON object required by the supplied response schema. "
                "Do not emit analysis, reasoning, commentary, markdown, code fences, or keys outside "
                "that schema. Do not invent target API names, symbols, versions, repository paths, "
                "external facts, or evidence identifiers. Use only evidence_refs shown in the grounded "
                "context. Direct prerequisite section results are authoritative continuity constraints; "
                "do not regenerate or restate unrelated worksheet sections."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {statement}\n"
                "Acceptance observations supplied by the requirement:\n"
                f"{acceptance_text}\n"
                "Grounded implementation evidence:\n"
                f"{_evidence_context(evidence)}\n\n"
                "Direct prerequisite worksheet sections:\n"
                f"{prerequisite_context}\n\n"
                f"{worksheet_section_prompt(section)}"
            ),
        },
    ]


def _structured_output_text(exc: BaseException) -> str:
    return str(getattr(exc, "output", "") or "")


def _chunk_messages(
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    section: str,
    evidence: list[Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
    *,
    chunk_index: int,
    chunk_count: int,
    concerns: tuple[str, ...],
    include_evidence: bool = False,
    repair_error: str = "",
) -> list[dict[str, str]]:
    statement = _text(requirement.get("statement"))
    acceptance = requirement.get("acceptance")
    acceptance_rows = (
        [_text(item) for item in acceptance if _text(item)]
        if isinstance(acceptance, list)
        else []
    )
    acceptance_text = "\n".join(f"- {row}" for row in acceptance_rows) or "- none supplied"
    prerequisite_context = _section_dependency_context(
        section, selected_sections, completed
    )
    instruction = (
        f"Complete atomic concern chunk {chunk_index}/{chunk_count} of engineering worksheet section {section!r}. "
        "Return only the JSON object matching the supplied template skeleton. "
        "Do not output JSON Schema definitions (no 'type', 'properties', 'required', 'additionalProperties'). "
        "Do not emit analysis, reasoning, commentary, markdown, code fences, or undeclared keys. "
        "Do not invent target API names, symbols, versions, repository paths, "
        "external facts, or evidence identifiers. Use only evidence_refs shown in the grounded context."
    )
    if repair_error:
        instruction += f" Previous output failed validation: {repair_error[:800]}. Please repair."

    return [
        {
            "role": "system",
            "content": instruction,
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {statement}\n"
                "Acceptance observations supplied by the requirement:\n"
                f"{acceptance_text}\n"
                "Grounded implementation evidence:\n"
                f"{_evidence_context(evidence)}\n\n"
                "Direct prerequisite worksheet sections:\n"
                f"{prerequisite_context}\n\n"
                f"{worksheet_chunk_prompt(section, chunk_index, chunk_count, concerns, include_evidence=include_evidence)}"
            ),
        },
    ]


def _generate_chunk(
    router: Any,
    messages: list[dict[str, str]],
    *,
    section: str,
    index: int,
    concerns: Sequence[str],
    chunk_schema: Mapping[str, Any],
) -> dict[str, Any]:
    tool_name = f"submit_{section}_{index}_chunk"
    description = f"Submit worksheet specifications for {section}: {', '.join(concerns)}."

    if hasattr(router, "generate_tool_decision"):
        try:
            raw_decision = router.generate_tool_decision(
                "planner",
                messages,
                tool_name=tool_name,
                parameters=chunk_schema,
                description=description,
            )
            if isinstance(raw_decision, Mapping):
                return dict(raw_decision)
        except Exception as exc:
            from .model_adapters import ModelConfigurationError

            if isinstance(exc, ModelConfigurationError):
                raise
            # Fall back to text generation if native tool call fails or is not enabled for role

    raw = generate_fixed_template_text(router,
        "planner",
        messages,
        response_schema=chunk_schema,
        enable_tools=False,
    )
    from .planning_contract_ssot import is_schema_definition_echo

    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise ValueError("chunk output must be a JSON object")
    if is_schema_definition_echo(decoded):
        raise ValueError(
            "Model returned JSON Schema definition instead of concrete data records. "
            "Please output records matching the template skeleton."
        )
    return dict(decoded)


def _compile_worksheet_section(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    section: str,
    evidence: list[Mapping[str, Any]],
    allowed: set[str],
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate and validate one dependency-scoped worksheet section via atomic concern chunks."""

    requirement_ref = _text(requirement.get("requirement_id"))
    operation = f"detailed_section:{section}"
    chunks_def = pack_section_concerns(section)
    chunk_count = len(chunks_def)
    chunk_results: list[dict[str, Any]] = []

    try:
        with planner_operation(operation):
            for index, concerns in enumerate(chunks_def, start=1):
                is_first = index == 1
                chunk_schema = worksheet_chunk_schema(
                    section, concerns, include_evidence=is_first
                )
                messages = _chunk_messages(
                    requirement,
                    selected_sections,
                    section,
                    evidence,
                    completed,
                    chunk_index=index,
                    chunk_count=chunk_count,
                    concerns=concerns,
                    include_evidence=is_first,
                )
                try:
                    decoded = _generate_chunk(
                        router,
                        messages,
                        section=section,
                        index=index,
                        concerns=concerns,
                        chunk_schema=chunk_schema,
                    )
                except (json.JSONDecodeError, ValueError) as parse_err:
                    repair_messages = _chunk_messages(
                        requirement,
                        selected_sections,
                        section,
                        evidence,
                        completed,
                        chunk_index=index,
                        chunk_count=chunk_count,
                        concerns=concerns,
                        include_evidence=is_first,
                        repair_error=str(parse_err),
                    )
                    decoded = _generate_chunk(
                        router,
                        repair_messages,
                        section=section,
                        index=index,
                        concerns=concerns,
                        chunk_schema=chunk_schema,
                    )

                chunk_results.append(decoded)

            return merge_worksheet_section_chunks(section, chunk_results, allowed)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raw_text = _structured_output_text(exc)
        emit_root_cause(
            "detailed_section_failure",
            stage="planning_state",
            operation=operation,
            gate="structured_section_validation",
            result="FAIL",
            reason=f"{type(exc).__name__}: {exc}",
            details={
                "requirement_ref": requirement_ref,
                "section": section,
                "prerequisite_sections": list(
                    _section_dependencies(section, selected_sections)
                ),
                "raw_output": raw_text,
                "raw_output_chars": len(raw_text),
                "response_format": "json",
                "chunks_count": chunk_count,
                "allowed_evidence_refs": sorted(allowed),
                "parser_rule": (
                    "atomic JSON worksheet chunks merged and validated by host; "
                    "evidence refs are validated by the host"
                ),
            },
            exc=exc,
        )
        raise


def _host_derived_capabilities(
    worksheet: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "capability": f"Preserve the {section} contract: {json.dumps(row['specification'], ensure_ascii=False, separators=(',', ':'))}",
            "constraint_evidence_refs": [],
        }
        for section, row in worksheet.items()
    ]


def _host_derived_obligations(
    worksheet: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "obligation": f"Implement and verify the {section} contract: {json.dumps(row['specification'], ensure_ascii=False, separators=(',', ':'))}",
            "constraint_evidence_refs": [],
        }
        for section, row in worksheet.items()
    ]


def _host_derived_checks(
    requirement: Mapping[str, Any], worksheet: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    acceptance = requirement.get("acceptance")
    checks = (
        [_text(item) for item in acceptance if _text(item)]
        if isinstance(acceptance, list)
        else []
    )
    if not checks:
        verification = worksheet.get("verification", {})
        specification = verification.get("specification")
        if specification:
            checks = [json.dumps(specification, ensure_ascii=False, separators=(",", ":"))]
    if not checks:
        checks = [
            "Given the requirement is exercised, verify the observable behavior: "
            + _text(requirement.get("statement"))
        ]
    return [{"check": check, "constraint_evidence_refs": []} for check in checks]


def _assemble_requirement_plan(
    requirement: Mapping[str, Any],
    requirement_ref: str,
    selected_sections: tuple[str, ...],
    worksheet: Mapping[str, Mapping[str, Any]],
    allowed: set[str],
) -> dict[str, Any]:
    validated_worksheet = validate_worksheet(worksheet, allowed, selected_sections)
    plan = {
        "requirement_ref": requirement_ref,
        "required_detail_sections": list(selected_sections),
        "engineering_worksheet": validated_worksheet,
        "worksheet_contract": "authored_concern_records",
        "implementation_capabilities": _host_derived_capabilities(validated_worksheet),
        "implementation_obligations": _host_derived_obligations(validated_worksheet),
        "artifact_obligations": [],
        "grounded_bindings": [],
        "reuse_candidates": [],
        "verification_obligations": _host_derived_checks(
            requirement, validated_worksheet
        ),
    }
    validate_detailed_plan_grounding(plan, allowed)
    return plan


def _host_section_selection(
    requirements: list[Mapping[str, Any]],
    required_sections_by_requirement: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    requirement_ids = [str(item.get("requirement_id") or "") for item in requirements]
    if required_sections_by_requirement is None:
        return {
            requirement_id: normalize_required_sections()
            for requirement_id in requirement_ids
        }
    if not isinstance(required_sections_by_requirement, Mapping):
        raise ValueError("DETAILED_PLAN_SECTIONS: host selection must be a requirement mapping")
    unknown = set(str(key) for key in required_sections_by_requirement) - set(requirement_ids)
    if unknown:
        raise ValueError(
            "DETAILED_PLAN_SECTIONS: selection cites unknown requirement(s): "
            + ", ".join(sorted(unknown))
        )
    return {
        requirement_id: normalize_required_sections(
            required_sections_by_requirement.get(requirement_id)
        )
        for requirement_id in requirement_ids
    }


def _compile_requirement_plans_dag(
    router: Any,
    state: Mapping[str, Any],
    requirements: list[Mapping[str, Any]],
    section_selection: Mapping[str, tuple[str, ...]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Schedule all ready ``(requirement, section)`` nodes across one global pool."""

    jobs: list[dict[str, Any]] = []
    section_rank = {section: index for index, section in enumerate(WORKSHEET_SECTIONS)}
    for requirement in requirements:
        requirement_ref = _text(requirement.get("requirement_id"))
        evidence, allowed = _requirement_grounding(state, requirement_ref)
        selected_sections = section_selection[requirement_ref]
        jobs.append(
            {
                "requirement": requirement,
                "requirement_ref": requirement_ref,
                "selected_sections": selected_sections,
                "evidence": evidence,
                "allowed": allowed,
                "completed": {},
                "pending": set(selected_sections),
                "submitted": set(),
            }
        )

    def ready_nodes() -> list[tuple[int, str]]:
        ready: list[tuple[int, str]] = []
        for job_index, job in enumerate(jobs):
            selected_sections = job["selected_sections"]
            completed = job["completed"]
            for section in selected_sections:
                if section not in job["pending"] or section in job["submitted"]:
                    continue
                dependencies = _section_dependencies(section, selected_sections)
                if all(dependency in completed for dependency in dependencies):
                    ready.append((job_index, section))
        return sorted(ready, key=lambda item: (section_rank[item[1]], item[0]))

    max_workers = max(1, workers)
    future_to_node: dict[Future[dict[str, Any]], tuple[int, str]] = {}
    future_deadlines: dict[Future[dict[str, Any]], float] = {}
    pool = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="planning-detail",
    )
    try:
        while any(job["pending"] for job in jobs) or future_to_node:
            for job_index, section in ready_nodes():
                if len(future_to_node) >= max_workers:
                    break
                job = jobs[job_index]
                dependencies = _section_dependencies(
                    section, job["selected_sections"]
                )
                prerequisite_snapshot = {
                    dependency: deepcopy(job["completed"][dependency])
                    for dependency in dependencies
                }
                deadline = time.monotonic() + planning_work_unit_timeout_seconds()
                context_copy = copy_context()
                future = pool.submit(
                    context_copy.run,
                    run_with_model_execution_deadline,
                    deadline,
                    _compile_worksheet_section,
                    router,
                    requirement=job["requirement"],
                    selected_sections=job["selected_sections"],
                    section=section,
                    evidence=job["evidence"],
                    allowed=job["allowed"],
                    completed=prerequisite_snapshot,
                )
                job["submitted"].add(section)
                future_to_node[future] = (job_index, section)
                future_deadlines[future] = deadline

            if not future_to_node:
                pending = {
                    job["requirement_ref"]: sorted(
                        job["pending"], key=section_rank.__getitem__
                    )
                    for job in jobs
                    if job["pending"]
                }
                raise RuntimeError(
                    "DETAILED_PLAN_DAG_DEADLOCK: no ready section nodes; "
                    + repr(pending)
                )

            nearest_deadline = min(future_deadlines.values())
            timeout = max(0.0, nearest_deadline - time.monotonic())
            done, _ = wait(
                tuple(future_to_node),
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                now = time.monotonic()
                expired = [
                    future
                    for future, deadline in future_deadlines.items()
                    if now >= deadline
                ]
                if not expired:
                    continue
                nodes = [future_to_node[future] for future in expired]
                for future in expired:
                    future.cancel()
                raise TimeoutError(f"DETAILED_PLAN_SECTION_TIMEOUT: {nodes}")

            completed_futures = sorted(
                done,
                key=lambda future: (
                    section_rank[future_to_node[future][1]],
                    future_to_node[future][0],
                ),
            )
            for future in completed_futures:
                job_index, section = future_to_node.pop(future)
                future_deadlines.pop(future, None)
                job = jobs[job_index]
                result = future.result(timeout=0)
                job["completed"][section] = result
                job["pending"].remove(section)
                job["submitted"].remove(section)
    finally:
        for future in future_to_node:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    compiled: list[dict[str, Any]] = []
    for job in jobs:
        worksheet = {
            section: job["completed"][section]
            for section in job["selected_sections"]
        }
        compiled.append(
            _assemble_requirement_plan(
                job["requirement"],
                job["requirement_ref"],
                job["selected_sections"],
                worksheet,
                job["allowed"],
            )
        )
    return compiled


def compile_detailed_implementation_plans(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    required_sections_by_requirement: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    validate_planning_state(state, prompt=prompt)
    value = deepcopy(dict(state))
    requirements = _requirement_decisions(value)
    if not requirements:
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: no researched requirements exist")

    _preflight_detailed_planning(value, requirements)
    section_selection = _host_section_selection(requirements, required_sections_by_requirement)
    total_section_tasks = sum(
        len(section_selection[_text(requirement.get("requirement_id"))])
        for requirement in requirements
    )
    workers = max(
        1,
        min(total_section_tasks, router_native_model_parallelism(router)),
    )
    compiled = _compile_requirement_plans_dag(
        router,
        value,
        requirements,
        section_selection,
        workers=workers,
    )

    detailed: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for index, plan in enumerate(compiled, start=1):
        decision_id = f"detail_{index:03d}"
        detailed.append(
            {"decision_id": decision_id, "decision_type": "detailed_implementation_plan", **plan}
        )
        coverage.append(
            {
                "requirement_ref": plan["requirement_ref"],
                "status": "covered",
                "detailed_plan_ref": decision_id,
            }
        )

    value["decisions"] = [
        item
        for item in value.get("decisions", [])
        if not (
            isinstance(item, Mapping)
            and item.get("decision_type") == "detailed_implementation_plan"
        )
    ] + detailed
    value["coverage"] = coverage
    blocking = [
        item
        for item in value.get("unresolved", [])
        if isinstance(item, Mapping) and item.get("status") != "resolved"
    ]
    value["plan_ready"] = not blocking and len(coverage) == len(requirements)
    if not value["plan_ready"]:
        raise ValueError("DETAILED_PLAN_NOT_READY: unresolved planning obligations remain")

    result = _rehash(value)
    validate_planning_state(result, prompt=prompt)
    return result


__all__ = ["SECTION_DEPENDENCIES", "compile_detailed_implementation_plans"]