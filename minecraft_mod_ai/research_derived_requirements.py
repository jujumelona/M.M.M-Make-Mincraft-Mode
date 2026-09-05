from __future__ import annotations

"""Evidence-backed derivation of implementation requirements.

Authored requirements remain immutable. The host first closes each fixed engineering
facet from the requirement-bound PlanIR task slice, then lets the small planner add only
external-evidence-backed obligations. One model turn handles one complete requirement
template; generic absence of research can never manufacture dozens of unresolved facets.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .evidence_first_planning import validate_evidence_first_plan
from .research_requirement_evidence import (
    evidence_catalog,
    facet_relevant_refs,
    requirement_evidence_window,
)
from .research_requirement_plan_slice import (
    host_facet_baseline,
    render_task_slice,
    requirement_task_slice,
)
from .research_requirement_schema import (
    DISPOSITIONS,
    FACETS,
    REQUIREMENT_RESPONSE_SCHEMA,
)

SCHEMA = "mmm/research-derived-requirements-v2"


class ResearchRequirementError(ValueError):
    """Raised when evidence exposes an implementation facet that cannot be closed safely."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _require_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ResearchRequirementError(
            f"derived requirement field {field!r} must be non-empty"
        )
    return text


def _model_requirement_facets(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    evidence_window: Sequence[Mapping[str, Any]],
    relevant_refs: Mapping[str, Sequence[str]],
) -> Mapping[str, Mapping[str, Any]]:
    parent = _require_text(
        requirement.get("requirement_id"),
        field="parent_requirement_ref",
    )
    payload = {
        "requirement": {
            "requirement_id": parent,
            "capability": requirement.get("capability"),
            "statement": str(requirement.get("statement") or "")[:1200],
            "implementation_capabilities": list(
                _strings(requirement.get("implementation_capabilities"))
            ),
            "artifact_obligations": list(
                _strings(requirement.get("artifact_obligations"))
            ),
            "acceptance": list(_strings(requirement.get("acceptance"))),
        },
        "host_task_slice": render_task_slice(tasks),
        "host_baseline": [dict(baseline[facet]) for facet in FACETS],
        "facet_relevant_evidence_refs": {
            facet: list(relevant_refs.get(facet, ())) for facet in FACETS
        },
        "evidence_catalog": list(evidence_window),
    }
    system = (
        "You are filling one host-owned research-to-requirements template for a "
        "Minecraft mod. The requirement identity, seven facet names, target/task "
        "ownership, and host baseline are immutable. Return all seven facets exactly "
        "once. Do not invent paths, APIs, versions, repositories, or evidence refs. "
        "The host baseline is authoritative. You may change a baseline disposition to "
        "derived ONLY when supplied external evidence directly requires an additional "
        "implementation obligation. A derived facet must cite only refs listed for that "
        "facet and include concrete implementation obligations plus observable acceptance. "
        "Use unresolved only when a facet has listed relevant external evidence indicating "
        "a real requirement but that evidence is insufficient to specify it safely. Never "
        "use unresolved merely because generic research is absent."
    )
    try:
        raw = router.generate_text(
            "planner",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": _canonical(payload)},
            ],
            response_format="json",
            response_schema=REQUIREMENT_RESPONSE_SCHEMA,
            enable_tools=False,
        )
        decoded = json.loads(raw)
    except Exception as exc:
        raise ResearchRequirementError(
            f"research-derived requirement decision failed for {parent!r}"
        ) from exc

    facets = decoded.get("facets") if isinstance(decoded, Mapping) else None
    if not isinstance(facets, list):
        raise ResearchRequirementError(
            f"research-derived requirement decision for {parent!r} has no facets"
        )
    by_facet: dict[str, Mapping[str, Any]] = {}
    for item in facets:
        if not isinstance(item, Mapping):
            raise ResearchRequirementError(
                f"research-derived requirement decision for {parent!r} contains a non-object facet"
            )
        facet = str(item.get("facet") or "").strip()
        if facet not in FACETS or facet in by_facet:
            raise ResearchRequirementError(
                f"requirement {parent!r} has an invalid or duplicate facet {facet!r}"
            )
        by_facet[facet] = item
    missing = [facet for facet in FACETS if facet not in by_facet]
    if missing:
        raise ResearchRequirementError(
            f"requirement {parent!r} omitted derivation facets: {missing}"
        )
    return by_facet


def _derived_decision(
    parent: str,
    facet: str,
    candidate: Mapping[str, Any],
    *,
    facet_allowed_refs: set[str],
    allowed_refs: set[str],
) -> dict[str, Any]:
    rationale = _require_text(
        candidate.get("rationale"),
        field=f"{parent}.{facet}.rationale",
    )
    evidence_refs = _strings(candidate.get("evidence_refs"))
    if not set(evidence_refs) <= allowed_refs:
        unknown = sorted(set(evidence_refs) - allowed_refs)
        raise ResearchRequirementError(
            f"requirement {parent!r} facet {facet!r} cites unknown evidence: {unknown}"
        )
    statement = str(candidate.get("statement") or "").strip()
    acceptance = _strings(candidate.get("acceptance"))
    obligations = _strings(candidate.get("implementation_obligations"))
    if (
        not statement
        or not evidence_refs
        or not acceptance
        or not obligations
        or not set(evidence_refs) <= facet_allowed_refs
    ):
        raise ResearchRequirementError(
            f"derived facet {parent}.{facet} lacks facet-bound "
            "evidence/statement/acceptance/obligations"
        )
    return {
        "facet": facet,
        "disposition": "derived",
        "statement": statement,
        "rationale": rationale,
        "evidence_refs": list(evidence_refs),
        "acceptance": list(acceptance),
        "implementation_obligations": list(obligations),
    }


def _merge_model_with_baseline(
    *,
    requirement: Mapping[str, Any],
    baseline: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Mapping[str, Any]] | None,
    relevant_refs: Mapping[str, Sequence[str]],
    allowed_refs: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    parent = _require_text(
        requirement.get("requirement_id"),
        field="parent_requirement_ref",
    )
    decisions: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for facet in FACETS:
        selected = dict(baseline[facet])
        candidate = model.get(facet) if model is not None else None

        if candidate is not None:
            disposition = str(candidate.get("disposition") or "").strip().casefold()
            if disposition not in DISPOSITIONS:
                raise ResearchRequirementError(
                    f"requirement {parent!r} facet {facet!r} "
                    f"has invalid disposition {disposition!r}"
                )
            rationale = _require_text(
                candidate.get("rationale"),
                field=f"{parent}.{facet}.rationale",
            )
            facet_allowed = set(relevant_refs.get(facet, ()))
            if disposition == "derived":
                selected = _derived_decision(
                    parent,
                    facet,
                    candidate,
                    facet_allowed_refs=facet_allowed,
                    allowed_refs=allowed_refs,
                )
            elif disposition == "unresolved":
                if facet_allowed:
                    unresolved.append(f"{parent}:{facet}")
                    selected = {
                        "facet": facet,
                        "disposition": "unresolved",
                        "statement": str(candidate.get("statement") or "").strip(),
                        "rationale": rationale,
                        "evidence_refs": list(
                            _strings(candidate.get("evidence_refs"))
                        ),
                        "acceptance": list(_strings(candidate.get("acceptance"))),
                        "implementation_obligations": list(
                            _strings(candidate.get("implementation_obligations"))
                        ),
                    }
            else:
                selected["rationale"] = (
                    str(selected.get("rationale") or "")
                    + " Model review: "
                    + rationale
                )

        decision = {
            "derived_requirement_id": "derived_"
            + _sha(
                {
                    "parent": parent,
                    "facet": facet,
                    "statement": selected.get("statement", ""),
                    "disposition": selected["disposition"],
                }
            )[7:27],
            "parent_requirement_ref": parent,
            "provenance_role": "logically_derived",
            "facet": facet,
            "disposition": selected["disposition"],
            "statement": str(selected.get("statement") or "").strip(),
            "rationale": _require_text(
                selected.get("rationale"),
                field=f"{parent}.{facet}.rationale",
            ),
            "evidence_refs": list(_strings(selected.get("evidence_refs"))),
            "acceptance": list(_strings(selected.get("acceptance"))),
            "implementation_obligations": list(
                _strings(selected.get("implementation_obligations"))
            ),
        }
        decisions.append(decision)

    return decisions, unresolved


def derive_research_requirements(
    router: Any,
    *,
    prompt: str,
    evidence_plan: Mapping[str, Any],
    research_brief: Any,
    technical_evidence: Any,
    game_design: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce a complete derivation matrix with at most one model turn per requirement."""

    validate_evidence_first_plan(evidence_plan, prompt=prompt)
    request_catalog = evidence_plan.get("request_catalog")
    if not isinstance(request_catalog, Mapping):
        raise ResearchRequirementError("evidence plan has no request catalog")
    requirements = request_catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ResearchRequirementError("evidence plan has no authored requirements")

    evidence = evidence_catalog(research_brief, technical_evidence, game_design)
    allowed_refs = {str(item["evidence_ref"]) for item in evidence}
    decisions: list[dict[str, Any]] = []
    unresolved: list[str] = []
    model_calls = 0

    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise ResearchRequirementError("authored requirement is not an object")
        parent = _require_text(
            requirement.get("requirement_id"),
            field="parent_requirement_ref",
        )
        tasks = requirement_task_slice(evidence_plan, parent)
        baseline = host_facet_baseline(requirement, tasks)
        relevant_refs = facet_relevant_refs(evidence, requirement, baseline)
        window = requirement_evidence_window(evidence, relevant_refs)

        model: Mapping[str, Mapping[str, Any]] | None = None
        if window:
            model = _model_requirement_facets(
                router,
                requirement=requirement,
                tasks=tasks,
                baseline=baseline,
                evidence_window=window,
                relevant_refs=relevant_refs,
            )
            model_calls += 1

        merged, requirement_unresolved = _merge_model_with_baseline(
            requirement=requirement,
            baseline=baseline,
            model=model,
            relevant_refs=relevant_refs,
            allowed_refs=allowed_refs,
        )
        decisions.extend(merged)
        unresolved.extend(requirement_unresolved)

    if unresolved:
        raise ResearchRequirementError(
            "research could not close required implementation facets: "
            + ", ".join(unresolved)
        )

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA,
        "prompt_sha256": request_catalog.get("prompt_sha256"),
        "evidence_catalog": list(evidence),
        "facet_decisions": decisions,
        "model_call_policy": {
            "unit": "one_requirement_template",
            "actual_calls": model_calls,
            "max_calls_per_requirement": 1,
            "skip_without_relevant_external_evidence": True,
        },
        "ledger_sha256": "",
    }
    ledger["ledger_sha256"] = _sha(
        {
            key: value
            for key, value in ledger.items()
            if key != "ledger_sha256"
        }
    )
    return ledger


def attach_derived_requirement_ledger(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the immutable sidecar to PlanIR without changing authored requirements."""

    result = json.loads(_canonical(plan))
    result["derived_requirement_ledger"] = json.loads(_canonical(ledger))
    result["plan_sha256"] = ""
    result["plan_sha256"] = _sha(result)
    validate_evidence_first_plan(result)
    return result


__all__ = [
    "FACETS",
    "ResearchRequirementError",
    "attach_derived_requirement_ledger",
    "derive_research_requirements",
]
