from __future__ import annotations

"""Evidence-backed augmentation of host-owned implementation requirements.

Authored requirements, target receipts, facet identity, task identity, and baseline
acceptance remain host-owned.  The small planner receives one evidence-bearing facet at
a time and may only decide whether external evidence adds a missing implementation
obligation.  Invalid or empty model output is contained to that optional augmentation;
it can never destroy the host template or abort planning by itself.
"""

import hashlib
import json
import logging
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
from .research_requirement_schema import FACETS
from .research_requirement_template import (
    DECISIONS,
    FACET_AUGMENTATION_RESPONSE_SCHEMA,
    build_facet_slot,
    build_host_planning_context,
)

SCHEMA = "mmm/research-derived-requirements-v3"
_LOGGER = logging.getLogger(__name__)


class ResearchRequirementError(ValueError):
    """Raised when host structure or valid evidence exposes an unsafe unresolved facet."""


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


def _facet_evidence(
    evidence_window: Sequence[Mapping[str, Any]],
    allowed_refs: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    allowed = set(allowed_refs)
    return tuple(
        item
        for item in evidence_window
        if str(item.get("evidence_ref") or "") in allowed
    )


def _validated_augmentation(
    *,
    parent: str,
    facet: str,
    candidate: Mapping[str, Any],
    allowed_refs: Sequence[str],
) -> dict[str, Any]:
    decision = str(candidate.get("decision") or "").strip().casefold()
    if decision not in DECISIONS:
        raise ResearchRequirementError(
            f"requirement {parent!r} facet {facet!r} has invalid decision {decision!r}"
        )
    rationale = _require_text(
        candidate.get("rationale"),
        field=f"{parent}.{facet}.rationale",
    )
    evidence_refs = _strings(candidate.get("evidence_refs"))
    obligations = _strings(candidate.get("implementation_obligations"))
    acceptance = _strings(candidate.get("acceptance"))
    allowed = set(allowed_refs)
    unknown = sorted(set(evidence_refs) - allowed)
    if unknown:
        raise ResearchRequirementError(
            f"requirement {parent!r} facet {facet!r} cites disallowed evidence: {unknown}"
        )

    if decision == "add_obligation":
        if not evidence_refs or not obligations or not acceptance:
            raise ResearchRequirementError(
                f"requirement {parent!r} facet {facet!r} add_obligation requires "
                "evidence_refs, implementation_obligations, and acceptance"
            )
    elif decision == "insufficient_evidence":
        if not evidence_refs or obligations or acceptance:
            raise ResearchRequirementError(
                f"requirement {parent!r} facet {facet!r} insufficient_evidence must "
                "cite evidence and leave obligations/acceptance empty"
            )
    elif obligations or acceptance:
        raise ResearchRequirementError(
            f"requirement {parent!r} facet {facet!r} no_addition must leave "
            "obligations/acceptance empty"
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "evidence_refs": list(evidence_refs),
        "implementation_obligations": list(obligations),
        "acceptance": list(acceptance),
    }


def _model_facet_augmentation(
    router: Any,
    *,
    parent: str,
    facet: str,
    slot: Mapping[str, Any],
    allowed_refs: Sequence[str],
) -> tuple[Mapping[str, Any] | None, int, list[dict[str, Any]]]:
    """Fill one semantic slot; malformed model output degrades to the host baseline."""

    system = (
        "Fill exactly ONE semantic decision slot in an immutable host-owned Minecraft "
        "mod planning template. Everything under host_owned is read-only and must never "
        "be repeated, rewritten, renamed, or inferred. In particular, do not output the "
        "facet name, requirement/task IDs, Minecraft version, loader, Java version, "
        "mappings, Gradle/Fabric coordinates, paths, or baseline fields. Return only the "
        "five fields allowed by model_slot. Choose add_obligation only when supplied "
        "allowed evidence directly requires concrete work absent from host_baseline. "
        "Choose insufficient_evidence only when supplied evidence exposes a real missing "
        "requirement but cannot safely specify it. Otherwise choose no_addition. Never "
        "invent evidence refs, APIs, repositories, versions, or implementation details."
    )
    events: list[dict[str, Any]] = []
    calls = 0
    for attempt in (1, 2):
        raw: Any = ""
        calls += 1
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": _canonical(slot)},
            ]
            if attempt == 2:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "The previous response was rejected. Return only one object "
                            "matching the response schema exactly; do not add prose."
                        ),
                    }
                )
            raw = router.generate_text(
                "planner",
                messages,
                response_format="json",
                response_schema=FACET_AUGMENTATION_RESPONSE_SCHEMA,
                enable_tools=False,
            )
            if isinstance(raw, Mapping):
                decoded: Any = dict(raw)
            else:
                decoded = json.loads(str(raw))
            if not isinstance(decoded, Mapping):
                raise ResearchRequirementError("facet augmentation response is not an object")
            validated = _validated_augmentation(
                parent=parent,
                facet=facet,
                candidate=decoded,
                allowed_refs=allowed_refs,
            )
            events.append(
                {
                    "stage": "research_facet_augmentation",
                    "requirement_id": parent,
                    "facet": facet,
                    "attempt": attempt,
                    "status": "accepted",
                    "fallback_used": False,
                    "decision": validated["decision"],
                }
            )
            return validated, calls, events
        except Exception as exc:  # model/structured-output failures are optional here
            snippet = str(raw or "").strip().replace("\n", " ")[:1000]
            terminal = attempt == 2
            event = {
                "stage": "research_facet_augmentation",
                "requirement_id": parent,
                "facet": facet,
                "attempt": attempt,
                "status": "fallback_host_baseline" if terminal else "retry",
                "fallback_used": terminal,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "raw_response_snippet": snippet,
            }
            events.append(event)
            _LOGGER.warning(
                "research_facet_augmentation_failed requirement=%s facet=%s "
                "attempt=%s fallback=%s error_type=%s error=%s raw=%r",
                parent,
                facet,
                attempt,
                terminal,
                type(exc).__name__,
                str(exc)[:500],
                snippet[:500],
            )

    return None, calls, events


def _merge_model_with_baseline(
    *,
    requirement: Mapping[str, Any],
    baseline: Mapping[str, Mapping[str, Any]],
    augmentations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    parent = _require_text(
        requirement.get("requirement_id"),
        field="parent_requirement_ref",
    )
    decisions: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for facet in FACETS:
        selected = dict(baseline[facet])
        candidate = augmentations.get(facet)
        if candidate is not None:
            decision = str(candidate["decision"])
            if decision == "add_obligation":
                obligations = _strings(candidate.get("implementation_obligations"))
                selected = {
                    "facet": facet,
                    "disposition": "derived",
                    "statement": (
                        f"External evidence adds {len(obligations)} implementation "
                        f"obligation(s) to host facet {facet}."
                    ),
                    "rationale": str(candidate["rationale"]),
                    "evidence_refs": list(_strings(candidate.get("evidence_refs"))),
                    "acceptance": list(_strings(candidate.get("acceptance"))),
                    "implementation_obligations": list(obligations),
                }
            elif decision == "insufficient_evidence":
                unresolved.append(f"{parent}:{facet}")
                selected = {
                    "facet": facet,
                    "disposition": "unresolved",
                    "statement": (
                        f"External evidence exposes an unresolved obligation for {facet}."
                    ),
                    "rationale": str(candidate["rationale"]),
                    "evidence_refs": list(_strings(candidate.get("evidence_refs"))),
                    "acceptance": [],
                    "implementation_obligations": [],
                }
            else:
                selected["rationale"] = (
                    str(selected.get("rationale") or "")
                    + " Evidence-slot review: "
                    + str(candidate["rationale"])
                )
                selected["evidence_refs"] = list(
                    _strings(candidate.get("evidence_refs"))
                )

        decision_record = {
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
        decisions.append(decision_record)

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
    """Build host templates and fill only evidence-bearing semantic slots."""

    validate_evidence_first_plan(evidence_plan, prompt=prompt)
    request_catalog = evidence_plan.get("request_catalog")
    if not isinstance(request_catalog, Mapping):
        raise ResearchRequirementError("evidence plan has no request catalog")
    requirements = request_catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ResearchRequirementError("evidence plan has no authored requirements")

    try:
        planning_context = build_host_planning_context(router, game_design)
    except Exception as exc:
        raise ResearchRequirementError(
            "host planning template has no resolved platform context"
        ) from exc

    evidence = evidence_catalog(research_brief, technical_evidence, game_design)
    decisions: list[dict[str, Any]] = []
    unresolved: list[str] = []
    augmentation_events: list[dict[str, Any]] = []
    model_calls = 0
    evidence_bearing_slots = 0

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
        rendered_tasks = render_task_slice(tasks)
        augmentations: dict[str, Mapping[str, Any]] = {}

        for facet in FACETS:
            refs = tuple(dict.fromkeys(str(ref) for ref in relevant_refs.get(facet, ()) if str(ref)))
            if not refs:
                continue
            facet_catalog = _facet_evidence(window, refs)
            if not facet_catalog:
                continue
            evidence_bearing_slots += 1
            slot = build_facet_slot(
                planning_context=planning_context,
                requirement=requirement,
                facet=facet,
                baseline=baseline[facet],
                task_slice=rendered_tasks,
                evidence_catalog=facet_catalog,
                allowed_evidence_refs=refs,
            )
            candidate, calls, events = _model_facet_augmentation(
                router,
                parent=parent,
                facet=facet,
                slot=slot,
                allowed_refs=refs,
            )
            model_calls += calls
            augmentation_events.extend(events)
            if candidate is not None:
                augmentations[facet] = candidate

        merged, requirement_unresolved = _merge_model_with_baseline(
            requirement=requirement,
            baseline=baseline,
            augmentations=augmentations,
        )
        decisions.extend(merged)
        unresolved.extend(requirement_unresolved)

    if unresolved:
        raise ResearchRequirementError(
            "research evidence exposed implementation facets that remain semantically "
            "underspecified: " + ", ".join(unresolved)
        )

    fallback_facets = [
        f"{event['requirement_id']}:{event['facet']}"
        for event in augmentation_events
        if event.get("status") == "fallback_host_baseline"
    ]
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA,
        "prompt_sha256": request_catalog.get("prompt_sha256"),
        "host_template": {
            "authority": "host_only_except_bounded_evidence_slots",
            "facet_order": list(FACETS),
            "planning_context": planning_context,
            "host_owned_fields": [
                "platform target receipt and versions",
                "requirement identity and authored acceptance",
                "facet identity/order/purpose",
                "task identity/ownership/dependencies",
                "baseline disposition/acceptance/obligations",
                "derived requirement IDs and ledger hashes",
            ],
        },
        "evidence_catalog": list(evidence),
        "facet_decisions": decisions,
        "augmentation_events": augmentation_events,
        "degraded_augmentation_facets": list(dict.fromkeys(fallback_facets)),
        "model_call_policy": {
            "unit": "one_evidence_bearing_facet_slot",
            "evidence_bearing_slots": evidence_bearing_slots,
            "actual_calls_including_retries": model_calls,
            "max_attempts_per_evidence_bearing_facet": 2,
            "skip_without_relevant_external_evidence": True,
            "malformed_output_fallback": "immutable_host_baseline",
            "model_may_mutate_host_fields": False,
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
