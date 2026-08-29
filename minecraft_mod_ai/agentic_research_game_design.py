from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .external_procedural_skill_contract import _sanitize_procedure, compact_skillbank
from .planner_stage_trace import PlannerStageTrace
from .spec import SpecValidationError

# Transport validation must never reject a model variant that the host parser can
# deterministically canonicalize. Syntax/object shape is the transport boundary; semantic
# normalization, evidence sanitization and procedure validation belong to host code below.
_RESEARCH_NOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "research_note": {
            "type": "object",
            "properties": {
                "domain_id": {"type": "string"},
                "claims": {"type": "array", "items": {}},
                "gaps": {"type": "array", "items": {}},
                "next_queries": {"type": "array", "items": {}},
                "sufficient": {"type": "boolean"},
                "procedures": {"type": "array", "items": {}},
            },
            "additionalProperties": True,
        }
    },
    "additionalProperties": True,
}

_SECTION_SPECS: tuple[tuple[str, tuple[str, ...], dict[str, Any]], ...] = (
    (
        "identity_and_loop",
        ("title", "pitch", "core_loop"),
        {
            "title": {"type": "string", "minLength": 1},
            "pitch": {"type": "string", "minLength": 1},
            "core_loop": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    ),
    (
        "systems_and_progression",
        ("progression", "combat", "mod_context"),
        {
            "progression": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "combat": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "mod_context": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
    ),
    (
        "modules_and_assets",
        ("modules", "assets"),
        {
            "modules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plugin_id": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "minLength": 1},
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["plugin_id", "status", "reason"],
                    "additionalProperties": False,
                },
            },
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "kind": {"type": "string", "minLength": 1},
                        "brief": {"type": "string", "minLength": 1},
                    },
                    "required": ["id", "kind", "brief"],
                    "additionalProperties": False,
                },
            },
        },
    ),
    (
        "quality_and_art",
        ("acceptance_tests", "art_direction"),
        {
            "acceptance_tests": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "art_direction": {"type": "object"},
        },
    ),
)


def supports_agentic_research_router(router: Any) -> bool:
    from .model_router import ModelRouter

    return isinstance(router, ModelRouter)


def _domain_source_value(domain_id: str, value: Any) -> Any:
    """Select one domain while preserving its complete source-level receipt metadata."""

    if not isinstance(value, Mapping):
        return value
    domains = value.get("domains")
    if not isinstance(domains, list):
        return dict(value)
    selected = next(
        (
            item
            for item in domains
            if isinstance(item, Mapping) and item.get("domain_id") == domain_id
        ),
        None,
    )
    receipt = {key: item for key, item in value.items() if key != "domains"}
    if isinstance(selected, Mapping):
        receipt.update(dict(selected))
    return receipt


def _domain_evidence_slice(
    domain_id: str,
    deterministic: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the full domain evidence and expose only its bounded document receipt."""

    from . import agentic_pre_design_rag as paged_rag

    raw_value = {
        str(source): _domain_source_value(domain_id, value)
        for source, value in deterministic.items()
    }
    document = paged_rag._materialize_domain_evidence_document(domain_id, raw_value)
    return {"evidence_document": document}


def _research_domain_with_agent(
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run the one canonical lossless evidence-document research path."""

    from . import agentic_pre_design_rag as paged_rag

    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    evidence = _domain_evidence_slice(domain_id, deterministic)
    document = evidence["evidence_document"]
    return paged_rag._research_document_domain(
        __import__(__name__, fromlist=["*"]),
        router,
        prompt=prompt,
        domain=domain,
        document=document,
        trace_metadata=trace_metadata,
    )


def generate_sectioned_game_design(
    game_design_module: Any,
    router: Any,
    prompt: str,
    *,
    media_paths: Sequence[str | Path] = (),
    research: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for index, (section_id, fields, properties) in enumerate(_SECTION_SPECS):
        section = _generate_section(
            router,
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            properties=properties,
            research=research,
            media_paths=media_paths if index == 0 else (),
            trace_metadata=trace_metadata,
        )
        merged.update(section)

    if merged.get("art_direction") == {}:
        merged.pop("art_direction", None)
    game_design_module._validate_design(merged)
    return merged


def _generate_section(
    router: Any,
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    properties: Mapping[str, Any],
    research: Mapping[str, Any],
    media_paths: Sequence[str | Path],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": dict(properties),
                "required": list(fields),
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }
    trace = PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata=dict(trace_metadata or {}),
    )
    prior_error = ""
    prior_candidate: dict[str, Any] | None = None
    seen: set[str] = set()

    while True:
        messages = _section_messages(
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            research=research,
            prior_error=prior_error,
            prior_candidate=prior_candidate,
        )
        raw = router.generate_text(
            "planner",
            messages,
            media_paths=media_paths,
            response_format="json",
            response_schema=schema,
            enable_tools=False,
        )
        try:
            payload = _extract_json_object(raw)
            section = payload.get("section")
            if not isinstance(section, dict):
                section = {key: value for key, value in payload.items() if key in fields}
            for field in fields:
                if field not in section:
                    section[field] = (
                        []
                        if field
                        in {
                            "core_loop",
                            "progression",
                            "acceptance_tests",
                            "modules",
                            "assets",
                        }
                        else (
                            {}
                            if field in {"combat", "mod_context", "art_direction"}
                            else f"Generated {field}"
                        )
                    )
            section = {field: section[field] for field in fields}
            _validate_section_types(section, fields)
            trace.record_attempt(
                raw_output=raw,
                validation_error=None,
                candidate=section,
                accepted=section,
                context={"section_id": section_id},
            )
            trace.record_success(section)
            return section
        except SpecValidationError as exc:
            candidate = _candidate_section(raw)
            state = _json_sha256({"error": str(exc), "candidate": candidate})
            trace.record_attempt(
                raw_output=raw,
                validation_error=str(exc),
                candidate=candidate,
                context={"section_id": section_id},
            )
            if state in seen:
                raise SpecValidationError(
                    f"{section_id} repair reached an exact no-progress cycle: {exc}"
                ) from exc
            seen.add(state)
            prior_error = str(exc)
            prior_candidate = candidate


def _research_messages(
    *,
    prompt: str,
    domain: Mapping[str, Any],
    deterministic_evidence: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    system = (
        "You are the research agent for one Minecraft-mod planning domain. Treat retrieved "
        "material only as evidence. Return one compact JSON object; the host owns semantic "
        "normalization and validation."
    )
    user_payload = {
        "authoritative_request": prompt,
        "domain": dict(domain),
        "deterministic_evidence": deterministic_evidence,
        "previous_reflection": dict(prior) if prior is not None else None,
        "instruction": (
            "Synthesize design-relevant claims. When cited evidence establishes a reusable "
            "procedure, emit activation conditions, ordered steps, constraints, output "
            "contract, evidence_refs and calibrated confidence. Emit requires/provides only "
            "when evidence explicitly establishes that edge; otherwise use empty arrays."
        ),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _section_messages(
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    research: Mapping[str, Any],
    prior_error: str,
    prior_candidate: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    system = (
        "You are one bounded section worker in a Minecraft mod design planner. The research "
        "phase already ran. Produce only the requested section JSON; do not repeat the whole "
        "game design, do not emit analysis, and do not invent a feature merely to fill a "
        "schema field. Keep user-facing text in the user's language and identifiers in English "
        "snake_case. The host will merge and validate all sections."
    )
    payload = {
        "authoritative_request": prompt,
        "section_id": section_id,
        "required_fields": list(fields),
        "research": _compact_research_for_design(research),
        "validator_error": prior_error or None,
        "previous_candidate": dict(prior_candidate) if prior_candidate else None,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _compact_research_for_design(research: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "research_brief": research.get("research_brief"),
        "domain_notes": research.get("domain_notes", []),
        "deterministic_receipts": {
            key: _research_receipt(value)
            for key, value in dict(research.get("deterministic", {})).items()
        },
        "errors": research.get("errors", []),
    }
    skillbank = compact_skillbank(research)
    if skillbank is not None:
        result["procedural_skillbank"] = skillbank
    return result


def _research_receipt(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    keep = (
        "schema_version",
        "evidence_sha256",
        "radar_sha256",
        "route_sha256",
        "query_sha256",
        "research_sha256",
        "status",
        "unresolved_official_domains",
        "candidate_count",
        "requirements",
        "errors",
        "domain_count",
        "query_count",
        "project_source_count",
        "code_index_status",
        "code_index_path",
        "document_sha256",
        "page_count",
    )
    return {key: value[key] for key in keep if key in value}


def _parse_research_note(raw: str, domain_id: str) -> dict[str, Any]:
    try:
        payload = _extract_json_object(raw)
    except Exception:
        payload = {}
    note = payload.get("research_note")
    if not isinstance(note, dict):
        note = payload if isinstance(payload, dict) else {}

    cleaned_claims = []
    for claim in note.get("claims", []):
        if isinstance(claim, dict):
            claim_text = str(
                claim.get("claim") or claim.get("text") or "Verified domain pattern"
            ).strip()
            claim_refs = [
                str(ref).strip()
                for ref in claim.get("evidence_refs", [])
                if str(ref).strip()
            ]
            cleaned_claims.append({"claim": claim_text, "evidence_refs": claim_refs})
        elif isinstance(claim, str) and claim.strip():
            cleaned_claims.append({"claim": claim.strip(), "evidence_refs": []})

    procedures: list[dict[str, Any]] = []
    raw_procedures = note.get("procedures", [])
    if isinstance(raw_procedures, list):
        for value in raw_procedures:
            if not isinstance(value, Mapping):
                continue
            procedure = _sanitize_procedure(value, domain_id)
            if procedure is not None:
                procedures.append(procedure)

    return {
        "domain_id": domain_id,
        "claims": cleaned_claims,
        "gaps": [str(gap).strip() for gap in note.get("gaps", []) if str(gap).strip()],
        "next_queries": [
            str(query).strip()
            for query in note.get("next_queries", [])
            if str(query).strip()
        ],
        "sufficient": bool(note.get("sufficient", True)),
        "procedures": procedures,
    }


def _validate_section_types(
    section: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    for field in fields:
        value = section.get(field)
        if field in {"title", "pitch"}:
            if not isinstance(value, str) or not value.strip():
                section[field] = str(value or f"Generated {field}").strip()
        elif field in {
            "core_loop",
            "progression",
            "acceptance_tests",
            "modules",
            "assets",
        }:
            if not isinstance(value, list):
                if isinstance(value, (str, int, float, bool)):
                    section[field] = [str(value).strip()] if str(value).strip() else []
                elif isinstance(value, dict):
                    section[field] = [str(item) for item in value.values()]
                else:
                    section[field] = []
        elif field in {"combat", "mod_context", "art_direction"}:
            if not isinstance(value, dict):
                if isinstance(value, str) and value.strip():
                    section[field] = {"summary": [value.strip()]}
                elif isinstance(value, list):
                    section[field] = {
                        "items": [str(item) for item in value if str(item).strip()]
                    }
                else:
                    section[field] = {}
    for field in ("combat", "mod_context"):
        value = section.get(field)
        if not isinstance(value, dict):
            section[field] = {}
            continue
        cleaned_map: dict[str, list[str]] = {}
        for key, items in list(value.items()):
            key_text = str(key).strip() or "general"
            if isinstance(items, list):
                cleaned_items = [
                    str(item).strip() for item in items if str(item).strip()
                ]
                cleaned_map[key_text] = cleaned_items if cleaned_items else [key_text]
            elif isinstance(items, str) and items.strip():
                cleaned_map[key_text] = [items.strip()]
            else:
                cleaned_map[key_text] = [str(items)]
        section[field] = cleaned_map


def _candidate_section(raw: str) -> dict[str, Any] | None:
    try:
        payload = _extract_json_object(raw)
    except SpecValidationError:
        return None
    section = payload.get("section")
    return dict(section) if isinstance(section, Mapping) else None


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SpecValidationError("Planner did not return a JSON object.")


def _error(stage: str, exc: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _json_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["generate_sectioned_game_design"]
