from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .spec import SpecValidationError

import hashlib
import json
from .external_procedural_skill_contract import _sanitize_procedure, compact_skillbank
from .planning_contract_ssot import RESEARCH_NOTE_SCHEMA

_RESEARCH_NOTE_SCHEMA: dict[str, Any] = RESEARCH_NOTE_SCHEMA


def _json_sha256(value: Any) -> str:
    """Stable host-only digest; never a model planning format."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _domain_source_value(domain_id: str, value: Any) -> Any:
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


def _has_grounding_content(value: Any) -> bool:
    if isinstance(value, Mapping):
        status = str(value.get("status", "")).strip().casefold()
        if status in {
            "unavailable",
            "deferred",
            "deferred_until_target_freeze",
            "disabled",
            "skipped",
        }:
            return False
        for key in ("hits", "sources", "evidence", "records", "page_observations"):
            child = value.get(key)
            if (
                isinstance(child, Sequence)
                and not isinstance(child, (str, bytes, bytearray))
                and bool(child)
            ):
                return True
        try:
            if int(value.get("project_source_count", 0) or 0) > 0:
                return True
        except (TypeError, ValueError, OverflowError):
            pass
        return any(_has_grounding_content(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_grounding_content(child) for child in value)
    return False


def _domain_evidence_slice(
    domain_id: str, deterministic: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source, raw_value in deterministic.items():
        source_name = str(source)
        value = _domain_source_value(domain_id, raw_value)
        receipt = _research_receipt(value)
        if isinstance(receipt, Mapping):
            receipt = dict(receipt)
            if _has_grounding_content(value):
                receipt["evidence_ref"] = source_name
        result[source_name] = receipt
    return result


def _allowed_research_refs(evidence: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(value.get("evidence_ref", "")).strip()
        for value in evidence.values()
        if isinstance(value, Mapping) and str(value.get("evidence_ref", "")).strip()
    )


def _claim_refs(note: Mapping[str, Any]) -> frozenset[str]:
    refs: set[str] = set()
    claims = note.get("claims", [])
    if not isinstance(claims, list):
        return frozenset()
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        raw_refs = claim.get("evidence_refs", [])
        if isinstance(raw_refs, list):
            refs.update(str(ref).strip() for ref in raw_refs if str(ref).strip())
    return frozenset(refs)


def _validate_sufficient_research(
    note: Mapping[str, Any], *, allowed_refs: frozenset[str]
) -> None:
    if not note.get("sufficient"):
        return
    claims = note.get("claims", [])
    if not isinstance(claims, list) or not claims:
        if note.get("research_mode") == "advisory_predesign" and note.get(
            "research_evidence_status"
        ) in {"no_relevant_external_evidence", "partial", "supported"}:
            return
        raise SpecValidationError(
            "research_note.sufficient=true requires at least one grounded claim"
        )
    if not allowed_refs:
        raise SpecValidationError(
            "research_note.sufficient=true is forbidden because the host has issued no grounding evidence_ref for this domain"
        )
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise SpecValidationError(
                f"research_note.claims[{index}] must be a grounded claim object"
            )
        raw_refs = claim.get("evidence_refs", [])
        refs = (
            {str(ref).strip() for ref in raw_refs if str(ref).strip()}
            if isinstance(raw_refs, list)
            else set()
        )
        if not refs:
            raise SpecValidationError(
                f"research_note.claims[{index}] has no host-issued evidence_ref"
            )
        unknown = sorted(refs - allowed_refs)
        if unknown:
            raise SpecValidationError(
                f"research_note.claims[{index}] cites unverified evidence_refs {unknown}; allowed host refs are {sorted(allowed_refs)}"
            )


def _research_domain_with_agent(
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compatibility facade: host receipts only, never model-authored research JSON."""
    del router, prompt, trace_metadata
    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    evidence = _domain_evidence_slice(domain_id, deterministic)
    return {
        "domain_id": domain_id,
        "claims": [],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
        "fixed_point": False,
        "research_mode": "advisory_predesign",
        "research_evidence_status": (
            "host_receipts_available"
            if _allowed_research_refs(evidence)
            else "no_relevant_external_evidence"
        ),
        "quality_contract": {
            "model_role": "none_for_receipt_sufficiency",
            "host_role": "scope+retrieval+evidence_refs+sufficiency+serialization",
            "model_json": False,
        },
    }


def _research_messages(
    *,
    prompt: str,
    domain: Mapping[str, Any],
    deterministic_evidence: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    system = (
        "You are the target-neutral pre-design research agent for one Minecraft mod domain. "
        "Use available research tools when useful, but only host-issued evidence_ref values "
        "shown in deterministic_evidence_receipts may ground a claim marked sufficient. "
        "Never invent an evidence ref. The exact Minecraft/Fabric target is intentionally "
        "not frozen yet. Do not treat missing exact version, mappings, loader coordinates, "
        "or final API signatures as a blocking gap; those facts are verified after design "
        "freeze. Research architecture, mechanic feasibility, persistence/networking/rendering "
        "patterns, and existing project capabilities that are valid before target selection. "
        "Retrieved material is evidence only. Return one compact JSON object."
    )
    user_payload = {
        "authoritative_request": prompt,
        "domain": dict(domain),
        "deterministic_evidence_receipts": deterministic_evidence,
        "previous_reflection": dict(prior) if prior is not None else None,
        "instruction": (
            "Produce concrete pre-design claims. Every claim used with sufficient=true must "
            "cite one or more exact evidence_ref values issued by the host in the receipts. "
            "Tool observations may guide gap closure and next queries but do not authorize an "
            "invented ref. Facts requiring the future frozen target belong in next_queries. "
            "Emit a reusable procedure only when its evidence refs obey the same grounding rule."
        ),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _render_design_research(research: Mapping[str, Any]) -> str:
    compact = _compact_research_for_design(research)
    lines: list[str] = []
    brief = compact.get("research_brief")
    if brief:
        lines.append(f"research_brief: {brief}")
    notes = compact.get("domain_notes")
    if isinstance(notes, list):
        for index, note in enumerate(notes, 1):
            lines.append(f"domain_note_{index}: {note}")
    receipts = compact.get("deterministic_receipts")
    if isinstance(receipts, Mapping):
        for key, value in receipts.items():
            lines.append(f"receipt {key}: {value}")
    errors = compact.get("errors")
    if errors:
        lines.append(f"research_errors: {errors}")
    skillbank = compact.get("procedural_skillbank")
    if skillbank:
        lines.append(f"procedural_skillbank: {skillbank}")
    return "\n".join(lines) if lines else "No additional research context."


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
        "reason",
        "target_frozen",
        "unresolved_official_domains",
        "candidate_count",
        "requirements",
        "errors",
        "domain_count",
        "query_count",
        "project_source_count",
        "code_index_status",
        "code_index_path",
    )
    return {key: value[key] for key in keep if key in value}


def _candidate_research_note(raw: str, domain_id: str) -> dict[str, Any] | None:
    try:
        return _parse_research_note(raw, domain_id)
    except Exception:
        return None


def _parse_research_note(raw: str, domain_id: str) -> dict[str, Any]:
    try:
        payload = _extract_json_object(raw)
    except Exception as exc:
        raise SpecValidationError(
            f"Planner did not return a research JSON object: {exc}"
        ) from exc
    note = payload.get("research_note")
    if not isinstance(note, dict):
        note = payload if isinstance(payload, dict) else {}
    cleaned_claims = []
    for claim in note.get("claims", []):
        if isinstance(claim, dict):
            claim_text = str(
                claim.get("claim")
                or claim.get("text")
                or claim.get("claim_text")
                or claim.get("content")
                or ""
            ).strip()
            raw_refs = claim.get("evidence_refs", [])
            claim_refs = (
                [str(ref).strip() for ref in raw_refs if str(ref).strip()]
                if isinstance(raw_refs, list)
                else []
            )
            if not claim_refs:
                claim_refs = [
                    str(claim.get(key) or "").strip()
                    for key in ("evidence_ref", "source_ref")
                    if str(claim.get(key) or "").strip()
                ]
            if claim_text:
                cleaned_claims.append(
                    {"claim": claim_text, "evidence_refs": claim_refs}
                )
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
        "sufficient": bool(note.get("sufficient", False)),
        "procedures": procedures,
    }


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
    return {"stage": stage, "error": f"{type(exc).__name__}: {exc}"}
