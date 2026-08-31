from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .external_procedural_skill_contract import _sanitize_procedure, compact_skillbank
from .planner_stage_trace import PlannerStageTrace
from .spec import SpecValidationError

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

# These are host-side type contracts. They are deliberately NOT sent to the model as a
# response schema. Design drafting is prose/Markdown; structured values are produced by the
# host only after generation.
_SECTION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity_and_loop", ("title", "pitch", "core_loop")),
    ("systems_and_progression", ("progression", "combat", "mod_context")),
    ("modules_and_assets", ("modules", "assets")),
    ("quality_and_art", ("acceptance_tests", "art_direction")),
)

_LIST_FIELDS = frozenset({"core_loop", "progression", "acceptance_tests"})
_MAP_FIELDS = frozenset({"combat", "mod_context", "art_direction"})


def supports_agentic_research_router(router: Any) -> bool:
    from .model_router import ModelRouter

    return isinstance(router, ModelRouter)


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
    domain_id: str,
    deterministic: Mapping[str, Any],
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
    note: Mapping[str, Any],
    *,
    allowed_refs: frozenset[str],
) -> None:
    if not note.get("sufficient"):
        return
    claims = note.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise SpecValidationError(
            "research_note.sufficient=true requires at least one grounded claim"
        )
    if not allowed_refs:
        raise SpecValidationError(
            "research_note.sufficient=true is forbidden because the host has issued no "
            "grounding evidence_ref for this domain"
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
                f"research_note.claims[{index}] cites unverified evidence_refs {unknown}; "
                f"allowed host refs are {sorted(allowed_refs)}"
            )


def _research_domain_with_agent(
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Research one target-neutral domain and fail closed on ungrounded claims.

    Research notes remain structured because they are evidence/accounting records, not the
    creative game-design drafting path.
    """

    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    evidence = _domain_evidence_slice(domain_id, deterministic)
    allowed_refs = _allowed_research_refs(evidence)
    trace = PlannerStageTrace(
        stage="pre_design_research",
        prompt=prompt,
        metadata={"domain_id": domain_id, **dict(trace_metadata or {})},
    )
    prior: dict[str, Any] | None = None
    seen_frontiers: set[frozenset[str]] = set()

    while True:
        raw = router.generate_text(
            "planner",
            _research_messages(
                prompt=prompt,
                domain=domain,
                deterministic_evidence=evidence,
                prior=prior,
            ),
            response_format="json",
            response_schema=_RESEARCH_NOTE_SCHEMA,
            tool_stage="research",
            enable_tools=True,
        )
        try:
            note = _parse_research_note(raw, domain_id)
            _validate_sufficient_research(note, allowed_refs=allowed_refs)
        except SpecValidationError as exc:
            candidate = _candidate_research_note(raw, domain_id)
            frontier = (
                frozenset(_claim_refs(candidate) & allowed_refs)
                if isinstance(candidate, Mapping)
                else frozenset()
            )
            trace.record_attempt(
                raw_output=raw,
                validation_error=str(exc),
                candidate=candidate,
                context={
                    "domain_id": domain_id,
                    "allowed_evidence_refs": sorted(allowed_refs),
                },
            )
            if frontier in seen_frontiers:
                return {
                    "domain_id": domain_id,
                    "claims": [],
                    "gaps": [str(exc)],
                    "next_queries": list(domain.get("queries", [])),
                    "procedures": [],
                    "sufficient": False,
                    "fixed_point": True,
                }
            seen_frontiers.add(frontier)
            prior = {
                "domain_id": domain_id,
                "claims": [],
                "gaps": [str(exc)],
                "next_queries": list(domain.get("queries", [])),
                "procedures": [],
                "sufficient": False,
                "allowed_evidence_refs": sorted(allowed_refs),
            }
            continue

        trace.record_attempt(
            raw_output=raw,
            validation_error=None,
            candidate=note,
            accepted=note if note["sufficient"] else None,
            context={
                "domain_id": domain_id,
                "allowed_evidence_refs": sorted(allowed_refs),
            },
        )
        if note["sufficient"]:
            trace.record_success(note)
            return note

        frontier = _claim_refs(note) & allowed_refs
        if frontier in seen_frontiers:
            return {**note, "fixed_point": True}
        seen_frontiers.add(frontier)
        prior = note


def generate_sectioned_game_design(
    game_design_module: Any,
    router: Any,
    prompt: str,
    *,
    media_paths: Sequence[str | Path] = (),
    research: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate design prose in bounded sections and structure it only on the host.

    There is intentionally no model-side JSON schema and no model repair loop here.
    """

    merged: dict[str, Any] = {}
    for index, (section_id, fields) in enumerate(_SECTION_SPECS):
        section = _generate_section(
            router,
            prompt=prompt,
            section_id=section_id,
            fields=fields,
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
    research: Mapping[str, Any],
    media_paths: Sequence[str | Path],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trace = PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata=dict(trace_metadata or {}),
    )
    raw = router.generate_text(
        "planner",
        _section_messages(
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            research=research,
        ),
        media_paths=media_paths,
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    try:
        section = _parse_markdown_section(raw, fields)
    except SpecValidationError as exc:
        trace.record_attempt(
            raw_output=raw,
            validation_error=str(exc),
            candidate=None,
            context={"section_id": section_id, "format": "markdown"},
        )
        raise

    trace.record_attempt(
        raw_output=raw,
        validation_error=None,
        candidate=section,
        accepted=section,
        context={"section_id": section_id, "format": "markdown"},
    )
    trace.record_success(section)
    return section


def _normalize_heading(value: str) -> str:
    value = value.strip().strip("`").casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def _parse_markdown_section(raw: str, fields: Sequence[str]) -> dict[str, Any]:
    expected = {_normalize_heading(field): field for field in fields}
    bodies: dict[str, list[str]] = {}
    current: str | None = None

    for line in str(raw or "").splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match:
            heading = expected.get(_normalize_heading(match.group(1)))
            current = heading
            if current is not None:
                bodies.setdefault(current, [])
            continue
        if current is not None:
            bodies[current].append(line)

    missing = [field for field in fields if field not in bodies]
    if missing:
        raise SpecValidationError(
            "Planner prose omitted required Markdown heading(s): " + ", ".join(missing)
        )

    section: dict[str, Any] = {}
    for field in fields:
        body = "\n".join(bodies[field]).strip()
        if field in {"title", "pitch"}:
            value = _plain_text(body)
            if not value:
                raise SpecValidationError(f"Planner prose left ## {field} empty")
            section[field] = value
        elif field in _LIST_FIELDS:
            values = _markdown_list(body)
            if not values:
                raise SpecValidationError(f"Planner prose left ## {field} empty")
            section[field] = values
        elif field in _MAP_FIELDS:
            section[field] = _markdown_map(body)
        elif field == "modules":
            section[field] = _module_rows(body)
        elif field == "assets":
            section[field] = _asset_rows(body)
        else:
            raise SpecValidationError(f"Unsupported host design field: {field}")
    return section


def _plain_text(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        cleaned = _strip_list_marker(line)
        if cleaned:
            lines.append(cleaned)
    return " ".join(lines).strip()


def _strip_list_marker(line: str) -> str:
    value = line.strip()
    if not value:
        return ""
    return re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", value).strip()


def _markdown_list(body: str) -> list[str]:
    values: list[str] = []
    for line in body.splitlines():
        if line.lstrip().startswith("### "):
            continue
        value = _strip_list_marker(line)
        if value:
            values.append(value)
    return values


def _markdown_map(body: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    current = "summary"
    for line in body.splitlines():
        heading = re.match(r"^\s*###\s+(.+?)\s*$", line)
        if heading:
            current = _normalize_heading(heading.group(1)) or "summary"
            result.setdefault(current, [])
            continue
        value = _strip_list_marker(line)
        if value:
            result.setdefault(current, []).append(value)
    return {key: values for key, values in result.items() if values}


def _module_rows(body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in body.splitlines():
        value = _strip_list_marker(line)
        if not value or value.casefold() in {"none", "n/a", "없음"}:
            continue
        parts = [part.strip() for part in value.split("|")]
        if len(parts) < 3 or not all(parts[:3]):
            raise SpecValidationError(
                "Each ## modules row must be: plugin_id | status | reason"
            )
        rows.append(
            {
                "plugin_id": parts[0],
                "status": parts[1],
                "reason": " | ".join(parts[2:]),
            }
        )
    return rows


def _asset_rows(body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in body.splitlines():
        value = _strip_list_marker(line)
        if not value or value.casefold() in {"none", "n/a", "없음"}:
            continue
        parts = [part.strip() for part in value.split("|")]
        if len(parts) < 3 or not all(parts[:3]):
            raise SpecValidationError("Each ## assets row must be: id | kind | brief")
        rows.append(
            {
                "id": parts[0],
                "kind": parts[1],
                "brief": " | ".join(parts[2:]),
            }
        )
    return rows


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


def _section_messages(
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    research: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are one bounded section worker in a Minecraft mod design planner. The research "
        "phase already ran. Write design content as Markdown, not JSON and not a code block. "
        "Use exactly one level-2 heading for every requested field, spelled exactly as given, "
        "and no other level-2 headings. Do not repeat the whole design and do not invent a "
        "feature merely to fill a field. User-facing text stays in the user's language; "
        "identifiers stay English snake_case. For combat/mod_context/art_direction, optional "
        "level-3 subheadings are allowed. For modules use one bullet per row as "
        "plugin_id | status | reason. For assets use id | kind | brief. Use 'none' when a "
        "modules/assets list is intentionally empty."
    )
    research_text = _render_design_research(research)
    requested = "\n".join(f"- ## {field}" for field in fields)
    user = (
        f"AUTHORITATIVE REQUEST\n{prompt}\n\n"
        f"SECTION\n{section_id}\n\n"
        f"REQUIRED HEADINGS\n{requested}\n\n"
        f"RESEARCH CONTEXT\n{research_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
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


__all__ = ["generate_sectioned_game_design"]
