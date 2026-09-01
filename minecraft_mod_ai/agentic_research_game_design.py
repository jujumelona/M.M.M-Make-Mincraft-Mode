from __future__ import annotations

import hashlib
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

# One host-owned contract defines model instructions, Markdown parsing, host structure,
# and validation. Runtime installers must not mutate this schema or replace its parser.
_SECTION_SPECS: tuple[tuple[str, tuple[str, ...], dict[str, Any]], ...] = (
    (
        "identity_and_loop",
        ("title", "pitch", "core_loop"),
        {
            "title": {"type": "string", "minLength": 1},
            "pitch": {"type": "string", "minLength": 1},
            "core_loop": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
    ),
    (
        "systems_and_progression",
        ("progression", "combat", "mod_context"),
        {
            "progression": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "combat": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string", "minLength": 1}}},
            "mod_context": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string", "minLength": 1}}},
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
                        "requirement_refs": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "uniqueItems": True,
                        },
                        "implementation_obligations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                            "uniqueItems": True,
                        },
                    },
                    "required": [
                        "plugin_id",
                        "status",
                        "reason",
                        "requirement_refs",
                        "implementation_obligations",
                    ],
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
            "acceptance_tests": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "art_direction": {"type": "object"},
        },
    ),
)

_LIST_FIELDS = frozenset({"core_loop", "progression", "acceptance_tests"})
_MAP_FIELDS = frozenset({"combat", "mod_context", "art_direction"})
_NONE_VALUES = frozenset({"none", "n/a", "없음"})

_PRODUCTION_DEPTH = (
    "PRODUCTION DEPTH: finish the game/mod design before implementation search. "
    "Decompose every requested mechanic into the smallest meaningful subsystems that can "
    "be independently implemented, tested, and searched for reuse. Split different player "
    "verbs, resources, state transitions, purchase/assembly steps, upgrade gates, travel "
    "phases, encounters, combat outcomes, world interactions, persistence-visible state, "
    "networking/client surfaces, and integration rules when they can fail independently. "
    "The modules section is the implementation-leaf index: every implementation-bearing "
    "core-loop/progression/combat/mod-context behavior must have a concrete modules row. "
    "Do not collapse an epic such as planet interaction, ship construction, trading, or "
    "progression into one generic module. Use as many leaf modules as the authored design "
    "genuinely needs; never add unrelated features. Use supplied research evidence for "
    "Minecraft/Fabric facts and unresolved assumptions, but donor/reuse selection happens "
    "only after this design is frozen."
)

_MODULE_FORMAT = (
    "For ## modules, prefer one Markdown record per module instead of a fragile fixed-width "
    "table. Use: ### <plugin_id>, then '- status: <value>', '- reason: <text>', "
    "'- requirement_refs: <comma-separated exact approved IDs>', and "
    "'- implementation_obligations:' followed by one or more nested bullets. "
    "A legacy one-line pipe record is also accepted as "
    "plugin_id | status | reason | requirement_refs | implementation_obligations, and the "
    "reason may itself contain pipe characters. requirement_refs must preserve exact approved "
    "requirement IDs (or literal 'none' only when no approved requirements exist). Never hide "
    "requirement_refs or implementation_obligations inside reason."
)

_ASSET_FORMAT = (
    "For ## assets, prefer one Markdown record per asset: ### <id>, then '- kind: <kind>' and "
    "'- brief: <description>'. A legacy id | kind | brief line is also accepted; brief may "
    "contain pipe characters."
)


def supports_agentic_research_router(router: Any) -> bool:
    from .model_router import ModelRouter

    return isinstance(router, ModelRouter)


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


def _active_requirement_ledger(prompt: str) -> tuple[dict[str, Any], ...]:
    """Read the already-frozen authored request authority without rebuilding scope."""
    from . import evidence_request_guard as request_guard

    active = request_guard._ACTIVE_REQUEST_CATALOG.get()
    if active is None or active[0] != prompt:
        return ()
    catalog = active[1]
    raw_requirements = catalog.get("requirements", [])
    if not isinstance(raw_requirements, list):
        return ()
    ledger: list[dict[str, Any]] = []
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            continue
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        span = raw.get("source_span")
        span_text = str(span.get("text") or "").strip() if isinstance(span, Mapping) else ""
        behavior = raw.get("observable_behavior")
        acceptance = raw.get("acceptance")
        ledger.append(
            {
                "requirement_id": requirement_id,
                "capability": str(raw.get("capability") or "").strip(),
                "authored_text": span_text or str(raw.get("statement") or "").strip(),
                "semantic_statement": str(raw.get("semantic_statement") or "").strip(),
                "observable_behavior": dict(behavior) if isinstance(behavior, Mapping) else {},
                "acceptance": [str(item).strip() for item in acceptance if str(item).strip()]
                if isinstance(acceptance, list)
                else [],
            }
        )
    return tuple(ledger)


def _render_requirement_ledger(ledger: Sequence[Mapping[str, Any]]) -> str:
    if not ledger:
        return "No approved requirement ledger is active."
    lines = ["APPROVED REQUIREMENTS (HOST AUTHORITY; preserve IDs exactly)"]
    for item in ledger:
        requirement_id = " ".join(str(item.get("requirement_id") or "").split())
        lines.append(f"- requirement_id: {requirement_id}")
        capability = " ".join(str(item.get("capability") or "").split())
        if capability:
            lines.append(f"  capability: {capability}")
        authored = " ".join(str(item.get("authored_text") or "").split())
        if authored:
            lines.append(f"  authored_text: {authored}")
        semantic = " ".join(str(item.get("semantic_statement") or "").split())
        if semantic:
            lines.append(f"  semantic_statement: {semantic}")
        acceptance = item.get("acceptance")
        if isinstance(acceptance, list):
            rendered = "; ".join(" ".join(str(value).split()) for value in acceptance if str(value).strip())
            if rendered:
                lines.append(f"  acceptance: {rendered}")
    return "\n".join(lines)


def _domain_source_value(domain_id: str, value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    domains = value.get("domains")
    if not isinstance(domains, list):
        return dict(value)
    selected = next(
        (item for item in domains if isinstance(item, Mapping) and item.get("domain_id") == domain_id),
        None,
    )
    receipt = {key: item for key, item in value.items() if key != "domains"}
    if isinstance(selected, Mapping):
        receipt.update(dict(selected))
    return receipt


def _has_grounding_content(value: Any) -> bool:
    if isinstance(value, Mapping):
        status = str(value.get("status", "")).strip().casefold()
        if status in {"unavailable", "deferred", "deferred_until_target_freeze", "disabled", "skipped"}:
            return False
        for key in ("hits", "sources", "evidence", "records", "page_observations"):
            child = value.get(key)
            if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)) and bool(child):
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


def _domain_evidence_slice(domain_id: str, deterministic: Mapping[str, Any]) -> dict[str, Any]:
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


def _validate_sufficient_research(note: Mapping[str, Any], *, allowed_refs: frozenset[str]) -> None:
    if not note.get("sufficient"):
        return
    claims = note.get("claims", [])
    if not isinstance(claims, list) or not claims:
        if (
            note.get("research_mode") == "advisory_predesign"
            and note.get("research_evidence_status")
            in {"no_relevant_external_evidence", "partial", "supported"}
        ):
            return
        raise SpecValidationError("research_note.sufficient=true requires at least one grounded claim")
    if not allowed_refs:
        raise SpecValidationError(
            "research_note.sufficient=true is forbidden because the host has issued no grounding evidence_ref for this domain"
        )
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise SpecValidationError(f"research_note.claims[{index}] must be a grounded claim object")
        raw_refs = claim.get("evidence_refs", [])
        refs = {str(ref).strip() for ref in raw_refs if str(ref).strip()} if isinstance(raw_refs, list) else set()
        if not refs:
            raise SpecValidationError(f"research_note.claims[{index}] has no host-issued evidence_ref")
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
            _research_messages(prompt=prompt, domain=domain, deterministic_evidence=evidence, prior=prior),
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
            frontier = frozenset(_claim_refs(candidate) & allowed_refs) if isinstance(candidate, Mapping) else frozenset()
            trace.record_attempt(
                raw_output=raw,
                validation_error=str(exc),
                candidate=candidate,
                context={"domain_id": domain_id, "allowed_evidence_refs": sorted(allowed_refs)},
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
            context={"domain_id": domain_id, "allowed_evidence_refs": sorted(allowed_refs)},
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
    """Generate bounded structured sections, then validate requirement coverage."""
    merged: dict[str, Any] = {}
    for index, (section_id, fields, host_properties) in enumerate(_SECTION_SPECS):
        section = _generate_section(
            router,
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            host_properties=host_properties,
            research=research,
            media_paths=media_paths if index == 0 else (),
            trace_metadata=trace_metadata,
        )
        merged.update(section)
    if merged.get("art_direction") == {}:
        merged.pop("art_direction", None)
    game_design_module._validate_design(merged)
    return _validate_requirement_coverage(merged, _active_requirement_ledger(prompt))


def _generate_section(
    router: Any,
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    host_properties: Mapping[str, Any] | None = None,
    research: Mapping[str, Any],
    media_paths: Sequence[str | Path],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del host_properties
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
        tool_stage="game_design",
        enable_tools=False,
    )
    try:
        section = _parse_markdown_section(raw, fields)
        requirement_ids = tuple(
            item["requirement_id"] for item in _active_requirement_ledger(prompt)
        )
        _validate_section_types(section, fields, requirement_ids=requirement_ids)
    except (KeyError, SpecValidationError, ValueError, TypeError) as exc:
        trace.record_attempt(
            raw_output=raw,
            validation_error=str(exc),
            candidate=None,
            context={"section_id": section_id, "format": "host_parsed_markdown"},
        )
        raise
    trace.record_attempt(
        raw_output=raw,
        validation_error=None,
        candidate=section,
        accepted=section,
        context={"section_id": section_id, "format": "host_parsed_markdown"},
    )
    trace.record_success(section)
    return section
def _normalize_heading(value: str) -> str:
    value = value.strip().strip("`").casefold()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _parse_markdown_section(raw: str, fields: Sequence[str]) -> dict[str, Any]:
    expected = {_normalize_heading(field): field for field in fields}
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in str(raw or "").splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match:
            current = expected.get(_normalize_heading(match.group(1)))
            if current is not None:
                bodies.setdefault(current, [])
            continue
        if current is not None:
            bodies[current].append(line)
    missing = [field for field in fields if field not in bodies]
    if missing:
        raise SpecValidationError("Planner prose omitted required Markdown heading(s): " + ", ".join(missing))
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
    lines = [_strip_list_marker(line) for line in body.splitlines()]
    return " ".join(line for line in lines if line).strip()


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
    normalized = " ".join(_strip_list_marker(line) for line in body.splitlines() if _strip_list_marker(line)).strip().casefold()
    if normalized in _NONE_VALUES:
        return {}
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


def _split_csv(value: str) -> list[str]:
    text = value.strip().strip("[](){}")
    if text.casefold() in _NONE_VALUES:
        return []
    return list(
        dict.fromkeys(
            item.strip().strip("`'\"")
            for item in re.split(r"\s*[,;，；]\s*", text)
            if item.strip().strip("`'\"")
        )
    )


def _split_obligations(value: str) -> list[str]:
    text = value.strip().strip("[]")
    if text.casefold() in _NONE_VALUES:
        return []
    separators = r"\s*(?:;|；|<br\s*/?>)\s*"
    parts = [item.strip().strip("`'\"") for item in re.split(separators, text, flags=re.IGNORECASE)]
    return list(dict.fromkeys(item for item in parts if item))


def _record_key_value(value: str) -> tuple[str, str] | None:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_ -]*)\s*:\s*(.*)$", value.strip())
    if not match:
        return None
    key = _normalize_heading(match.group(1))
    return key, match.group(2).strip()


def _pipe_parts(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [part.strip() for part in text.split("|")]


def _is_markdown_table_separator(parts: Sequence[str]) -> bool:
    return bool(parts) and all(re.fullmatch(r":?-{3,}:?", part.replace(" ", "")) for part in parts if part)


def _finalize_module_record(record: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    plugin_id = str(record.get("plugin_id") or "").strip()
    status = str(record.get("status") or "").strip()
    reason = str(record.get("reason") or "").strip()
    raw_refs = record.get("requirement_refs")
    refs = list(raw_refs) if isinstance(raw_refs, list) else _split_csv(str(raw_refs or ""))
    raw_obligations = record.get("implementation_obligations")
    obligations = (
        [str(item).strip() for item in raw_obligations if str(item).strip()]
        if isinstance(raw_obligations, list)
        else _split_obligations(str(raw_obligations or ""))
    )
    obligations = list(dict.fromkeys(obligations))
    if not plugin_id or not status or not reason or not obligations:
        raise SpecValidationError(
            "Each ## modules record requires plugin_id, status, reason, requirement_refs, "
            f"and concrete implementation_obligations; malformed record: {source}"
        )
    return {
        "plugin_id": plugin_id,
        "status": status,
        "reason": reason,
        "requirement_refs": refs,
        "implementation_obligations": obligations,
    }


def _module_rows(body: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_list_key: str | None = None

    def flush() -> None:
        nonlocal current, active_list_key
        if current is not None:
            rows.append(_finalize_module_record(current, source=str(current)))
        current = None
        active_list_key = None

    for raw_line in body.splitlines():
        heading = re.match(r"^\s*###\s+(.+?)\s*$", raw_line)
        if heading:
            flush()
            heading_value = heading.group(1).strip().strip("`")
            heading_value = re.sub(r"^(?:module|plugin_id)\s*:\s*", "", heading_value, flags=re.IGNORECASE)
            if heading_value.casefold() in _NONE_VALUES:
                continue
            current = {"plugin_id": heading_value}
            continue

        value = _strip_list_marker(raw_line)
        if not value:
            continue
        if value.casefold() in _NONE_VALUES and current is None:
            continue

        parts = _pipe_parts(value)
        if "|" in value:
            normalized_header = [_normalize_heading(part) for part in parts]
            if normalized_header[:2] == ["plugin_id", "status"] or _is_markdown_table_separator(parts):
                continue
            if len(parts) >= 5 and all(parts[:2]) and parts[-2] and parts[-1]:
                flush()
                rows.append(
                    _finalize_module_record(
                        {
                            "plugin_id": parts[0],
                            "status": parts[1],
                            "reason": " | ".join(parts[2:-2]).strip(),
                            "requirement_refs": _split_csv(parts[-2]),
                            "implementation_obligations": _split_obligations(parts[-1]),
                        },
                        source=value,
                    )
                )
                continue

        key_value = _record_key_value(value)
        if key_value is not None:
            key, item_value = key_value
            if key == "plugin_id":
                if current is not None and current.get("plugin_id"):
                    flush()
                current = {"plugin_id": item_value}
                continue
            if current is None:
                current = {}
            if key in {"status", "reason"}:
                current[key] = item_value
                active_list_key = None
                continue
            if key == "requirement_refs":
                current[key] = _split_csv(item_value)
                active_list_key = "requirement_refs" if not item_value else None
                continue
            if key == "implementation_obligations":
                current[key] = _split_obligations(item_value)
                active_list_key = "implementation_obligations" if not item_value else None
                continue

        if current is not None and active_list_key == "implementation_obligations":
            current.setdefault("implementation_obligations", []).append(value)
            continue
        if current is not None and active_list_key == "requirement_refs":
            current.setdefault("requirement_refs", []).extend(_split_csv(value))
            continue
        if current is not None and current.get("reason"):
            current["reason"] = f"{current['reason']} {value}".strip()
            continue
        raise SpecValidationError(
            "Could not parse a ## modules record. Use labeled Markdown records or the supported legacy pipe record."
        )

    flush()
    return rows


def _finalize_asset_record(record: Mapping[str, Any], *, source: str) -> dict[str, str]:
    asset_id = str(record.get("id") or "").strip()
    kind = str(record.get("kind") or "").strip()
    brief = str(record.get("brief") or "").strip()
    if not asset_id or not kind or not brief:
        raise SpecValidationError(
            f"Each ## assets record requires id, kind, and brief; malformed record: {source}"
        )
    return {"id": asset_id, "kind": kind, "brief": brief}


def _asset_rows(body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            rows.append(_finalize_asset_record(current, source=str(current)))
        current = None

    for raw_line in body.splitlines():
        heading = re.match(r"^\s*###\s+(.+?)\s*$", raw_line)
        if heading:
            flush()
            heading_value = heading.group(1).strip().strip("`")
            heading_value = re.sub(r"^(?:asset|id)\s*:\s*", "", heading_value, flags=re.IGNORECASE)
            if heading_value.casefold() in _NONE_VALUES:
                continue
            current = {"id": heading_value}
            continue

        value = _strip_list_marker(raw_line)
        if not value:
            continue
        if value.casefold() in _NONE_VALUES and current is None:
            continue
        parts = _pipe_parts(value)
        if "|" in value:
            normalized_header = [_normalize_heading(part) for part in parts]
            if normalized_header[:2] == ["id", "kind"] or _is_markdown_table_separator(parts):
                continue
            if len(parts) >= 3 and all(parts[:2]):
                flush()
                rows.append(
                    _finalize_asset_record(
                        {"id": parts[0], "kind": parts[1], "brief": " | ".join(parts[2:]).strip()},
                        source=value,
                    )
                )
                continue

        key_value = _record_key_value(value)
        if key_value is not None:
            key, item_value = key_value
            if key == "id":
                if current is not None and current.get("id"):
                    flush()
                current = {"id": item_value}
                continue
            if current is None:
                current = {}
            if key in {"kind", "brief"}:
                current[key] = item_value
                continue
        if current is not None and current.get("brief"):
            current["brief"] = f"{current['brief']} {value}".strip()
            continue
        raise SpecValidationError(
            "Could not parse a ## assets record. Use labeled Markdown records or id | kind | brief."
        )

    flush()
    return rows


def _nonempty_text_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SpecValidationError(f"{field} must be a non-empty list; empty accepted design is forbidden")
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if len(cleaned) != len(value) or not cleaned:
        raise SpecValidationError(f"{field} must contain only non-empty authored design entries")
    return cleaned


def _validate_section_types(
    section: Mapping[str, Any],
    fields: Sequence[str],
    *,
    requirement_ids: Sequence[str] = (),
) -> None:
    required_ids = {str(value).strip() for value in requirement_ids if str(value).strip()}
    for field in fields:
        if field not in section:
            raise SpecValidationError(f"section omitted required field {field!r}")
        value = section.get(field)
        if field in {"title", "pitch"}:
            if not isinstance(value, str) or not value.strip():
                raise SpecValidationError(f"{field} must be a non-empty string")
        elif field in {"core_loop", "progression", "acceptance_tests"}:
            _nonempty_text_list(value, field=field)
        elif field == "assets":
            if not isinstance(value, list):
                raise SpecValidationError("assets must be a list")
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise SpecValidationError(f"assets[{index}] must be an object")
                for key in ("id", "kind", "brief"):
                    if not str(item.get(key) or "").strip():
                        raise SpecValidationError(f"assets[{index}].{key} must be non-empty")
        elif field == "modules":
            if not isinstance(value, list):
                raise SpecValidationError("modules must be a list")
            if required_ids and not value:
                raise SpecValidationError("modules must be non-empty while approved authored requirements exist")
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise SpecValidationError(f"modules[{index}] must be an object")
                for key in ("plugin_id", "status", "reason"):
                    if not str(item.get(key) or "").strip():
                        raise SpecValidationError(f"modules[{index}].{key} must be non-empty")
                _nonempty_text_list(item.get("implementation_obligations"), field=f"modules[{index}].implementation_obligations")
                refs = item.get("requirement_refs")
                if not isinstance(refs, list):
                    raise SpecValidationError(f"modules[{index}].requirement_refs must be a list")
                if required_ids:
                    refs = _nonempty_text_list(refs, field=f"modules[{index}].requirement_refs")
                    unknown = sorted(set(refs) - required_ids)
                    if unknown:
                        raise SpecValidationError("module cites unknown requirement ids: " + ", ".join(unknown))
        elif field in {"combat", "mod_context", "art_direction"}:
            if not isinstance(value, dict):
                raise SpecValidationError(f"{field} must be an object")
            if field in {"combat", "mod_context"}:
                for key, items in value.items():
                    if not str(key).strip():
                        raise SpecValidationError(f"{field} contains an empty key")
                    _nonempty_text_list(items, field=f"{field}.{key}")


def _validate_requirement_coverage(
    design: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_ids = tuple(
        str(item.get("requirement_id") or "").strip()
        for item in ledger
        if str(item.get("requirement_id") or "").strip()
    )
    if not required_ids:
        return dict(design)
    known = set(required_ids)
    modules = design.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SpecValidationError("design readiness failed: approved requirements exist but modules are empty")
    covered: set[str] = set()
    binding_rows = {
        requirement_id: {"requirement_id": requirement_id, "module_ids": [], "implementation_obligations": []}
        for requirement_id in required_ids
    }
    for index, item in enumerate(modules):
        if not isinstance(item, Mapping):
            raise SpecValidationError(f"modules[{index}] must be an object")
        module_id = str(item.get("plugin_id") or "").strip()
        refs = _nonempty_text_list(item.get("requirement_refs"), field=f"modules[{index}].requirement_refs")
        obligations = _nonempty_text_list(
            item.get("implementation_obligations"),
            field=f"modules[{index}].implementation_obligations",
        )
        unknown = sorted(set(refs) - known)
        if unknown:
            raise SpecValidationError("design readiness failed: unknown requirement refs " + ", ".join(unknown))
        for requirement_id in refs:
            covered.add(requirement_id)
            row = binding_rows[requirement_id]
            if module_id not in row["module_ids"]:
                row["module_ids"].append(module_id)
            for obligation in obligations:
                if obligation not in row["implementation_obligations"]:
                    row["implementation_obligations"].append(obligation)
    missing = [requirement_id for requirement_id in required_ids if requirement_id not in covered]
    if missing:
        raise SpecValidationError(
            "design readiness failed: approved requirements have no implementation-bearing design module: " + ", ".join(missing)
        )
    result = dict(design)
    result["_requirement_design_bindings"] = {
        "schema_version": "mmm/requirement-design-binding-v1",
        "requirement_ids": list(required_ids),
        "bindings": [binding_rows[requirement_id] for requirement_id in required_ids],
    }
    return result


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
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
    ]


def _section_messages(
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    research: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are a bounded Minecraft mod design worker. Output only the requested Markdown "
        "sections. For every requested field write a heading exactly as '## <field>'. "
        "Use plain text or bullets under it. For map fields use '- key: value' or nested bullets. "
        + _MODULE_FORMAT
        + " "
        + _ASSET_FORMAT
        + " Preserve exact approved requirement IDs. Write design content as Markdown, not JSON. "
        + "No code fences, <think>, analysis, "
        "or fields outside the requested headings. "
        + _PRODUCTION_DEPTH
    )
    ledger = _active_requirement_ledger(prompt)
    user = (
        "AUTHORITATIVE REQUEST\n"
        + prompt
        + "\n\nSECTION\n"
        + section_id
        + "\n\nREQUESTED FIELDS\n"
        + "\n".join(f"- {field}" for field in fields)
        + "\n\n"
        + _render_requirement_ledger(ledger)
        + "\n\nRESEARCH CONTEXT\n"
        + _render_design_research(research)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
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
        raise SpecValidationError(f"Planner did not return a research JSON object: {exc}") from exc
    note = payload.get("research_note")
    if not isinstance(note, dict):
        note = payload if isinstance(payload, dict) else {}
    cleaned_claims = []
    for claim in note.get("claims", []):
        if isinstance(claim, dict):
            claim_text = str(
                claim.get("claim") or claim.get("text") or claim.get("claim_text") or claim.get("content") or ""
            ).strip()
            raw_refs = claim.get("evidence_refs", [])
            claim_refs = [str(ref).strip() for ref in raw_refs if str(ref).strip()] if isinstance(raw_refs, list) else []
            if not claim_refs:
                claim_refs = [
                    str(claim.get(key) or "").strip()
                    for key in ("evidence_ref", "source_ref")
                    if str(claim.get(key) or "").strip()
                ]
            if claim_text:
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
        "next_queries": [str(query).strip() for query in note.get("next_queries", []) if str(query).strip()],
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
