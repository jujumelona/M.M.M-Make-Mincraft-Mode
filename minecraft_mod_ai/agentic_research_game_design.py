from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .planner_stage_trace import PlannerStageTrace
from .spec import SpecValidationError


# One host-owned contract defines model instructions, Markdown parsing, host structure,
# and validation. Runtime installers must not mutate this schema or replace its parser.


from .design_markdown import (
    _LIST_FIELDS as _LIST_FIELDS,
    _MAP_FIELDS as _MAP_FIELDS,
    _NONE_VALUES as _NONE_VALUES,
    _section_field_body as _section_field_body,
    _strip_accidental_field_wrapper as _strip_accidental_field_wrapper,
    _parse_field_output as _parse_field_output,
    _normalize_heading as _normalize_heading,
    _parse_markdown_section as _parse_markdown_section,
    _plain_text as _plain_text,
    _strip_list_marker as _strip_list_marker,
    _markdown_list as _markdown_list,
    _markdown_map as _markdown_map,
    _split_csv as _split_csv,
    _split_obligations as _split_obligations,
    _record_key_value as _record_key_value,
    _pipe_parts as _pipe_parts,
    _is_markdown_table_separator as _is_markdown_table_separator,
    _finalize_module_record as _finalize_module_record,
    _module_rows as _module_rows,
    _finalize_asset_record as _finalize_asset_record,
    _asset_rows as _asset_rows,
)
from .design_requirement_contract import (
    _REQUIREMENT_ID_RE as _REQUIREMENT_ID_RE,
    _referenced_requirement_ids as _referenced_requirement_ids,
    _assert_known_requirement_ids as _assert_known_requirement_ids,
    _active_requirement_ledger as _active_requirement_ledger,
    _render_requirement_ledger as _render_requirement_ledger,
    _nonempty_text_list as _nonempty_text_list,
    _validate_section_types as _validate_section_types,
    _validate_requirement_coverage as _validate_requirement_coverage,
)
from .design_research_context import (
    _RESEARCH_NOTE_SCHEMA as _RESEARCH_NOTE_SCHEMA,
    _json_sha256 as _json_sha256,
    _domain_source_value as _domain_source_value,
    _has_grounding_content as _has_grounding_content,
    _domain_evidence_slice as _domain_evidence_slice,
    _allowed_research_refs as _allowed_research_refs,
    _claim_refs as _claim_refs,
    _validate_sufficient_research as _validate_sufficient_research,
    _research_domain_with_agent as _research_domain_with_agent,
    _research_messages as _research_messages,
    _render_design_research as _render_design_research,
    _compact_research_for_design as _compact_research_for_design,
    _research_receipt as _research_receipt,
    _candidate_research_note as _candidate_research_note,
    _parse_research_note as _parse_research_note,
    _extract_json_object as _extract_json_object,
    _error as _error,
)
from .design_section_schema import (
    _SECTION_SPECS as _SECTION_SPECS,
)


def supports_agentic_research_router(router: Any) -> bool:
    from .model_router import ModelRouter

    return isinstance(router, ModelRouter)


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
    """Compile independently validated fields; never replace failed content with defaults."""
    from .planner_field_worker import generate_field
    from .research_requirement_evidence import evidence_catalog, requirement_score

    del host_properties
    trace = PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata=dict(trace_metadata or {}),
    )
    ledger = _active_requirement_ledger(prompt)
    evidence = evidence_catalog(research, {}, {})
    section: dict[str, Any] = {}
    failures: list[str] = []
    asset_decisions: dict[str, str] = {}
    for field in fields:
        # Identity/overall progression remain coherent; implementation content has one owner.
        scoped = field in {
            "modules",
            "assets",
            "combat",
            "mod_context",
            "acceptance_tests",
        }
        scopes = [(item,) for item in ledger] if scoped and ledger else [ledger]
        values: list[Any] = []
        for scope in scopes:
            requirement = scope[0] if scoped and len(scope) == 1 else None
            ref = str(requirement["requirement_id"]) if requirement else ""
            ranked = sorted(
                (
                    (max((requirement_score(item, r) for r in scope), default=0), item)
                    for item in evidence
                ),
                key=lambda pair: (-pair[0], pair[1]["evidence_ref"]),
            )
            context = {
                "domain_notes": [item for score, item in ranked if score > 0][:12]
            }
            messages = _field_messages(
                prompt=prompt,
                section_id=section_id,
                field=field,
                research=context,
                ledger=scope,
                host_module=bool(ref and field == "modules"),
            )

            def parse(raw: str) -> Any:
                body = raw.strip()
                if re.search(r"^\s*(?:#\s+)?##\s+", body, re.MULTILINE):
                    body = _section_field_body(body, field, fields)
                if not body.strip():
                    raise SpecValidationError(f"{field} content is missing")
                if field == "modules" and requirement:
                    obligations = _parse_field_output(body, "core_loop")
                    if not obligations or any(
                        item.strip().casefold() in _NONE_VALUES for item in obligations
                    ):
                        raise SpecValidationError(
                            "modules requires concrete implementation obligations, not none"
                        )
                    if _referenced_requirement_ids(obligations):
                        raise SpecValidationError(
                            "Return behavior only; the host owns requirement IDs"
                        )
                    value = [
                        {
                            "plugin_id": "design_" + ref,
                            "status": "custom_required",
                            "capability": requirement.get("capability", ""),
                            "reason": requirement.get("semantic_statement")
                            or requirement.get("authored_text"),
                            "requirement_refs": [ref],
                            "implementation_obligations": obligations,
                        }
                    ]
                elif (
                    field == "assets"
                    and requirement
                    and body.casefold().startswith("none:")
                ):
                    reason = body.split(":", 1)[1].strip()
                    if not reason:
                        raise SpecValidationError("Explain why existing assets suffice")
                    asset_decisions[ref] = reason
                    value = []
                else:
                    value = _parse_field_output(body, field)
                    if field == "assets" and requirement:
                        if not value:
                            raise SpecValidationError(
                                "Empty assets require 'none: <reason existing assets suffice>'"
                            )
                        for row in value:
                            row["id"] = ref + "_" + row["id"]
                        asset_decisions[ref] = "dedicated_assets_specified"
                _validate_section_types(
                    {field: value},
                    (field,),
                    requirement_ids=[item["requirement_id"] for item in scope],
                )
                return value

            try:
                values.append(
                    generate_field(
                        router,
                        messages=messages,
                        parse=parse,
                        trace=trace,
                        field=field,
                        requirement_ref=ref,
                        media_paths=media_paths if not section and not values else (),
                    )
                )
            except SpecValidationError as exc:
                failures.append(str(exc))
        if not values:
            continue
        if isinstance(values[0], list):
            section[field] = [item for value in values for item in value]
        elif isinstance(values[0], dict):
            merged: dict[str, list[str]] = {}
            for value in values:
                for key, entries in value.items():
                    merged.setdefault(key, []).extend(entries)
            section[field] = merged
        else:
            section[field] = values[0]
    if failures:
        raise SpecValidationError(
            "Design fields remain unresolved: " + "; ".join(failures)
        )
    if asset_decisions:
        section["_asset_design_decisions"] = asset_decisions
    trace.record_success(section)
    return section


def _field_messages(
    *,
    prompt: str,
    section_id: str,
    field: str,
    research: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]] | None = None,
    host_module: bool = False,
) -> list[dict[str, str]]:
    if host_module:
        format_instruction = "Return concise bullets of concrete implementation obligations for exactly the supplied behavior. No IDs, keys, module metadata or other requirements; the host owns the record."
    elif field in {"title", "pitch"}:
        format_instruction = "Return only the field text."
    elif field in _LIST_FIELDS:
        format_instruction = "Return only one or more concise bullet lines."
    elif field in _MAP_FIELDS:
        format_instruction = (
            "Return 'none' or use ### subgroup headings followed by bullets."
        )
    elif field == "modules":
        format_instruction = (
            "Return module records only. For each module use ### <plugin_id>, then "
            "- status: <value>, - reason: <text>, - requirement_refs: <exact comma-separated approved IDs>, "
            "and - implementation_obligations: followed by one or more nested bullets."
        )
    elif field == "assets":
        format_instruction = (
            "Return asset records only. For each asset use ### <id>, then - kind: <kind> and - brief: <description>. "
            "Return 'none: <reason existing assets suffice>' when no dedicated asset is required."
        )
    else:
        format_instruction = "Return only the requested field content."
    system = (
        "You are a bounded Minecraft mod design worker. The host already owns the field name and final structure. "
        "Generate semantic content for exactly one field. Do not write the field name, a Markdown ## heading, JSON, "
        "code fences, <think>, analysis, or unrelated fields. Never invent requirement IDs; cite only exact host-approved IDs. "
        + format_instruction
        + " No JSON. "
        + " Preserve authored behavior and state transitions; do not invent game mechanics or target APIs."
    )
    ledger = _active_requirement_ledger(prompt) if ledger is None else ledger
    user = (
        "AUTHORITATIVE REQUEST\n"
        + prompt
        + "\n\nSECTION\n"
        + section_id
        + "\n\nFIELD\n"
        + field
        + "\n\n"
        + _render_requirement_ledger(ledger)
        + "\n\nRESEARCH CONTEXT\n"
        + _render_design_research(research)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


__all__ = ["generate_sectioned_game_design"]
