from __future__ import annotations

"""Host-owned semantic requirement authority.

The small language model only separates authored behaviors and classifies each one into
an existing host capability. It cannot mint architecture identifiers, prerequisite
edges, implementation obligations, artifact kinds, source offsets, hashes, or task IDs.
Unknown authored behavior is represented by one ``custom.semantic`` sentinel and the
host derives the actual stable custom identifier from grounded source provenance.
"""

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

from . import evidence_first_planning as _evidence
from . import evidence_request_guard as _guard
from .minecraft_template_catalog import (
    CUSTOM_CAPABILITY_SENTINEL,
    capability_catalog_for_model,
    is_known_capability,
    profile_for_capability,
    semantic_capability_choices,
)
from .root_cause_trace import emit_root_cause

_INSTALLED = False
_SCHEMA = "mmm/approved-requirement-graph-v2"
_ALLOWED_AUTHORING_ROLES = frozenset({"explicit", "logically_derived"})
_ALL_PROVENANCE_ROLES = frozenset(
    {
        "explicit",
        "logically_derived",
        "selected_design_alternative",
        "implementation_obligation",
    }
)
_CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_CUSTOM_HOST_ID = re.compile(r"^custom\.semantic_[0-9a-f]{16}$")
_WORD = re.compile(r"\w+", re.UNICODE)
_SOFTWARE_PERFORMANCE_MARKERS = (
    "software.performance",
    "performance.optimization",
    "runtime.performance",
    "code.optimization",
    "performance optimization",
    "tick budget",
    "latency budget",
    "throughput budget",
    "memory budget",
    "profiling",
    "benchmark",
    "mixin",
    "코드 성능",
    "성능 최적화",
    "프로파일링",
    "벤치마크",
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _semantic_type(capability: str, statement: str, supplied: Any = "") -> str:
    supplied_value = str(supplied or "").strip().casefold()
    evidence = f"{capability} {statement}".casefold()
    if any(marker in evidence for marker in _SOFTWARE_PERFORMANCE_MARKERS):
        return "software_quality"
    if supplied_value == "software_quality":
        # Never let a model relabel ordinary gameplay as software-quality work without
        # authored optimization/performance evidence.
        return "gameplay_mechanic"
    return "gameplay_mechanic"


def _implementation_profile(
    capability: str,
    *,
    semantic_type: str = "gameplay_mechanic",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    profile = profile_for_capability(capability, semantic_type=semantic_type)
    return profile.implementation_capabilities, profile.artifact_kinds


def _design_resolution_obligations(
    capability: str,
    *,
    semantic_type: str = "gameplay_mechanic",
) -> tuple[str, ...]:
    return profile_for_capability(
        capability,
        semantic_type=semantic_type,
    ).design_resolution_obligations


def _diagnostic(
    error_code: str,
    json_path: str,
    offending_value: Any,
    expected_contract: str,
    repair_scope: str,
) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "json_path": json_path,
        "offending_value": offending_value,
        "expected_contract": expected_contract,
        "repair_scope": repair_scope,
    }


def _clause_records(prompt: str) -> list[dict[str, Any]]:
    spans = list(_evidence._semantic_clause_spans(prompt))
    if not spans and prompt.strip():
        start = len(prompt) - len(prompt.lstrip())
        end = len(prompt.rstrip())
        spans = [(start, end)]
    records: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(spans):
        text = prompt[start:end]
        records.append(
            {
                "clause_index": index,
                "char_start": start,
                "char_end": end,
                "text": text,
                "text_sha256": _sha256(text),
            }
        )
    if not records:
        raise _evidence.EvidencePlanError(
            "REQ_SOURCE_EMPTY: authored request has no semantic clause."
        )
    return records


def _semantic_schema(max_clause_index: int) -> dict[str, Any]:
    """Expose only host-recognized capability choices and observable semantics."""

    return {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "source_clause_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max(0, max_clause_index),
                        },
                        "capability_id": {
                            "type": "string",
                            "enum": list(semantic_capability_choices()),
                        },
                        "source_anchor": {"type": "string", "minLength": 1},
                        "semantic_statement": {"type": "string", "minLength": 1},
                        "given": {"type": "string", "minLength": 1},
                        "when": {"type": "string", "minLength": 1},
                        "then": {"type": "string", "minLength": 1},
                        "semantic_type": {
                            "type": "string",
                            "enum": ["gameplay_mechanic", "software_quality"],
                        },
                    },
                    "required": [
                        "source_clause_index",
                        "capability_id",
                        "source_anchor",
                        "semantic_statement",
                        "given",
                        "when",
                        "then",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["requirements"],
        "additionalProperties": False,
    }


def _parse_json(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"semantic response type {type(raw).__name__} is not JSON text/object")


def _model_messages(
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "Interpret only independently observable behaviors explicitly authored in the "
        "supplied Minecraft-mod clauses. The host owns architecture and planning. Split "
        "a clause when it contains independently implementable player/world behaviors, "
        "but never invent a genre mechanic, implementation API, persistence scheme, UI, "
        "network feature, progression gate, or dependency that the author did not request. "
        "For capability_id choose EXACTLY one ID from host_capability_catalog. Do not make "
        "up a dotted identifier. Use custom.semantic only when no catalog capability "
        "accurately describes that authored behavior. source_anchor is a short locator in "
        "the supplied clause; the host resolves exact offsets. Return one concrete "
        "Given/When/Then behavior per leaf using only authored conditions and outcomes. "
        "semantic_type is software_quality only for explicit code/runtime optimization, "
        "profiling, latency, throughput, memory, tick or frame-rate requirements; ordinary "
        "gameplay stats such as vehicle speed or durability are gameplay_mechanic. Never "
        "emit prerequisites, requirement IDs, task IDs, implementation obligations, file "
        "paths, versions, loader details, source offsets or hashes."
    )
    payload = {
        "host_capability_catalog": list(capability_catalog_for_model()),
        "custom_capability_sentinel": CUSTOM_CAPABILITY_SENTINEL,
        "host_owned_clauses": [
            {
                "source_clause_index": int(clause["clause_index"]),
                "text": str(clause["text"]),
            }
            for clause in clauses
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _canonical(payload)},
    ]


def _call_semantic_model(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> Any:
    max_clause_index = max(int(clause["clause_index"]) for clause in clauses)
    parameters = _semantic_schema(max_clause_index)
    messages = _model_messages(clauses)
    emit_root_cause(
        "planner_model_request",
        stage="planning",
        operation="compile_semantic_requirements",
        gate="semantic_typing",
        result="START",
        details={"clauses": clauses, "messages": messages, "schema": parameters},
    )
    native = getattr(router, "generate_tool_decision", None)
    if callable(native):
        result = native(
            "planner",
            messages,
            tool_name="compile_semantic_requirements",
            parameters=parameters,
            description=(
                "Classify every independently observable authored behavior into one "
                "host-provided Minecraft capability; host owns all planning structure."
            ),
        )
        emit_root_cause(
            "planner_model_response",
            stage="planning",
            operation="compile_semantic_requirements",
            gate="semantic_typing",
            result="PASS",
            details={"raw_response": result},
        )
        return result
    raw = router.generate_text(
        "planner",
        messages,
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    result = _parse_json(raw)
    emit_root_cause(
        "planner_model_response",
        stage="planning",
        operation="compile_semantic_requirements",
        gate="semantic_typing",
        result="PASS",
        details={"raw_response": raw, "parsed_response": result},
    )
    return result


def _similarity_projection(value: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    raw_positions: list[int] = []
    for raw_index, character in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if normalized.isspace() or unicodedata.category(normalized).startswith("P"):
                continue
            characters.append(normalized)
            raw_positions.append(raw_index)
    return "".join(characters), raw_positions


def _semantic_terms(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            match.group(0).casefold()
            for value in values
            for match in _WORD.finditer(str(value or ""))
        )
    )


def _projected_span(
    clause: Mapping[str, Any],
    positions: Sequence[int],
    start: int,
    end: int,
) -> tuple[int, int, str]:
    text = str(clause["text"])
    raw_start = positions[start]
    raw_end = positions[end - 1] + 1
    absolute_start = int(clause["char_start"]) + raw_start
    return absolute_start, absolute_start + (raw_end - raw_start), text[raw_start:raw_end]


def _ground_source_anchor(
    clause: Mapping[str, Any],
    source_anchor: str,
) -> dict[str, Any] | None:
    """Resolve a model locator to exact host-owned prompt provenance."""

    text = str(clause["text"])
    anchor = str(source_anchor or "").strip()
    if not anchor:
        return None

    raw_start = text.find(anchor)
    if raw_start >= 0 and text.find(anchor, raw_start + len(anchor)) < 0:
        absolute_start = int(clause["char_start"]) + raw_start
        return {
            "source_quote": text[raw_start : raw_start + len(anchor)],
            "source_start": absolute_start,
            "source_end": absolute_start + len(anchor),
            "grounding_method": "exact",
            "grounding_similarity": 1.0,
            "model_anchor": anchor,
        }

    anchor_form, _ = _similarity_projection(anchor)
    text_form, text_positions = _similarity_projection(text)
    if not anchor_form or not text_form:
        return None

    normalized_start = text_form.find(anchor_form)
    if normalized_start >= 0:
        start, end, quote = _projected_span(
            clause,
            text_positions,
            normalized_start,
            normalized_start + len(anchor_form),
        )
        return {
            "source_quote": quote,
            "source_start": start,
            "source_end": end,
            "grounding_method": "normalized_exact_host_alignment",
            "grounding_similarity": 1.0,
            "model_anchor": anchor,
        }

    anchor_terms = set(_semantic_terms((anchor,)))
    authored_terms = set(_semantic_terms((text,)))
    if anchor_terms.isdisjoint(authored_terms):
        return None

    matcher = SequenceMatcher(None, anchor_form, text_form, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return None
    projected_start = min(block.b for block in blocks)
    projected_end = max(block.b + block.size for block in blocks)
    if projected_start >= projected_end:
        return None

    start, end, quote = _projected_span(
        clause,
        text_positions,
        projected_start,
        projected_end,
    )
    return {
        "source_quote": quote,
        "source_start": start,
        "source_end": end,
        "grounding_method": "fuzzy_host_alignment",
        "grounding_similarity": round(matcher.ratio(), 6),
        "model_anchor": anchor,
    }


def _host_capability_id(
    model_capability: str,
    *,
    grounding: Mapping[str, Any],
    clause_index: int,
    item_index: int,
) -> str:
    if model_capability != CUSTOM_CAPABILITY_SENTINEL:
        if not is_known_capability(model_capability):
            raise ValueError(f"capability is outside the host catalog: {model_capability!r}")
        return model_capability
    digest = _sha256(
        {
            "source_start": grounding["source_start"],
            "source_end": grounding["source_end"],
            "source_quote": grounding["source_quote"],
            "source_clause_index": clause_index,
            "model_leaf_ordinal": item_index,
        }
    )[7:23]
    return f"custom.semantic_{digest}"


def _normalize_requirement(
    raw: Any,
    *,
    item_index: int,
    clauses_by_index: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    path = f"$.requirements[{item_index}]"
    if not isinstance(raw, Mapping):
        return None, _diagnostic(
            "REQ_SCHEMA_ITEM", path, raw, "semantic requirement object", path
        ), None

    allowed_fields = {
        "source_clause_index",
        "capability_id",
        "source_anchor",
        "semantic_statement",
        "given",
        "when",
        "then",
        "semantic_type",
    }
    unexpected = sorted(set(raw) - allowed_fields)
    if unexpected:
        return None, _diagnostic(
            "REQ_MODEL_AUTHORITY_OVERREACH",
            path,
            unexpected,
            "only bounded semantic classification fields; host owns prerequisites and architecture",
            path,
        ), None

    clause_index = raw.get("source_clause_index")
    if type(clause_index) is not int or clause_index not in clauses_by_index:
        return None, _diagnostic(
            "REQ_SOURCE_CLAUSE",
            path + ".source_clause_index",
            clause_index,
            f"one supplied host clause index: {sorted(clauses_by_index)}",
            path,
        ), None

    model_capability = str(raw.get("capability_id") or "").strip().casefold()
    if model_capability not in set(semantic_capability_choices()):
        return None, _diagnostic(
            "REQ_CAPABILITY_CATALOG",
            path + ".capability_id",
            raw.get("capability_id"),
            "one exact host capability enum value or custom.semantic",
            f"clause:{clause_index}",
        ), clause_index

    semantic_statement = str(raw.get("semantic_statement") or "").strip()
    given = str(raw.get("given") or "").strip()
    when = str(raw.get("when") or "").strip()
    then = str(raw.get("then") or "").strip()
    if not semantic_statement or not (given and when and then):
        return None, _diagnostic(
            "REQ_SEMANTIC_CONTRACT",
            path,
            {
                "semantic_statement": raw.get("semantic_statement"),
                "given": raw.get("given"),
                "when": raw.get("when"),
                "then": raw.get("then"),
            },
            "non-empty semantic_statement and concrete given/when/then strings",
            f"clause:{clause_index}",
        ), clause_index

    source_anchor = str(raw.get("source_anchor") or "").strip()
    grounding = _ground_source_anchor(clauses_by_index[clause_index], source_anchor)
    if grounding is None:
        return None, _diagnostic(
            "REQ_SOURCE_GROUNDING",
            path + ".source_anchor",
            source_anchor,
            "a semantic locator supported by the authored clause; host owns exact source offsets",
            f"clause:{clause_index}",
        ), clause_index

    try:
        capability = _host_capability_id(
            model_capability,
            grounding=grounding,
            clause_index=clause_index,
            item_index=item_index,
        )
    except ValueError as exc:
        return None, _diagnostic(
            "REQ_CAPABILITY_CATALOG",
            path + ".capability_id",
            model_capability,
            str(exc),
            f"clause:{clause_index}",
        ), clause_index

    semantic_type = _semantic_type(
        capability,
        semantic_statement,
        raw.get("semantic_type"),
    )
    node = {
        "capability_id": capability,
        "model_capability_choice": model_capability,
        "semantic_type": semantic_type,
        "provenance_role": "explicit",
        "source_clause_index": clause_index,
        **grounding,
        "semantic_statement": semantic_statement,
        "derived_from": [],
        "depends_on": [],
        "derivation_reason": "",
        "observable_behavior": {
            "given": given,
            "when": when,
            "then": then,
        },
        "required_prerequisite_capabilities": [],
        "optional_prerequisite_capabilities": [],
    }
    return node, None, clause_index


def _evaluate_batch(
    payload: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, Any]]]:
    clauses_by_index = {int(clause["clause_index"]): clause for clause in clauses}
    all_indices = set(clauses_by_index)
    if not isinstance(payload, Mapping):
        return [], set(all_indices), [
            _diagnostic(
                "REQ_SCHEMA_ROOT",
                "$",
                type(payload).__name__,
                "JSON object with a requirements array",
                "semantic_batch",
            )
        ]

    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        return [], set(all_indices), [
            _diagnostic(
                "REQ_SCHEMA_REQUIREMENTS",
                "$.requirements",
                raw_requirements,
                "non-empty requirements array",
                "semantic_batch",
            )
        ]

    nodes: list[dict[str, Any]] = []
    invalid_clauses: set[int] = set()
    diagnostics: list[dict[str, Any]] = []
    global_failure = False
    for item_index, raw in enumerate(raw_requirements):
        node, diagnostic, clause_index = _normalize_requirement(
            raw,
            item_index=item_index,
            clauses_by_index=clauses_by_index,
        )
        if diagnostic is not None:
            diagnostics.append(diagnostic)
            if clause_index is None:
                global_failure = True
            else:
                invalid_clauses.add(clause_index)
            continue
        assert node is not None
        nodes.append(node)
        emit_root_cause(
            "semantic_requirement_normalized",
            stage="planning",
            operation="normalize_requirement",
            gate="semantic_schema_and_grounding",
            result="PASS",
            details={"item_index": item_index, "model_item": raw, "normalized_node": node},
        )

    if global_failure:
        invalid_clauses = set(all_indices)

    covered = {int(node["source_clause_index"]) for node in nodes}
    for clause_index in sorted(all_indices - covered):
        invalid_clauses.add(clause_index)
        diagnostics.append(
            _diagnostic(
                "REQ_SOURCE_COVERAGE",
                "$.requirements",
                clause_index,
                "at least one explicit semantic requirement for every supplied clause",
                f"clause:{clause_index}",
            )
        )

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for node in nodes:
        key = (
            int(node["source_clause_index"]),
            str(node["capability_id"]),
            str(node["semantic_statement"]).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(node)
    return deduplicated, invalid_clauses, diagnostics


def _assign_local_ids(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[int, int] = {}
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for raw in sorted(
        nodes,
        key=lambda item: (
            int(item["source_clause_index"]),
            int(item["source_start"]),
            str(item["capability_id"]),
        ),
    ):
        key = (
            int(raw["source_clause_index"]),
            str(raw["capability_id"]),
            str(raw["semantic_statement"]).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        clause_index = int(raw["source_clause_index"])
        ordinal = counters.get(clause_index, 0)
        counters[clause_index] = ordinal + 1
        item = dict(raw)
        item["local_id"] = f"c{clause_index}_{ordinal}"
        result.append(item)
    return result


def _generate_approved_nodes(
    prompt: str,
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        payload = _call_semantic_model(router, clauses)
    except Exception as exc:
        raise _evidence.EvidencePlanError(
            "semantic requirement authority model call failed: "
            + _canonical(
                _diagnostic(
                    "REQ_MODEL_RESPONSE",
                    "$",
                    f"{type(exc).__name__}: {exc}",
                    "one bounded semantic-classification payload",
                    "semantic_batch",
                )
            )
        ) from exc

    nodes, invalid_clauses, diagnostics = _evaluate_batch(payload, clauses)
    if invalid_clauses:
        raise _evidence.EvidencePlanError(
            "semantic requirement authority rejected invalid model output: "
            + _canonical(
                {
                    "invalid_clause_indices": sorted(invalid_clauses),
                    "diagnostics": diagnostics,
                }
            )
        )
    return _assign_local_ids(nodes)


def _build_catalog(
    prompt: str,
    nodes: Sequence[Mapping[str, Any]],
    clauses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prompt_hash = _sha256(prompt)
    requirement_ids: dict[str, str] = {}
    for item in nodes:
        requirement_ids[str(item["local_id"])] = _evidence._stable_id(
            "req",
            str(item["capability_id"]),
            {
                "prompt_sha256": prompt_hash,
                "source_clause_index": item["source_clause_index"],
                "source_span": [item["source_start"], item["source_end"]],
                "provenance_role": item["provenance_role"],
            },
        )

    requirements: list[dict[str, Any]] = []
    for item in nodes:
        requirement_id = requirement_ids[str(item["local_id"])]
        clause = clauses[int(item["source_clause_index"])]
        quote = str(item["source_quote"])
        source_start = int(item["source_start"])
        source_end = int(item["source_end"])
        behavior = dict(item["observable_behavior"])
        semantic_type = str(item["semantic_type"])
        implementation_capabilities, artifact_kinds = _implementation_profile(
            str(item["capability_id"]),
            semantic_type=semantic_type,
        )
        artifact_task_ids = [
            _evidence._stable_id(
                "task",
                implementation_capability,
                {"requirement_id": requirement_id, "layer": "implementation"},
            )
            for implementation_capability in implementation_capabilities
        ]
        design_obligations = _design_resolution_obligations(
            str(item["capability_id"]),
            semantic_type=semantic_type,
        )
        requirement = {
            "requirement_id": requirement_id,
            "capability": item["capability_id"],
            "statement": str(clause["text"]),
            "semantic_statement": item["semantic_statement"],
            "mandatory": True,
            "provenance_role": "explicit",
            "source_span": {
                "source_id": "requested_prompt",
                "char_start": source_start,
                "char_end": source_end,
                "text": quote,
                "text_sha256": _sha256(quote),
                "source_clause_index": item["source_clause_index"],
                "source_clause_sha256": clause["text_sha256"],
                "grounding_method": item["grounding_method"],
                "grounding_similarity": item["grounding_similarity"],
                "model_anchor": item["model_anchor"],
            },
            "derived_from": [],
            "depends_on": [],
            "provides": [_evidence._canonical_capability(item["capability_id"])],
            "gameplay_capabilities": [item["capability_id"]],
            "implementation_capabilities": list(implementation_capabilities),
            "artifact_task_ids": artifact_task_ids,
            "semantic_type": semantic_type,
            "unlock_policy": {
                "required_capabilities": [],
                "required_requirement_refs": [],
                "optional_capabilities": [],
                "optional_requirement_refs": [],
                "policy": "host_feature_model_and_authored_state_only",
            },
            "artifact_obligations": [
                {"kind": kind, "status": "REQUIRED_DESIGN_AND_GENERATION"}
                for kind in artifact_kinds
            ],
            "design_resolution_obligations": list(design_obligations),
            "runtime_acceptance": [
                (
                    f"Given {behavior['given']}; when {behavior['when']} is exercised in "
                    f"a disposable server-authoritative GameTest scenario; then {behavior['then']} "
                    "and every changed persistent, inventory, world, entity, UI and network-visible "
                    "state that this template owns is independently observed."
                )
            ],
            "semantic_status": "RESOLVED",
            "unresolved_spans": [],
            "acceptance": [
                f"Given {behavior['given']}; when {behavior['when']}; then {behavior['then']}."
            ],
            "observable_behavior": behavior,
            "template_profile": {
                "template_id": profile_for_capability(
                    str(item["capability_id"]), semantic_type=semantic_type
                ).template_id,
                "model_capability_choice": item.get("model_capability_choice"),
                "architecture_owner": "host",
            },
        }
        requirements.append(requirement)
        emit_root_cause(
            "requirement_contract_compiled",
            stage="planning",
            operation="build_requirement_catalog",
            gate="semantic_to_implementation",
            result="PASS",
            details={
                "requirement": requirement,
                "required_dependency_refs": [],
                "optional_dependency_refs": [],
                "dependency_owner": "host_feature_model",
            },
        )

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "prompt_sha256": prompt_hash,
        "prompt_char_length": len(prompt),
        "purpose": prompt.strip(),
        "requirements": requirements,
        "constraints": [],
        "non_goals": [],
        "deployment_expectations": [],
        "requirement_graph": {
            "node_ids": [item["requirement_id"] for item in requirements],
            "edges": [],
        },
        "semantic_audit": {
            "status": "APPROVED",
            "authored_clause_count": len(clauses),
            "covered_clause_count": len(clauses),
            "unresolved_clause_count": 0,
            "unsupported_design_choice_count": 0,
            "provenance_roles": sorted(_ALL_PROVENANCE_ROLES),
            "normal_model_turns": 1,
            "max_repair_turns": 0,
            "generation_policy": "host_catalog_classification_only",
            "capability_id_owner": "host_catalog",
            "dependency_owner": "host",
            "implementation_architecture_owner": "host",
            "source_grounding_owner": "host",
        },
        "catalog_sha256": "",
    }
    payload["catalog_sha256"] = _evidence._hash_without(payload, "catalog_sha256")
    return payload


def _valid_host_capability(capability: str) -> bool:
    return is_known_capability(capability) or bool(_CUSTOM_HOST_ID.fullmatch(capability))


def validate_approved_requirement_catalog(
    catalog: Mapping[str, Any],
    *,
    prompt: str,
) -> None:
    if catalog.get("schema_version") != _SCHEMA:
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_SCHEMA: approved requirement catalog schema is missing."
        )
    if catalog.get("catalog_sha256") != _evidence._hash_without(
        catalog, "catalog_sha256"
    ):
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_HASH: approved requirement catalog hash mismatch."
        )
    if catalog.get("prompt_sha256") != _sha256(prompt):
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_PROMPT: requirement authority is stale for this prompt."
        )

    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_EMPTY: approved requirement graph has no nodes."
        )

    ids: set[str] = set()
    covered_clauses: set[int] = set()
    for index, raw in enumerate(requirements):
        if not isinstance(raw, Mapping):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_NODE: requirement[{index}] is not an object."
            )
        requirement_id = str(raw.get("requirement_id") or "")
        if not requirement_id or requirement_id in ids:
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_ID: invalid/duplicate requirement id at index {index}."
            )
        ids.add(requirement_id)

        capability = str(raw.get("capability") or "")
        if not _CAPABILITY_ID.fullmatch(capability) or not _valid_host_capability(capability):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_CAPABILITY: capability is not host-owned: {capability!r}."
            )

        role = str(raw.get("provenance_role") or "")
        if role not in _ALLOWED_AUTHORING_ROLES:
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_OVERREACH: {role!r} cannot be a mandatory authored node."
            )

        span = raw.get("source_span")
        if not isinstance(span, Mapping):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_SOURCE: missing source span for {requirement_id}."
            )
        start, end = span.get("char_start"), span.get("char_end")
        text = str(span.get("text") or "")
        if (
            type(start) is not int
            or type(end) is not int
            or not (0 <= start < end <= len(prompt))
            or prompt[start:end] != text
            or span.get("text_sha256") != _sha256(text)
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_SOURCE: stale source receipt for {requirement_id}."
            )
        clause_index = span.get("source_clause_index")
        if type(clause_index) is not int:
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_SOURCE: invalid clause index for {requirement_id}."
            )
        covered_clauses.add(clause_index)

        observable = raw.get("observable_behavior")
        if not isinstance(observable, Mapping) or not all(
            str(observable.get(field) or "").strip()
            for field in ("given", "when", "then")
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_ACCEPTANCE: concrete observable contract missing for {requirement_id}."
            )

        semantic_type = str(raw.get("semantic_type") or "")
        if semantic_type not in {"gameplay_mechanic", "software_quality"}:
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_SEMANTIC_TYPE: invalid semantic type for {requirement_id}."
            )
        expected_profile = profile_for_capability(capability, semantic_type=semantic_type)
        if list(raw.get("implementation_capabilities") or ()) != list(
            expected_profile.implementation_capabilities
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_IMPLEMENTATION: implementation profile is not host-derived for {requirement_id}."
            )
        expected_artifacts = [
            {"kind": kind, "status": "REQUIRED_DESIGN_AND_GENERATION"}
            for kind in expected_profile.artifact_kinds
        ]
        if raw.get("artifact_obligations") != expected_artifacts:
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_ARTIFACTS: artifact obligations are not host-derived for {requirement_id}."
            )
        if list(raw.get("design_resolution_obligations") or ()) != list(
            expected_profile.design_resolution_obligations
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_DESIGN_RESOLUTION: design obligations are not host-derived for {requirement_id}."
            )
        if not isinstance(raw.get("artifact_task_ids"), list) or not raw.get(
            "artifact_task_ids"
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_ARTIFACT_TASKS: artifact tasks missing for {requirement_id}."
            )
        if not isinstance(raw.get("runtime_acceptance"), list) or not raw.get(
            "runtime_acceptance"
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_RUNTIME_ACCEPTANCE: runtime scenario missing for {requirement_id}."
            )
        unlock = raw.get("unlock_policy")
        if not isinstance(unlock, Mapping) or unlock.get("policy") != (
            "host_feature_model_and_authored_state_only"
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_UNLOCK_POLICY: host-owned unlock policy missing for {requirement_id}."
            )
        for field in ("required_capabilities", "required_requirement_refs", "optional_capabilities", "optional_requirement_refs"):
            if not isinstance(unlock.get(field), list):
                raise _evidence.EvidencePlanError(
                    f"REQ_AUTHORITY_UNLOCK_POLICY: {field} must be a host-owned list for {requirement_id}."
                )

        for field in ("derived_from", "depends_on"):
            refs = raw.get(field, [])
            if not isinstance(refs, list):
                raise _evidence.EvidencePlanError(
                    f"REQ_AUTHORITY_GRAPH: {field} must be a list."
                )

    for raw in requirements:
        for field in ("derived_from", "depends_on"):
            if any(str(ref) not in ids for ref in raw.get(field, [])):
                raise _evidence.EvidencePlanError(
                    f"REQ_AUTHORITY_GRAPH: {field} contains unknown requirement IDs."
                )

    dependency_map = {
        str(raw["requirement_id"]): tuple(str(value) for value in raw.get("depends_on", ()))
        for raw in requirements
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(requirement_id: str) -> None:
        if requirement_id in visited:
            return
        if requirement_id in visiting:
            raise _evidence.EvidencePlanError(
                "REQ_AUTHORITY_GRAPH: dependency cycle detected."
            )
        visiting.add(requirement_id)
        for dependency in dependency_map.get(requirement_id, ()):
            visit(dependency)
        visiting.remove(requirement_id)
        visited.add(requirement_id)

    for requirement_id in dependency_map:
        visit(requirement_id)

    audit = catalog.get("semantic_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "APPROVED"
        or audit.get("unresolved_clause_count") != 0
        or audit.get("unsupported_design_choice_count") != 0
        or audit.get("source_grounding_owner") != "host"
        or audit.get("capability_id_owner") != "host_catalog"
        or audit.get("dependency_owner") != "host"
        or audit.get("implementation_architecture_owner") != "host"
    ):
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_BARRIER: semantic authority audit is not host-owned/approved."
        )
    expected_clause_count = audit.get("authored_clause_count")
    if type(expected_clause_count) is not int or len(covered_clauses) != expected_clause_count:
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_COVERAGE: authored clause coverage is incomplete."
        )


def build_approved_requirement_catalog(
    prompt: str,
    router: Any | None = None,
) -> dict[str, Any]:
    if router is None:
        return _guard._ORIGINAL_BUILD_REQUEST_CATALOG(prompt, {}, router=None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise _evidence.EvidencePlanError(
            "REQ_SOURCE_EMPTY: semantic authority requires a non-empty prompt."
        )
    clauses = _clause_records(prompt)
    nodes = _generate_approved_nodes(prompt, router, clauses)
    catalog = _build_catalog(prompt, nodes, clauses)
    validate_approved_requirement_catalog(catalog, prompt=prompt)
    return catalog


def install_semantic_requirement_authority() -> None:
    """Install the approved graph as the sole production semantic authority."""

    global _INSTALLED
    if _INSTALLED:
        return
    _guard.build_authoritative_request_catalog = build_approved_requirement_catalog

    original_validate = _evidence._validate_request_catalog
    if not getattr(original_validate, "__mmm_approved_requirement_authority__", False):

        def validate(catalog: Mapping[str, Any], *, prompt: str) -> None:
            original_validate(catalog, prompt=prompt)
            if catalog.get("schema_version") == _SCHEMA:
                validate_approved_requirement_catalog(catalog, prompt=prompt)

        validate.__mmm_approved_requirement_authority__ = True  # type: ignore[attr-defined]
        validate.__wrapped__ = original_validate  # type: ignore[attr-defined]
        _evidence._validate_request_catalog = validate

    _INSTALLED = True


__all__ = [
    "build_approved_requirement_catalog",
    "install_semantic_requirement_authority",
    "validate_approved_requirement_catalog",
]
