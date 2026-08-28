from __future__ import annotations

"""Host-owned semantic requirement authority and pre-retrieval approval barrier.

The authored prompt is immutable source material. A semantic model may map host-owned
clauses to gameplay requirements, but it cannot invent source text, turn a design
alternative into a mandatory user requirement, or authorize retrieval while coverage
is unresolved. Stable requirement identities are created exactly once in this module
and are then carried downstream unchanged.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import evidence_first_planning as _evidence
from . import evidence_request_guard as _guard

_INSTALLED = False
_SCHEMA = "mmm/approved-requirement-graph-v1"
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
_LOCAL_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_OPAQUE_CAPABILITY = re.compile(r"^(?:semantic_[0-9a-f]{6,}|unresolved:)", re.IGNORECASE)


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


def _semantic_schema(clause_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "minItems": clause_count,
                "maxItems": max(clause_count * 8, clause_count),
                "items": {
                    "type": "object",
                    "properties": {
                        "local_id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "capability_id": {"type": "string", "minLength": 3, "maxLength": 128},
                        "provenance_role": {
                            "type": "string",
                            "enum": sorted(_ALL_PROVENANCE_ROLES),
                        },
                        "source_clause_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max(0, clause_count - 1),
                        },
                        "source_quote": {"type": "string", "minLength": 1},
                        "semantic_statement": {"type": "string", "minLength": 1},
                        "derived_from": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1, "maxLength": 64},
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1, "maxLength": 64},
                        },
                        "derivation_reason": {"type": "string"},
                        "observable_behavior": {
                            "type": "object",
                            "properties": {
                                "given": {"type": "string", "minLength": 1},
                                "when": {"type": "string", "minLength": 1},
                                "then": {"type": "string", "minLength": 1},
                            },
                            "required": ["given", "when", "then"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "local_id",
                        "capability_id",
                        "provenance_role",
                        "source_clause_index",
                        "source_quote",
                        "semantic_statement",
                        "derived_from",
                        "depends_on",
                        "derivation_reason",
                        "observable_behavior",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["requirements"],
        "additionalProperties": False,
    }


def _clause_records(prompt: str) -> list[dict[str, Any]]:
    spans = list(_evidence._semantic_clause_spans(prompt))
    if not spans and prompt.strip():
        start = len(prompt) - len(prompt.lstrip())
        end = len(prompt.rstrip())
        spans = [(start, end)]
    result: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(spans):
        text = prompt[start:end]
        result.append(
            {
                "clause_index": index,
                "char_start": start,
                "char_end": end,
                "text": text,
                "text_sha256": _sha256(text),
            }
        )
    if not result:
        raise _evidence.EvidencePlanError(
            "REQ_SOURCE_EMPTY: authored request has no semantic clause."
        )
    return result


def _messages(
    prompt: str,
    clauses: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any] | None,
    previous_candidate: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    system = (
        "You are the semantic requirement authority for a Minecraft mod planner. "
        "Map only authored meaning and logically necessary gameplay requirements. "
        "Do NOT choose implementation techniques, UI patterns, blueprint systems, boss "
        "variants, persistence mechanisms, networking schemes, or other design alternatives "
        "unless the authored text explicitly requires that behavior. The host owns source "
        "clauses and IDs. Every authored clause must have at least one provenance_role="
        "explicit requirement. Additional logically_derived requirements are allowed only "
        "when they are necessary for an explicit goal and must cite derived_from local IDs "
        "plus a non-empty derivation_reason. selected_design_alternative and "
        "implementation_obligation are forbidden in this phase; those belong to later "
        "design/artifact resolution. capability_id must be a meaningful lower-case dotted "
        "semantic ID, never a hash or opaque placeholder. source_quote must be the smallest "
        "verbatim substring of the assigned clause that grounds this requirement. If one "
        "clause contains several independent meanings, give each its own local_id and precise "
        "source_quote. depends_on expresses gameplay/progression causality between requirement "
        "local IDs, not Java/datagen implementation ordering. observable_behavior must state "
        "a concrete Given/When/Then player-visible contract. Return JSON only."
    )
    payload = {
        "authoritative_prompt_receipt": {
            "sha256": _sha256(prompt),
            "char_count": len(prompt),
        },
        "host_owned_clauses": [dict(item) for item in clauses],
        "repair_diagnostic": dict(diagnostic) if diagnostic else None,
        "previous_candidate": dict(previous_candidate) if previous_candidate else None,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _canonical(payload)},
    ]


def _find_quote(clause: Mapping[str, Any], quote: str) -> tuple[int, int] | None:
    text = str(clause["text"])
    first = text.find(quote)
    if first < 0:
        return None
    if text.find(quote, first + 1) >= 0:
        return None
    start = int(clause["char_start"]) + first
    return start, start + len(quote)


def _validate_candidate(
    payload: Any,
    *,
    prompt: str,
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    del prompt
    if not isinstance(payload, Mapping):
        return None, _diagnostic(
            "REQ_SCHEMA_ROOT",
            "$",
            type(payload).__name__,
            "JSON object with a requirements array",
            "whole_response",
        )
    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        return None, _diagnostic(
            "REQ_SCHEMA_REQUIREMENTS",
            "$.requirements",
            raw_requirements,
            "non-empty requirements array",
            "$.requirements",
        )

    local_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_requirements):
        path = f"$.requirements[{index}]"
        if not isinstance(raw, Mapping):
            return None, _diagnostic(
                "REQ_SCHEMA_ITEM", path, raw, "requirement object", path
            )
        local_id = str(raw.get("local_id") or "").strip().casefold()
        if not _LOCAL_ID.fullmatch(local_id):
            return None, _diagnostic(
                "REQ_LOCAL_ID",
                path + ".local_id",
                raw.get("local_id"),
                "unique lower_snake local ID",
                path,
            )
        if local_id in local_ids:
            return None, _diagnostic(
                "REQ_DUPLICATE_LOCAL_ID",
                path + ".local_id",
                local_id,
                "unique local ID",
                path,
            )
        local_ids.add(local_id)

        capability = str(raw.get("capability_id") or "").strip().casefold()
        if (
            not _CAPABILITY_ID.fullmatch(capability)
            or _OPAQUE_CAPABILITY.match(capability)
        ):
            return None, _diagnostic(
                "REQ_CAPABILITY_ID",
                path + ".capability_id",
                raw.get("capability_id"),
                "meaningful lower-case dotted semantic ID; no semantic hash/unresolved placeholder",
                path,
            )

        role = str(raw.get("provenance_role") or "").strip().casefold()
        if role not in _ALLOWED_AUTHORING_ROLES:
            return None, _diagnostic(
                "REQ_PROVENANCE_OVERREACH",
                path + ".provenance_role",
                raw.get("provenance_role"),
                "explicit or logically_derived only before design-alternative resolution",
                path,
            )

        clause_index = raw.get("source_clause_index")
        if (
            type(clause_index) is not int
            or clause_index < 0
            or clause_index >= len(clauses)
        ):
            return None, _diagnostic(
                "REQ_SOURCE_CLAUSE",
                path + ".source_clause_index",
                clause_index,
                f"integer in [0,{len(clauses) - 1}]",
                path,
            )
        clause = clauses[clause_index]
        quote = str(raw.get("source_quote") or "").strip()
        receipt = _find_quote(clause, quote)
        if receipt is None:
            return None, _diagnostic(
                "REQ_SOURCE_GROUNDING",
                path + ".source_quote",
                quote,
                "one unique verbatim substring inside the assigned host clause",
                path,
            )

        semantic_statement = str(raw.get("semantic_statement") or "").strip()
        if not semantic_statement:
            return None, _diagnostic(
                "REQ_SEMANTIC_STATEMENT",
                path + ".semantic_statement",
                raw.get("semantic_statement"),
                "non-empty language-neutral semantic statement",
                path,
            )
        derived_from = [
            str(value).strip().casefold()
            for value in raw.get("derived_from", [])
            if str(value).strip()
        ] if isinstance(raw.get("derived_from"), list) else []
        depends_on = [
            str(value).strip().casefold()
            for value in raw.get("depends_on", [])
            if str(value).strip()
        ] if isinstance(raw.get("depends_on"), list) else []
        reason = str(raw.get("derivation_reason") or "").strip()
        if role == "explicit" and derived_from:
            return None, _diagnostic(
                "REQ_EXPLICIT_DERIVATION",
                path + ".derived_from",
                derived_from,
                "explicit requirements do not derive from other requirements",
                path,
            )
        if role == "logically_derived" and (not derived_from or not reason):
            return None, _diagnostic(
                "REQ_DERIVATION_PROOF",
                path,
                {"derived_from": derived_from, "derivation_reason": reason},
                "logically_derived requires derived_from plus non-empty derivation_reason",
                path,
            )

        behavior = raw.get("observable_behavior")
        if not isinstance(behavior, Mapping):
            return None, _diagnostic(
                "REQ_ACCEPTANCE_OBJECT",
                path + ".observable_behavior",
                behavior,
                "object with non-empty given/when/then",
                path,
            )
        given = str(behavior.get("given") or "").strip()
        when = str(behavior.get("when") or "").strip()
        then = str(behavior.get("then") or "").strip()
        if not (given and when and then):
            return None, _diagnostic(
                "REQ_ACCEPTANCE_CONCRETE",
                path + ".observable_behavior",
                behavior,
                "non-empty given, when and then strings",
                path,
            )

        source_start, source_end = receipt
        normalized.append(
            {
                "local_id": local_id,
                "capability_id": capability,
                "provenance_role": role,
                "source_clause_index": clause_index,
                "source_quote": quote,
                "source_start": source_start,
                "source_end": source_end,
                "semantic_statement": semantic_statement,
                "derived_from": derived_from,
                "depends_on": depends_on,
                "derivation_reason": reason,
                "observable_behavior": {
                    "given": given,
                    "when": when,
                    "then": then,
                },
            }
        )

    known = {item["local_id"] for item in normalized}
    for index, item in enumerate(normalized):
        for field in ("derived_from", "depends_on"):
            unknown = [value for value in item[field] if value not in known]
            if unknown:
                return None, _diagnostic(
                    "REQ_GRAPH_UNKNOWN_REFERENCE",
                    f"$.requirements[{index}].{field}",
                    unknown,
                    "local IDs declared in the same requirements graph",
                    f"$.requirements[{index}]",
                )
            if item["local_id"] in item[field]:
                return None, _diagnostic(
                    "REQ_GRAPH_SELF_REFERENCE",
                    f"$.requirements[{index}].{field}",
                    item[field],
                    "acyclic reference set without self-reference",
                    f"$.requirements[{index}]",
                )

    explicit_by_clause = {
        int(item["source_clause_index"])
        for item in normalized
        if item["provenance_role"] == "explicit"
    }
    uncovered = [
        int(clause["clause_index"])
        for clause in clauses
        if int(clause["clause_index"]) not in explicit_by_clause
    ]
    if uncovered:
        return None, _diagnostic(
            "REQ_SOURCE_COVERAGE",
            "$.requirements",
            uncovered,
            "at least one explicit semantic requirement for every authored clause",
            "$.requirements",
        )

    for clause in clauses:
        members = [
            item
            for item in normalized
            if item["provenance_role"] == "explicit"
            and item["source_clause_index"] == clause["clause_index"]
        ]
        if len(members) > 1 and all(
            item["source_quote"] == clause["text"] for item in members
        ):
            return None, _diagnostic(
                "REQ_SOURCE_AMBIGUITY",
                "$.requirements",
                {
                    "source_clause_index": clause["clause_index"],
                    "local_ids": [item["local_id"] for item in members],
                },
                "multiple independent meanings need distinct smallest verbatim source_quote grounding",
                "$.requirements",
            )

    graph = {
        item["local_id"]: tuple(dict.fromkeys((*item["derived_from"], *item["depends_on"])))
        for item in normalized
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visited:
            return True
        if node in visiting:
            return False
        visiting.add(node)
        for parent in graph[node]:
            if not visit(parent):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    for node in graph:
        if not visit(node):
            return None, _diagnostic(
                "REQ_GRAPH_CYCLE",
                "$.requirements",
                graph,
                "acyclic semantic/progression dependency graph",
                "$.requirements",
            )

    return normalized, None


def _parse_json(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"semantic response type {type(raw).__name__} is not JSON text/object")


def _generate_approved_nodes(
    prompt: str,
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Approve authored meaning one host clause at a time.

    The model owns only the semantic description of the current authored clause.
    The host owns provenance, graph identity, clause identity, and all cross-clause
    references. A malformed response therefore invalidates only its own clause.
    """

    parameters = {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "capability_id": {"type": "string", "minLength": 3},
                        "source_quote": {"type": "string", "minLength": 1},
                        "semantic_statement": {"type": "string", "minLength": 1},
                        "given": {"type": "string", "minLength": 1},
                        "when": {"type": "string", "minLength": 1},
                        "then": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "capability_id", "source_quote", "semantic_statement",
                        "given", "when", "then",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["requirements"],
        "additionalProperties": False,
    }

    approved: list[dict[str, Any]] = []
    for ordinal, clause in enumerate(clauses):
        clause_index = int(clause["clause_index"])
        diagnostic: dict[str, Any] | None = None
        seen_failures: set[str] = set()
        accepted = False
        for attempt in range(3):
            request_payload = {
                "current_clause_index": clause_index,
                "current_clause": str(clause["text"]),
                "previous_clause_context": (
                    str(clauses[ordinal - 1]["text"]) if ordinal > 0 else ""
                ),
                "next_clause_context": (
                    str(clauses[ordinal + 1]["text"])
                    if ordinal + 1 < len(clauses)
                    else ""
                ),
                "repair_diagnostic": diagnostic,
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Interpret exactly the current authored clause. Do not add design choices, "
                        "implementation classes, APIs, dependencies, or requirements not stated by "
                        "that clause. Split only when the current clause independently states more "
                        "than one observable requirement. capability_id must be a meaningful lower-"
                        "case dotted semantic identifier, never an opaque hash. source_quote must be "
                        "the smallest unique verbatim substring of current_clause that grounds that "
                        "requirement. Adjacent clauses are context only and may not become new output "
                        "requirements. Return concrete Given/When/Then observable behavior."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False, sort_keys=True),
                },
            ]
            try:
                native = getattr(router, "generate_tool_decision", None)
                if callable(native):
                    payload = native(
                        "planner",
                        messages,
                        tool_name="approve_semantic_clause",
                        parameters=parameters,
                        description=(
                            "Return only semantic requirements grounded in the current authored clause."
                        ),
                    )
                else:
                    raw = router.generate_text(
                        "planner",
                        messages,
                        response_format="text",
                        response_schema=None,
                        enable_tools=False,
                    )
                    payload = _parse_json(raw)
            except Exception as exc:
                payload = {}
                diagnostic = _diagnostic(
                    "REQ_CLAUSE_MODEL_RESPONSE",
                    f"$.clauses[{clause_index}]",
                    f"{type(exc).__name__}: {exc}",
                    "one compact clause-local semantic payload",
                    f"clause:{clause_index}",
                )
            else:
                raw_requirements = (
                    payload.get("requirements") if isinstance(payload, Mapping) else None
                )
                if not isinstance(raw_requirements, list) or not raw_requirements:
                    diagnostic = _diagnostic(
                        "REQ_CLAUSE_SCHEMA",
                        f"$.clauses[{clause_index}].requirements",
                        raw_requirements,
                        "non-empty clause-local requirements array",
                        f"clause:{clause_index}",
                    )
                else:
                    raw_nodes: list[dict[str, Any]] = []
                    for item_index, item in enumerate(raw_requirements):
                        if not isinstance(item, Mapping):
                            raw_nodes.append({})
                            continue
                        raw_nodes.append(
                            {
                                "local_id": f"c{clause_index}_{item_index}",
                                "capability_id": item.get("capability_id"),
                                "provenance_role": "explicit",
                                "source_clause_index": 0,
                                "source_quote": item.get("source_quote"),
                                "semantic_statement": item.get("semantic_statement"),
                                "derived_from": [],
                                "depends_on": [],
                                "derivation_reason": "",
                                "observable_behavior": {
                                    "given": item.get("given"),
                                    "when": item.get("when"),
                                    "then": item.get("then"),
                                },
                            }
                        )
                    local_clause = dict(clause)
                    local_clause["clause_index"] = 0
                    local_nodes, validation_error = _validate_candidate(
                        {"requirements": raw_nodes},
                        prompt=prompt,
                        clauses=(local_clause,),
                    )
                    if local_nodes is not None:
                        for node in local_nodes:
                            node["source_clause_index"] = clause_index
                        approved.extend(local_nodes)
                        accepted = True
                        break
                    diagnostic = dict(validation_error or {})
                    diagnostic["repair_scope"] = f"clause:{clause_index}"

            failure_state = _sha256({"diagnostic": diagnostic, "candidate": payload})
            if failure_state in seen_failures:
                raise _evidence.EvidencePlanError(
                    "semantic clause approval reached a no-progress fixed point: "
                    + _canonical(diagnostic)
                )
            seen_failures.add(failure_state)
            if attempt == 2:
                raise _evidence.EvidencePlanError(
                    "semantic clause approval exhausted bounded repair attempts: "
                    + _canonical(diagnostic)
                )

        if not accepted:
            raise _evidence.EvidencePlanError(
                f"semantic clause {clause_index} was not approved"
            )

    merged_payload = {
        "requirements": [
            {
                "local_id": node["local_id"],
                "capability_id": node["capability_id"],
                "provenance_role": "explicit",
                "source_clause_index": node["source_clause_index"],
                "source_quote": node["source_quote"],
                "semantic_statement": node["semantic_statement"],
                "derived_from": [],
                "depends_on": [],
                "derivation_reason": "",
                "observable_behavior": dict(node["observable_behavior"]),
            }
            for node in approved
        ]
    }
    merged, final_error = _validate_candidate(
        merged_payload,
        prompt=prompt,
        clauses=clauses,
    )
    if merged is None:
        raise _evidence.EvidencePlanError(
            "host-merged semantic clauses violated the requirement contract: "
            + _canonical(final_error)
        )
    return merged


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
        derived_from = [requirement_ids[value] for value in item["derived_from"]]
        dependencies = [requirement_ids[value] for value in item["depends_on"]]
        requirements.append(
            {
                "requirement_id": requirement_id,
                "capability": item["capability_id"],
                "statement": str(clause["text"]),
                "semantic_statement": item["semantic_statement"],
                "mandatory": True,
                "provenance_role": item["provenance_role"],
                "source_span": {
                    "source_id": "requested_prompt",
                    "char_start": source_start,
                    "char_end": source_end,
                    "text": quote,
                    "text_sha256": _sha256(quote),
                    "source_clause_index": item["source_clause_index"],
                    "source_clause_sha256": clause["text_sha256"],
                },
                "derived_from": derived_from,
                "depends_on": dependencies,
                "provides": [_evidence._canonical_capability(item["capability_id"])],
                "gameplay_capabilities": [item["capability_id"]],
                "implementation_capabilities": [],
                "artifact_task_ids": [],
                "semantic_status": "RESOLVED",
                "unresolved_spans": [],
                "acceptance": [
                    (
                        f"Given {behavior['given']}; when {behavior['when']}; "
                        f"then {behavior['then']}."
                    )
                ],
                "observable_behavior": behavior,
            }
        )

    edges = sorted(
        {
            (parent, requirement["requirement_id"], "derived_from")
            for requirement in requirements
            for parent in requirement["derived_from"]
        }
        | {
            (parent, requirement["requirement_id"], "depends_on")
            for requirement in requirements
            for parent in requirement["depends_on"]
        }
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
            "edges": [
                {"from": source, "to": target, "relation": relation}
                for source, target, relation in edges
            ],
        },
        "semantic_audit": {
            "status": "APPROVED",
            "authored_clause_count": len(clauses),
            "covered_clause_count": len(clauses),
            "unresolved_clause_count": 0,
            "unsupported_design_choice_count": 0,
            "provenance_roles": sorted(_ALL_PROVENANCE_ROLES),
        },
        "catalog_sha256": "",
    }
    payload["catalog_sha256"] = _evidence._hash_without(payload, "catalog_sha256")
    return payload


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
        if _OPAQUE_CAPABILITY.match(capability) or not _CAPABILITY_ID.fullmatch(capability):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_CAPABILITY: invalid capability {capability!r}."
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
        acceptance = raw.get("observable_behavior")
        if not isinstance(acceptance, Mapping) or not all(
            str(acceptance.get(field) or "").strip()
            for field in ("given", "when", "then")
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_ACCEPTANCE: concrete observable contract missing for {requirement_id}."
            )

    for raw in requirements:
        for field in ("derived_from", "depends_on"):
            refs = raw.get(field, [])
            if not isinstance(refs, list) or any(str(ref) not in ids for ref in refs):
                raise _evidence.EvidencePlanError(
                    f"REQ_AUTHORITY_GRAPH: {field} contains unknown requirement IDs."
                )

    audit = catalog.get("semantic_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "APPROVED"
        or audit.get("unresolved_clause_count") != 0
        or audit.get("unsupported_design_choice_count") != 0
    ):
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_BARRIER: semantic coverage/overreach audit is not approved."
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
    """Install the approved requirement graph as the sole production semantic authority."""

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
