from __future__ import annotations

"""Host-owned semantic requirement authority.

The language model performs semantic interpretation only. The host owns authored source
text, exact offsets, provenance, stable IDs, and downstream authority. Semantic analysis
is batched so normal planning needs exactly one model turn. Invalid or ungrounded semantic
output fails closed at the host boundary; there is no semantic retry or repair turn.
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
_OPAQUE_CAPABILITY = re.compile(r"^(?:semantic_[0-9a-f]{6,}|unresolved:)", re.IGNORECASE)
_WORD = re.compile(r"\w+", re.UNICODE)


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
                        "capability_id": {"type": "string", "minLength": 3, "maxLength": 128},
                        "source_anchor": {"type": "string", "minLength": 1},
                        "semantic_statement": {"type": "string", "minLength": 1},
                        "given": {"type": "string", "minLength": 1},
                        "when": {"type": "string", "minLength": 1},
                        "then": {"type": "string", "minLength": 1},
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
        "Interpret only the authored Minecraft-mod requirements in the supplied host clauses. "
        "The request may contain typos, missing spaces or punctuation, shorthand, repeated words, "
        "or non-English text: normalize the MEANING, never the host-owned provenance text. "
        "Return every independent player-visible requirement, and cover every supplied clause "
        "at least once. A clause may contain many independent behaviors even without punctuation: "
        "split conjunctions, sequences, lists, resource flows, state transitions, purchases, "
        "assembly steps, upgrades, travel phases, combat outcomes, world interactions, and "
        "persistence-visible outcomes whenever each can be implemented and observed independently. "
        "A genre, theme, setting, category, or mode label is CONTEXT, not a separate requirement, "
        "unless the authored text gives that label its own independently observable behavior. "
        "Never create a generic integration/context umbrella node merely to make other requirements "
        "depend on it. Never compress multiple verbs or player-visible outcomes into an umbrella "
        "requirement to shorten the response. Preserve authored MODALITY exactly: can/may/optionally "
        "means the behavior is available, not that the player must perform it before another action. "
        "Do not strengthen an available upgrade, purchase, combat, exploration, construction step, "
        "or other option into a mandatory prerequisite unless the authored text explicitly says it "
        "is required. Preserve an explicit causal/temporal prerequisite only when the author actually "
        "states that the later behavior requires the earlier state; mere mention order is not enough. "
        "Do not add common genre mechanics or plausible consequences that the author did not request: "
        "for example, a broad activity such as colonization, trading, exploration, or upgrading must "
        "not silently acquire extra subfeatures such as base building, extraction, blueprints, NPC "
        "roles, mandatory progression gates, or other details unless they are authored. "
        "The authored request determines requirement cardinality; there is no fixed leaf target or "
        "per-clause leaf ceiling. Do not choose implementation classes, APIs, persistence/networking "
        "schemes, UI patterns, boss variants, blueprint systems, or any other design alternative "
        "unless that behavior is explicitly authored. capability_id must be a meaningful lower-case "
        "dotted semantic identifier and never an opaque hash. source_clause_index must identify the "
        "supplied host clause. source_anchor is only a short semantic locator; it does NOT need to be "
        "a byte-for-byte copy because the host, not the model, owns exact source text and offsets. "
        "Keep the anchor close to the smallest authored phrase that supports the requirement. Return "
        "one concrete Given/When/Then observable behavior for each leaf. Every condition and outcome "
        "in Given/When/Then must be supported by the authored request; do not fill missing details from "
        "Minecraft conventions or general game-design knowledge. Do not emit provenance roles, local "
        "IDs, dependencies, source offsets, or hashes."
    )
    payload = {
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
    native = getattr(router, "generate_tool_decision", None)
    if callable(native):
        return native(
            "planner",
            messages,
            tool_name="compile_semantic_requirements",
            parameters=parameters,
            description=(
                "Compile every independently observable authored behavior into semantic "
                "leaf requirements. The host owns exact source grounding."
            ),
        )
    raw = router.generate_text(
        "planner",
        messages,
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    return _parse_json(raw)


def _similarity_projection(value: str) -> tuple[str, list[int]]:
    """Normalize for matching while retaining a map back to raw character offsets."""

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
    """Resolve a model semantic locator to an exact host-owned source span.

    The locator may be short, repeated, whitespace-normalized, or contain a copy error.
    Grounding is based on evidence in the authored clause, not magic character counts or
    fixed similarity thresholds. Exact host text always remains the provenance authority.
    A typo-tolerant locator must itself retain authored lexical evidence; a related semantic
    statement cannot bootstrap provenance for an unrelated locator.
    """

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


def _normalize_requirement(
    raw: Any,
    *,
    item_index: int,
    clauses_by_index: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    path = f"$.requirements[{item_index}]"
    if not isinstance(raw, Mapping):
        return (
            None,
            _diagnostic(
                "REQ_SCHEMA_ITEM",
                path,
                raw,
                "semantic requirement object",
                path,
            ),
            None,
        )

    clause_index = raw.get("source_clause_index")
    if type(clause_index) is not int or clause_index not in clauses_by_index:
        return (
            None,
            _diagnostic(
                "REQ_SOURCE_CLAUSE",
                path + ".source_clause_index",
                clause_index,
                f"one supplied host clause index: {sorted(clauses_by_index)}",
                path,
            ),
            None,
        )

    capability = str(raw.get("capability_id") or "").strip().casefold()
    if not _CAPABILITY_ID.fullmatch(capability) or _OPAQUE_CAPABILITY.match(capability):
        return (
            None,
            _diagnostic(
                "REQ_CAPABILITY_ID",
                path + ".capability_id",
                raw.get("capability_id"),
                "meaningful lower-case dotted semantic ID; no opaque semantic hash",
                f"clause:{clause_index}",
            ),
            clause_index,
        )

    semantic_statement = str(raw.get("semantic_statement") or "").strip()
    given = str(raw.get("given") or "").strip()
    when = str(raw.get("when") or "").strip()
    then = str(raw.get("then") or "").strip()
    if not semantic_statement or not (given and when and then):
        return (
            None,
            _diagnostic(
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
            ),
            clause_index,
        )

    source_anchor = str(raw.get("source_anchor") or "").strip()
    grounding = _ground_source_anchor(clauses_by_index[clause_index], source_anchor)
    if grounding is None:
        return (
            None,
            _diagnostic(
                "REQ_SOURCE_GROUNDING",
                path + ".source_anchor",
                source_anchor,
                (
                    "a semantic locator supported by the authored clause; host owns exact "
                    "source offsets and does not require a minimum locator length"
                ),
                f"clause:{clause_index}",
            ),
            clause_index,
        )

    node = {
        "capability_id": capability,
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
    }
    return node, None, clause_index


def _evaluate_batch(
    payload: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, Any]]]:
    clauses_by_index = {int(clause["clause_index"]): clause for clause in clauses}
    all_indices = set(clauses_by_index)
    if not isinstance(payload, Mapping):
        diagnostic = _diagnostic(
            "REQ_SCHEMA_ROOT",
            "$",
            type(payload).__name__,
            "JSON object with a requirements array",
            "semantic_batch",
        )
        return [], set(all_indices), [diagnostic]

    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        diagnostic = _diagnostic(
            "REQ_SCHEMA_REQUIREMENTS",
            "$.requirements",
            raw_requirements,
            "non-empty requirements array",
            "semantic_batch",
        )
        return [], set(all_indices), [diagnostic]

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
    """Compile all authored clauses in exactly one semantic-model turn."""

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
                    "one batched semantic requirements payload",
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
        requirements.append(
            {
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
            "generation_policy": "single_pass_constrained",
            "source_grounding_owner": "host",
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
        clause_index = span.get("source_clause_index")
        if type(clause_index) is not int:
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_SOURCE: invalid clause index for {requirement_id}."
            )
        covered_clauses.add(clause_index)

        acceptance = raw.get("observable_behavior")
        if not isinstance(acceptance, Mapping) or not all(
            str(acceptance.get(field) or "").strip()
            for field in ("given", "when", "then")
        ):
            raise _evidence.EvidencePlanError(
                f"REQ_AUTHORITY_ACCEPTANCE: concrete observable contract missing for {requirement_id}."
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

    audit = catalog.get("semantic_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "APPROVED"
        or audit.get("unresolved_clause_count") != 0
        or audit.get("unsupported_design_choice_count") != 0
        or audit.get("source_grounding_owner") != "host"
    ):
        raise _evidence.EvidencePlanError(
            "REQ_AUTHORITY_BARRIER: semantic coverage/overreach audit is not approved."
        )
    expected_clause_count = audit.get("authored_clause_count")
    if (
        type(expected_clause_count) is not int
        or len(covered_clauses) != expected_clause_count
    ):
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
