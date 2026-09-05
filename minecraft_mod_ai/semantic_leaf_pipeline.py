from __future__ import annotations

"""Two-stage bounded semantic compilation for small planning models.

Stage 1 owns only semantic segmentation: the model identifies independently observable
authored behaviors and their source anchors. It cannot choose capability IDs.

The host grounds those anchors and enforces an exact language-neutral source partition.
Only after that immutable semantic leaf set is accepted does stage 2 classify each leaf
into one host capability. Classification cannot change source boundaries, statements, or
Given/When/Then behavior. This prevents a weak capability choice from silently shrinking
the authored source span that is supposed to justify it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from . import semantic_requirement_authority as _semantic
from .minecraft_template_catalog import (
    CUSTOM_CAPABILITY_SENTINEL,
    capability_catalog_for_model,
    semantic_capability_choices,
)
from .root_cause_trace import emit_root_cause
from .semantic_source_fidelity import fidelity_router, validate_semantic_source_partition

_MAX_ATTEMPTS = 2
_SEGMENT_FIELDS = frozenset(
    {
        "source_clause_index",
        "source_anchor",
        "semantic_statement",
        "given",
        "when",
        "then",
        "semantic_type",
    }
)
_CLASSIFICATION_FIELDS = frozenset({"leaf_index", "capability_id"})


def _segmentation_schema(max_clause_index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "leaves": {
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
        "required": ["leaves"],
        "additionalProperties": False,
    }


def _classification_schema(max_leaf_index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "minItems": max_leaf_index + 1,
                "maxItems": max_leaf_index + 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "leaf_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max_leaf_index,
                        },
                        "capability_id": {
                            "type": "string",
                            "enum": list(semantic_capability_choices()),
                        },
                    },
                    "required": ["leaf_index", "capability_id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }


def _segmentation_messages(clauses: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    system = (
        "Decompose only the independently observable behaviors explicitly authored in the "
        "supplied Minecraft-mod clauses. This stage is semantic segmentation only. You do "
        "not know and must not emit capability IDs, dependencies, implementation APIs, "
        "versions, paths, task IDs, persistence schemes, networking, UI, progression gates, "
        "or any architecture not explicitly authored. Preserve every authored behavior. "
        "Return concrete Given/When/Then semantics for each leaf. semantic_type is "
        "software_quality only for explicit code/runtime optimization, profiling, latency, "
        "throughput, memory, tick or frame-rate requirements; ordinary gameplay properties "
        "remain gameplay_mechanic."
    )
    payload = {
        "host_owned_clauses": [
            {
                "source_clause_index": int(clause["clause_index"]),
                "text": str(clause["text"]),
            }
            for clause in clauses
        ]
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _semantic._canonical(payload)},
    ]


def _classification_messages(
    leaves: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "Classify each immutable host-grounded semantic leaf into exactly one capability "
        "from host_capability_catalog. The source span and behavior are already approved "
        "and cannot be changed. Choose a catalog capability only when it fully describes "
        "ALL authored behavior in that leaf. Do not substitute a prerequisite, base entity, "
        "base state, storage primitive, networking primitive, or other implementation need "
        "for the directly authored interaction or outcome. The host adds prerequisites only "
        "after semantic classification. If no catalog capability fully covers the leaf, use "
        "custom.semantic. Emit only leaf_index and capability_id."
    )
    if diagnostics:
        system += (
            " Previous classification failed the host contract. Repair only capability "
            "choices using these diagnostics: "
            + _semantic._canonical(list(diagnostics))
        )
    payload = {
        "host_capability_catalog": list(capability_catalog_for_model()),
        "custom_capability_sentinel": CUSTOM_CAPABILITY_SENTINEL,
        "host_grounded_leaves": [
            {
                "leaf_index": index,
                "source_clause_index": int(leaf["source_clause_index"]),
                "source_text": str(leaf["source_quote"]),
                "semantic_statement": str(leaf["semantic_statement"]),
                "given": str(leaf["given"]),
                "when": str(leaf["when"]),
                "then": str(leaf["then"]),
                "semantic_type": str(leaf.get("semantic_type") or "gameplay_mechanic"),
            }
            for index, leaf in enumerate(leaves)
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _semantic._canonical(payload)},
    ]


def _call_model(
    router: Any,
    *,
    operation: str,
    messages: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    description: str,
) -> Any:
    emit_root_cause(
        "planner_model_request",
        stage="planning",
        operation=operation,
        gate="semantic_typing",
        result="START",
        details={"messages": list(messages), "schema": dict(parameters)},
    )
    native = getattr(router, "generate_tool_decision", None)
    if callable(native):
        result = native(
            "planner",
            list(messages),
            tool_name=operation,
            parameters=dict(parameters),
            description=description,
        )
        emit_root_cause(
            "planner_model_response",
            stage="planning",
            operation=operation,
            gate="semantic_typing",
            result="PASS",
            details={"raw_response": result},
        )
        return result

    raw = router.generate_text(
        "planner",
        list(messages),
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    result = _semantic._parse_json(raw)
    emit_root_cause(
        "planner_model_response",
        stage="planning",
        operation=operation,
        gate="semantic_typing",
        result="PASS",
        details={"raw_response": raw, "parsed_response": result},
    )
    return result


def _leaf_diagnostic(
    code: str,
    *,
    path: str,
    value: Any,
    expected: str,
    clause_index: int | None = None,
) -> dict[str, Any]:
    result = {
        "error_code": code,
        "json_path": path,
        "offending_value": value,
        "expected_contract": expected,
    }
    if clause_index is not None:
        result["clause_index"] = clause_index
    return result


def _normalize_segmented_leaves(
    payload: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    clauses_by_index = {int(clause["clause_index"]): clause for clause in clauses}
    diagnostics: list[dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        return [], (
            _leaf_diagnostic(
                "REQ_SEGMENT_SCHEMA_ROOT",
                path="$",
                value=type(payload).__name__,
                expected="JSON object with a leaves array",
            ),
        )
    raw_leaves = payload.get("leaves")
    if not isinstance(raw_leaves, list) or not raw_leaves:
        return [], (
            _leaf_diagnostic(
                "REQ_SEGMENT_SCHEMA_LEAVES",
                path="$.leaves",
                value=raw_leaves,
                expected="non-empty semantic leaves array",
            ),
        )

    leaves: list[dict[str, Any]] = []
    for leaf_index, raw in enumerate(raw_leaves):
        path = f"$.leaves[{leaf_index}]"
        if not isinstance(raw, Mapping):
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_SEGMENT_SCHEMA_ITEM",
                    path=path,
                    value=raw,
                    expected="semantic leaf object",
                )
            )
            continue
        unexpected = sorted(set(raw) - _SEGMENT_FIELDS)
        if unexpected:
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_MODEL_AUTHORITY_OVERREACH",
                    path=path,
                    value=unexpected,
                    expected="semantic segmentation fields only; no capability or planning authority",
                )
            )
            continue
        clause_index = raw.get("source_clause_index")
        if type(clause_index) is not int or clause_index not in clauses_by_index:
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_SOURCE_CLAUSE",
                    path=path + ".source_clause_index",
                    value=clause_index,
                    expected=f"one supplied host clause index: {sorted(clauses_by_index)}",
                )
            )
            continue
        semantic_statement = str(raw.get("semantic_statement") or "").strip()
        given = str(raw.get("given") or "").strip()
        when = str(raw.get("when") or "").strip()
        then = str(raw.get("then") or "").strip()
        if not semantic_statement or not (given and when and then):
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_SEMANTIC_CONTRACT",
                    path=path,
                    value={
                        "semantic_statement": raw.get("semantic_statement"),
                        "given": raw.get("given"),
                        "when": raw.get("when"),
                        "then": raw.get("then"),
                    },
                    expected="non-empty semantic_statement and concrete given/when/then",
                    clause_index=clause_index,
                )
            )
            continue
        source_anchor = str(raw.get("source_anchor") or "").strip()
        grounding = _semantic._ground_source_anchor(
            clauses_by_index[clause_index],
            source_anchor,
        )
        if grounding is None:
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_SOURCE_GROUNDING",
                    path=path + ".source_anchor",
                    value=source_anchor,
                    expected="a semantic locator supported by the authored clause",
                    clause_index=clause_index,
                )
            )
            continue
        supplied_type = str(raw.get("semantic_type") or "").strip().casefold()
        semantic_type = (
            supplied_type
            if supplied_type in {"gameplay_mechanic", "software_quality"}
            else "gameplay_mechanic"
        )
        leaves.append(
            {
                "source_clause_index": clause_index,
                "source_anchor": source_anchor,
                "semantic_statement": semantic_statement,
                "given": given,
                "when": when,
                "then": then,
                "semantic_type": semantic_type,
                **grounding,
            }
        )

    covered = {int(leaf["source_clause_index"]) for leaf in leaves}
    for clause_index in sorted(set(clauses_by_index) - covered):
        diagnostics.append(
            _leaf_diagnostic(
                "REQ_SOURCE_COVERAGE",
                path="$.leaves",
                value=clause_index,
                expected="at least one semantic leaf for every supplied clause",
                clause_index=clause_index,
            )
        )

    diagnostics.extend(validate_semantic_source_partition(leaves, clauses))
    return leaves, tuple(diagnostics)


def _segment_batch(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    max_clause_index = max(int(clause["clause_index"]) for clause in clauses)
    diagnostics: tuple[dict[str, Any], ...] = ()
    for attempt_index in range(_MAX_ATTEMPTS):
        active_router = fidelity_router(router, diagnostics=diagnostics)
        payload = _call_model(
            active_router,
            operation="segment_semantic_requirements",
            messages=_segmentation_messages(clauses),
            parameters=_segmentation_schema(max_clause_index),
            description=(
                "Segment every authored behavior into host-grounded semantic leaves; "
                "do not classify capabilities."
            ),
        )
        leaves, diagnostics = _normalize_segmented_leaves(payload, clauses)
        if not diagnostics:
            return leaves, attempt_index + 1
    raise _semantic._evidence.EvidencePlanError(
        "semantic segmentation rejected after one diagnostic-guided repair: "
        + _semantic._canonical(list(diagnostics))
    )


def _classification_diagnostics(
    payload: Any,
    leaf_count: int,
) -> tuple[dict[int, str] | None, tuple[dict[str, Any], ...]]:
    diagnostics: list[dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        return None, (
            _leaf_diagnostic(
                "REQ_CLASSIFY_SCHEMA_ROOT",
                path="$",
                value=type(payload).__name__,
                expected="JSON object with classifications array",
            ),
        )
    raw_items = payload.get("classifications")
    if not isinstance(raw_items, list):
        return None, (
            _leaf_diagnostic(
                "REQ_CLASSIFY_SCHEMA_ITEMS",
                path="$.classifications",
                value=raw_items,
                expected="one classification per immutable semantic leaf",
            ),
        )

    choices = set(semantic_capability_choices())
    result: dict[int, str] = {}
    for item_index, raw in enumerate(raw_items):
        path = f"$.classifications[{item_index}]"
        if not isinstance(raw, Mapping):
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_CLASSIFY_SCHEMA_ITEM",
                    path=path,
                    value=raw,
                    expected="classification object",
                )
            )
            continue
        unexpected = sorted(set(raw) - _CLASSIFICATION_FIELDS)
        if unexpected:
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_MODEL_AUTHORITY_OVERREACH",
                    path=path,
                    value=unexpected,
                    expected="leaf_index and capability_id only",
                )
            )
            continue
        leaf_index = raw.get("leaf_index")
        capability = str(raw.get("capability_id") or "").strip().casefold()
        if type(leaf_index) is not int or not 0 <= leaf_index < leaf_count:
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_CLASSIFY_LEAF_INDEX",
                    path=path + ".leaf_index",
                    value=leaf_index,
                    expected=f"integer in range 0..{leaf_count - 1}",
                )
            )
            continue
        if leaf_index in result:
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_CLASSIFY_DUPLICATE",
                    path=path + ".leaf_index",
                    value=leaf_index,
                    expected="each immutable leaf classified exactly once",
                )
            )
            continue
        if capability not in choices:
            diagnostics.append(
                _leaf_diagnostic(
                    "REQ_CAPABILITY_CATALOG",
                    path=path + ".capability_id",
                    value=raw.get("capability_id"),
                    expected="one exact host capability enum value or custom.semantic",
                )
            )
            continue
        result[leaf_index] = capability

    missing = sorted(set(range(leaf_count)) - set(result))
    if missing:
        diagnostics.append(
            _leaf_diagnostic(
                "REQ_CLASSIFY_COVERAGE",
                path="$.classifications",
                value=missing,
                expected="classification for every immutable semantic leaf",
            )
        )
    return (result if not diagnostics else None), tuple(diagnostics)


def _classify_leaves(
    router: Any,
    leaves: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, str], int]:
    diagnostics: tuple[dict[str, Any], ...] = ()
    for attempt_index in range(_MAX_ATTEMPTS):
        payload = _call_model(
            router,
            operation="classify_semantic_requirements",
            messages=_classification_messages(leaves, diagnostics),
            parameters=_classification_schema(len(leaves) - 1),
            description=(
                "Classify immutable host-grounded semantic leaves into exact host "
                "capabilities without changing authored semantics."
            ),
        )
        result, diagnostics = _classification_diagnostics(payload, len(leaves))
        if result is not None:
            return result, attempt_index + 1
    raise _semantic._evidence.EvidencePlanError(
        "semantic capability classification rejected after one diagnostic-guided repair: "
        + _semantic._canonical(list(diagnostics))
    )


def compile_semantic_batch(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return approved semantic nodes plus exact bounded-model call accounting."""

    leaves, segmentation_attempts = _segment_batch(router, clauses)
    classifications, classification_attempts = _classify_leaves(router, leaves)

    raw_requirements = []
    for leaf_index, leaf in enumerate(leaves):
        raw_requirements.append(
            {
                "source_clause_index": int(leaf["source_clause_index"]),
                "capability_id": classifications[leaf_index],
                "source_anchor": str(leaf["source_anchor"]),
                "semantic_statement": str(leaf["semantic_statement"]),
                "given": str(leaf["given"]),
                "when": str(leaf["when"]),
                "then": str(leaf["then"]),
                "semantic_type": str(leaf["semantic_type"]),
            }
        )

    nodes, invalid_clauses, diagnostics = _semantic._evaluate_batch(
        {"requirements": raw_requirements},
        clauses,
    )
    fidelity_diagnostics = validate_semantic_source_partition(nodes, clauses)
    if invalid_clauses or diagnostics or fidelity_diagnostics:
        raise _semantic._evidence.EvidencePlanError(
            "host invariant failed after immutable semantic classification: "
            + _semantic._canonical(
                {
                    "invalid_clause_indices": sorted(invalid_clauses),
                    "semantic_diagnostics": diagnostics,
                    "source_fidelity_diagnostics": list(fidelity_diagnostics),
                }
            )
        )

    return nodes, {
        "segmentation_attempts": segmentation_attempts,
        "classification_attempts": classification_attempts,
        "semantic_model_calls_total": segmentation_attempts + classification_attempts,
        "semantic_repair_turns_used": max(0, segmentation_attempts - 1)
        + max(0, classification_attempts - 1),
        "segmentation_repaired": segmentation_attempts > 1,
        "classification_repaired": classification_attempts > 1,
    }


__all__ = ["compile_semantic_batch"]
