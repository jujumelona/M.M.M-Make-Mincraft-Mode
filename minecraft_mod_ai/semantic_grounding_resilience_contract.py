from __future__ import annotations

"""Adaptive host grounding for semantic requirements.

The model supplies semantic locators, not byte offsets. Host provenance therefore must not
fail merely because a locator is short, repeated, or has a small copy error. Grounding uses
the full semantic payload as evidence and derives exact source offsets from the authored
clause. No fixed anchor-length or similarity cutoff decides whether a requirement survives.
"""

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
import re
from typing import Any

_INSTALLED = False
_NORMALIZE_MARKER = "__mmm_adaptive_semantic_grounding__"
_EVALUATE_MARKER = "__mmm_leaf_preserving_semantic_batch__"
_WORD = re.compile(r"\w+", re.UNICODE)


def _terms(value: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0).casefold() for match in _WORD.finditer(str(value or ""))))


def _raw_span_from_projection(
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


def _semantic_support_terms(
    text_form: str,
    raw: Mapping[str, Any],
) -> tuple[str, ...]:
    values = (
        raw.get("source_anchor"),
        raw.get("semantic_statement"),
        raw.get("given"),
        raw.get("when"),
        raw.get("then"),
    )
    supported: list[str] = []
    for value in values:
        for term in _terms(value):
            normalized = "".join(character for character in term if not character.isspace())
            if normalized and normalized in text_form:
                supported.append(normalized)
    return tuple(dict.fromkeys(supported))


def _ground_requirement(
    semantic: Any,
    clause: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any] | None:
    text = str(clause["text"])
    anchor = str(raw.get("source_anchor") or "").strip()
    if not anchor:
        return None

    anchor_form, _ = semantic._similarity_projection(anchor)
    text_form, text_positions = semantic._similarity_projection(text)
    if not anchor_form or not text_form:
        return None

    # Normalized exact grounding works for anchors of any length. If the same semantic
    # locator appears more than once, the first authored occurrence is deterministic; the
    # clause receipt still preserves the full source context for auditability.
    exact_start = text_form.find(anchor_form)
    if exact_start >= 0:
        start, end, quote = _raw_span_from_projection(
            clause,
            text_positions,
            exact_start,
            exact_start + len(anchor_form),
        )
        return {
            "source_quote": quote,
            "source_start": start,
            "source_end": end,
            "grounding_method": "normalized_exact_host_alignment",
            "grounding_similarity": 1.0,
            "model_anchor": anchor,
        }

    # A fuzzy locator is admissible only when the semantic payload independently shares
    # authored lexical evidence with the clause. This rejects unrelated hallucinated
    # anchors without imposing a magic character-count or ratio threshold.
    support_terms = _semantic_support_terms(text_form, raw)
    if not support_terms:
        return None

    matcher = SequenceMatcher(None, anchor_form, text_form, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return None

    projected_start = min(block.b for block in blocks)
    projected_end = max(block.b + block.size for block in blocks)
    if projected_start >= projected_end:
        return None
    start, end, quote = _raw_span_from_projection(
        clause,
        text_positions,
        projected_start,
        projected_end,
    )
    return {
        "source_quote": quote,
        "source_start": start,
        "source_end": end,
        "grounding_method": "semantic_host_alignment",
        "grounding_similarity": round(matcher.ratio(), 6),
        "model_anchor": anchor,
    }


def _normalize_requirement(
    raw: Any,
    *,
    item_index: int,
    clauses_by_index: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    from . import semantic_requirement_authority as semantic

    path = f"$.requirements[{item_index}]"
    if not isinstance(raw, Mapping):
        return None, semantic._diagnostic(
            "REQ_SCHEMA_ITEM", path, raw, "semantic requirement object", path
        ), None

    clause_index = raw.get("source_clause_index")
    if type(clause_index) is not int or clause_index not in clauses_by_index:
        return None, semantic._diagnostic(
            "REQ_SOURCE_CLAUSE",
            path + ".source_clause_index",
            clause_index,
            f"one supplied host clause index: {sorted(clauses_by_index)}",
            path,
        ), None

    capability = str(raw.get("capability_id") or "").strip().casefold()
    if not semantic._CAPABILITY_ID.fullmatch(capability) or semantic._OPAQUE_CAPABILITY.match(capability):
        return None, semantic._diagnostic(
            "REQ_CAPABILITY_ID",
            path + ".capability_id",
            raw.get("capability_id"),
            "meaningful lower-case dotted semantic ID; no opaque semantic hash",
            f"clause:{clause_index}",
        ), clause_index

    semantic_statement = str(raw.get("semantic_statement") or "").strip()
    given = str(raw.get("given") or "").strip()
    when = str(raw.get("when") or "").strip()
    then = str(raw.get("then") or "").strip()
    if not semantic_statement or not (given and when and then):
        return None, semantic._diagnostic(
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

    grounding = _ground_requirement(semantic, clauses_by_index[clause_index], raw)
    if grounding is None:
        return None, semantic._diagnostic(
            "REQ_SOURCE_GROUNDING",
            path + ".source_anchor",
            raw.get("source_anchor"),
            "a semantic locator supported by the authored clause; host owns exact offsets",
            f"clause:{clause_index}",
        ), clause_index

    return {
        "capability_id": capability,
        "provenance_role": "explicit",
        "source_clause_index": clause_index,
        **grounding,
        "semantic_statement": semantic_statement,
        "derived_from": [],
        "depends_on": [],
        "derivation_reason": "",
        "observable_behavior": {"given": given, "when": when, "then": then},
    }, None, clause_index


def _evaluate_batch(
    payload: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, Any]]]:
    """Preserve valid leaves even when another leaf in the same clause needs repair."""

    from . import semantic_requirement_authority as semantic

    clauses_by_index = {int(clause["clause_index"]): clause for clause in clauses}
    all_indices = set(clauses_by_index)
    if not isinstance(payload, Mapping):
        return [], set(all_indices), [semantic._diagnostic(
            "REQ_SCHEMA_ROOT",
            "$",
            type(payload).__name__,
            "JSON object with a requirements array",
            "semantic_batch",
        )]

    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        return [], set(all_indices), [semantic._diagnostic(
            "REQ_SCHEMA_REQUIREMENTS",
            "$.requirements",
            raw_requirements,
            "non-empty requirements array",
            "semantic_batch",
        )]

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
        diagnostics.append(semantic._diagnostic(
            "REQ_SOURCE_COVERAGE",
            "$.requirements",
            clause_index,
            "at least one explicit semantic requirement for every supplied clause",
            f"clause:{clause_index}",
        ))

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


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import semantic_requirement_authority as semantic

    current_normalize = semantic._normalize_requirement
    if not getattr(current_normalize, _NORMALIZE_MARKER, False):
        _normalize_requirement.__wrapped__ = current_normalize  # type: ignore[attr-defined]
        setattr(_normalize_requirement, _NORMALIZE_MARKER, True)
        semantic._normalize_requirement = _normalize_requirement

    current_evaluate = semantic._evaluate_batch
    if not getattr(current_evaluate, _EVALUATE_MARKER, False):
        _evaluate_batch.__wrapped__ = current_evaluate  # type: ignore[attr-defined]
        setattr(_evaluate_batch, _EVALUATE_MARKER, True)
        semantic._evaluate_batch = _evaluate_batch

    _INSTALLED = True


__all__ = ["install"]
