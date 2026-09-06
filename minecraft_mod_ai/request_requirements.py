from __future__ import annotations

"""Authority-neutral semantic atomicity validation and bounded re-segmentation.

The model may split a compound semantic leaf, but the host owns exact source provenance.
Model anchors only have to ground each atomic behavior inside the parent span. Any remaining
non-executable connective/context text is retained deterministically as host-owned provenance
residue instead of asking the small model to reproduce an exact character partition.
"""

import enum
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


class LeafAtomicityStatus(str, enum.Enum):
    ATOMIC = "ATOMIC"
    COMPOUND = "COMPOUND"
    CONTEXT = "CONTEXT"
    CATCH_ALL = "CATCH_ALL"


# These patterns classify only non-executable framing. They do not map text to gameplay
# capabilities and therefore cannot manufacture semantic requirements.
_GENRE_CONTEXT_PATTERNS = (
    re.compile(r"^.{0,80}\bmod(?:e)?\b\s*(?:only|theme)?\.?$", re.IGNORECASE),
    re.compile(r"^.{0,80}모드\s*(?:인데|이고|입니다|이다|임)?\s*$", re.IGNORECASE),
)
_CATCH_ALL_PATTERNS = (
    re.compile(r"^(?:등\s*여러\s*가지(?:가\s*가능한\s*모드)?|기타\s*활동|기타\s*기능|등등)\s*$", re.IGNORECASE),
    re.compile(r"^(?:other\s+activities|and\s+more|etc\.?|and\s+other\s+activities|various\s+others)\s*$", re.IGNORECASE),
)

# Generic action families are used only as a conservative compound signal. They are not
# capability IDs and they do not provide Given/When/Then content.
_ACTION_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "create",
        re.compile(
            r"(?:만들|제작|조립|건설|설치|생성|craft|build|create|assemble|construct|place)",
            re.IGNORECASE,
        ),
    ),
    (
        "gather",
        re.compile(
            r"(?:파밍|채굴|수확|수집|모으|획득|gather|farm|mine|mining|harvest|collect|obtain)",
            re.IGNORECASE,
        ),
    ),
    (
        "trade",
        re.compile(r"(?:거래|교환|구매|판매|trade|exchange|buy|purchase|sell)", re.IGNORECASE),
    ),
    (
        "upgrade",
        re.compile(r"(?:업그레이드|강화|개선|upgrade|enhance|improve)", re.IGNORECASE),
    ),
    (
        "expand",
        re.compile(r"(?:확장|증설|expand|extend|increase\s+capacity)", re.IGNORECASE),
    ),
    (
        "manage",
        re.compile(r"(?:고용|배치|관리|모집|recruit|hire|assign|manage)", re.IGNORECASE),
    ),
    (
        "travel",
        re.compile(r"(?:이동|비행|출발|나가|진출|travel|launch|fly|move|teleport|leave)", re.IGNORECASE),
    ),
    (
        "discover",
        re.compile(r"(?:탐색|발견|찾|explore|discover|find|locate)", re.IGNORECASE),
    ),
    (
        "combat",
        re.compile(r"(?:싸움|싸우|전투|공격|방어|fight|combat|battle|attack|defend)", re.IGNORECASE),
    ),
    (
        "settle",
        re.compile(r"(?:식민|정착|settle|coloniz|found\s+(?:a\s+)?colony)", re.IGNORECASE),
    ),
    (
        "produce",
        re.compile(r"(?:생산|가공|합성|분해|요리|produce|process|smelt|cook|combine|refine)", re.IGNORECASE),
    ),
    (
        "interact",
        re.compile(r"(?:사용|열|닫|상호작용|선택|변경|use|open|close|interact|select|change)", re.IGNORECASE),
    ),
    (
        "care",
        re.compile(r"(?:치료|회복|길들이|번식|heal|recover|tame|breed)", re.IGNORECASE),
    ),
)
_PARALLEL_JOIN = re.compile(
    r"(?:,|/|\band\b|\bor\b|\bthen\b|및|그리고|하거나|또는|하고|하며)",
    re.IGNORECASE,
)
_CAPABILITY_IDENTIFIER = re.compile(
    r"\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\b", re.IGNORECASE
)
_ENGLISH_ACTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "create": re.compile(r"\b(?:craft|build|create|assemble|construct|place)\b", re.IGNORECASE),
    "gather": re.compile(r"\b(?:gather|farm|mine|mining|harvest|collect|obtain)\b", re.IGNORECASE),
    "trade": re.compile(r"\b(?:trade|exchange|buy|purchase|sell)\b", re.IGNORECASE),
    "upgrade": re.compile(r"\b(?:upgrade|enhance|improve)\b", re.IGNORECASE),
    "expand": re.compile(r"\b(?:expand|extend|increase\s+capacity)\b", re.IGNORECASE),
    "manage": re.compile(r"\b(?:recruit|hire|assign|manage)\b", re.IGNORECASE),
    "travel": re.compile(r"\b(?:travel|launch|fly|move|teleport|leave)\b", re.IGNORECASE),
    "discover": re.compile(r"\b(?:explore|discover|find|locate)\b", re.IGNORECASE),
    "combat": re.compile(r"\b(?:fight|combat|battle|attack|defend)\b", re.IGNORECASE),
    "settle": re.compile(
        r"\b(?:settle|coloniz(?:e|es|ed|ing|ation)?|found\s+(?:a\s+)?colony)\b",
        re.IGNORECASE,
    ),
    "produce": re.compile(r"\b(?:produce|process|smelt|cook|combine|refine)\b", re.IGNORECASE),
    "interact": re.compile(r"\b(?:use|open|close|interact|select|change)\b", re.IGNORECASE),
    "care": re.compile(r"\b(?:heal|recover|tame|breed)\b", re.IGNORECASE),
}
_RESEGMENT_FIELDS = frozenset(
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
_MAX_RESEGMENT_GROUPS = 8


def is_genre_context(statement: str, anchor: str = "") -> bool:
    """Return True only for a standalone theme/mod-description leaf."""
    candidates = (" ".join(statement.strip().split()), " ".join(anchor.strip().split()))
    return any(
        candidate and pattern.fullmatch(candidate)
        for candidate in candidates
        for pattern in _GENRE_CONTEXT_PATTERNS
    )


def is_pure_catch_all(statement: str, anchor: str = "") -> bool:
    """Return True only when a leaf is entirely non-verifiable catch-all text."""
    candidates = (
        re.sub(r"[.,;!?]+$", "", " ".join(statement.strip().split())),
        re.sub(r"[.,;!?]+$", "", " ".join(anchor.strip().split())),
    )
    return any(
        candidate and pattern.fullmatch(candidate)
        for candidate in candidates
        for pattern in _CATCH_ALL_PATTERNS
    )


def detected_action_families(text: str) -> tuple[str, ...]:
    """Return generic behavior families present in prose without treating IDs as prose."""
    prose = _CAPABILITY_IDENTIFIER.sub(" ", str(text or ""))
    non_english = re.sub(r"[A-Za-z]+", " ", prose)
    return tuple(
        name
        for name, pattern in _ACTION_FAMILIES
        if pattern.search(non_english) or _ENGLISH_ACTION_PATTERNS[name].search(prose)
    )


def _parallel_target_signal(text: str, action_count: int) -> bool:
    joins = len(_PARALLEL_JOIN.findall(text))
    return action_count >= 2 or (action_count == 1 and joins >= 2)


def validate_leaf_atomicity(
    leaf: Mapping[str, Any],
    clause_text: str = "",
) -> tuple[LeafAtomicityStatus, str]:
    """Conservatively reject leaves that visibly bundle independent behaviors."""
    del clause_text
    statement = str(leaf.get("semantic_statement") or "").strip()
    anchor = str(leaf.get("source_anchor") or "").strip()

    if is_pure_catch_all(statement, anchor):
        return (
            LeafAtomicityStatus.CATCH_ALL,
            f"Leaf {statement!r} is an unverifiable catch-all requirement.",
        )
    if is_genre_context(statement, anchor):
        return (
            LeafAtomicityStatus.CONTEXT,
            f"Leaf {statement!r} is request context rather than an executable behavior.",
        )

    statement_actions = detected_action_families(statement)
    if _parallel_target_signal(statement, len(statement_actions)):
        return (
            LeafAtomicityStatus.COMPOUND,
            "Leaf contains multiple independently observable action families: "
            + ", ".join(statement_actions),
        )

    return LeafAtomicityStatus.ATOMIC, ""


def filter_and_split_context(
    leaves: Sequence[Mapping[str, Any]],
    clause_text: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition model leaves without inventing replacements for rejected leaves."""
    atomic_leaves: list[dict[str, Any]] = []
    compound_leaves: list[dict[str, Any]] = []
    context_leaves: list[dict[str, Any]] = []
    dropped_catch_alls: list[dict[str, Any]] = []
    for leaf in leaves:
        status, reason = validate_leaf_atomicity(leaf, clause_text)
        if status == LeafAtomicityStatus.CONTEXT:
            context_leaves.append(dict(leaf))
        elif status == LeafAtomicityStatus.CATCH_ALL:
            dropped_catch_alls.append(dict(leaf))
        elif status == LeafAtomicityStatus.COMPOUND:
            compound_leaves.append({**dict(leaf), "_atomicity_violation": reason})
        else:
            atomic_leaves.append(dict(leaf))
    return atomic_leaves, compound_leaves, context_leaves, dropped_catch_alls


def _resegment_leaf_schema(max_clause_index: int) -> dict[str, Any]:
    return {
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
    }


def _resegment_schema(group_ids: Sequence[str], max_clause_index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "minItems": len(group_ids),
                "maxItems": len(group_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string", "enum": list(group_ids)},
                        "leaves": {
                            "type": "array",
                            "minItems": 1,
                            "items": _resegment_leaf_schema(max_clause_index),
                        },
                    },
                    "required": ["group_id", "leaves"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["groups"],
        "additionalProperties": False,
    }


def _resegment_messages(
    items: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]]
) -> list[dict[str, str]]:
    from . import semantic_requirement_authority as _semantic

    system = (
        "Re-segment each host-owned compound semantic group independently. Every returned leaf "
        "must represent exactly one independently observable behavior explicitly present inside "
        "that group's parent source span. source_anchor only needs to ground that behavior; the "
        "host owns exact character partitioning and retains connective/context residue. Do not "
        "merge groups, add capabilities, choose capability IDs, add prerequisites, or invent "
        "gameplay."
    )
    payload = {
        "compound_groups": [
            {
                "group_id": group_id,
                "source_clause_index": int(clause["clause_index"]),
                "parent_source_anchor": str(leaf.get("source_anchor") or ""),
                "parent_semantic_statement": str(leaf.get("semantic_statement") or ""),
                "parent_given": str(leaf.get("given") or ""),
                "parent_when": str(leaf.get("when") or ""),
                "parent_then": str(leaf.get("then") or ""),
                "authored_clause_text": str(clause["text"]),
            }
            for group_id, leaf, clause in items
        ]
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _semantic._canonical(payload)},
    ]


def _ground_parent_span(
    leaf: Mapping[str, Any], clause: Mapping[str, Any]
) -> tuple[int, int]:
    if "source_start" in leaf and "source_end" in leaf:
        return int(leaf["source_start"]), int(leaf["source_end"])
    from . import semantic_requirement_authority as _semantic

    grounding = _semantic._ground_source_anchor(clause, str(leaf.get("source_anchor") or ""))
    if grounding is None:
        raise ValueError("compound parent source anchor is not grounded")
    return int(grounding["source_start"]), int(grounding["source_end"])


def _contains_authored_text(value: str) -> bool:
    return any(
        not character.isspace()
        and not unicodedata.category(character).startswith(("P", "Z"))
        for character in value
    )


def _provenance_residue(
    *,
    clause_index: int,
    clause_text: str,
    clause_start: int,
    start: int,
    end: int,
) -> dict[str, Any] | None:
    text = clause_text[start - clause_start : end - clause_start]
    if not _contains_authored_text(text):
        return None
    action_families = detected_action_families(text)
    if action_families:
        raise ValueError(
            "re-segmentation left an executable-looking source residue unmodeled: "
            + ", ".join(action_families)
        )
    return {
        "source_clause_index": clause_index,
        "source_anchor": text,
        "semantic_statement": "",
        "given": "",
        "when": "",
        "then": "",
        "semantic_type": "gameplay_mechanic",
        "source_quote": text,
        "source_start": start,
        "source_end": end,
        "grounding_method": "host_provenance_residue",
        "grounding_similarity": 1.0,
        "model_anchor": "",
        "non_executable_reason": "unmodeled connective/context provenance retained by host",
    }


def _complete_parent_provenance(
    normalized: Sequence[Mapping[str, Any]],
    *,
    clause: Mapping[str, Any],
    parent_start: int,
    parent_end: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ordered semantic leaves plus deterministic non-executable source residue."""
    clause_index = int(clause["clause_index"])
    clause_start = int(clause["char_start"])
    clause_text = str(clause["text"])
    ordered = sorted(
        (dict(leaf) for leaf in normalized),
        key=lambda leaf: (int(leaf["source_start"]), int(leaf["source_end"])),
    )
    residue: list[dict[str, Any]] = []
    cursor = parent_start
    for leaf in ordered:
        start = int(leaf["source_start"])
        end = int(leaf["source_end"])
        if start < cursor:
            raise ValueError("re-segmentation source anchors overlap or are out of order")
        if start > cursor:
            gap = _provenance_residue(
                clause_index=clause_index,
                clause_text=clause_text,
                clause_start=clause_start,
                start=cursor,
                end=start,
            )
            if gap is not None:
                residue.append(gap)
        cursor = end
    if cursor < parent_end:
        gap = _provenance_residue(
            clause_index=clause_index,
            clause_text=clause_text,
            clause_start=clause_start,
            start=cursor,
            end=parent_end,
        )
        if gap is not None:
            residue.append(gap)
    return ordered, residue


def _normalize_resegmented_group(
    raw_leaves: Any,
    parent_leaf: Mapping[str, Any],
    clause: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from . import semantic_requirement_authority as _semantic
    from .semantic_source_fidelity import validate_semantic_source_partition

    if not isinstance(raw_leaves, list) or not raw_leaves:
        raise ValueError("re-segmentation group must contain leaves")

    clause_index = int(clause["clause_index"])
    parent_start, parent_end = _ground_parent_span(parent_leaf, clause)
    normalized: list[dict[str, Any]] = []
    for raw in raw_leaves:
        if not isinstance(raw, Mapping):
            raise TypeError("re-segmentation leaf must be an object")
        unexpected = set(raw) - _RESEGMENT_FIELDS
        if unexpected:
            raise ValueError(f"re-segmentation leaf overreached authority: {sorted(unexpected)}")
        if raw.get("source_clause_index") != clause_index:
            raise ValueError("re-segmentation leaf changed source clause")
        semantic_statement = str(raw.get("semantic_statement") or "").strip()
        given = str(raw.get("given") or "").strip()
        when = str(raw.get("when") or "").strip()
        then = str(raw.get("then") or "").strip()
        anchor = str(raw.get("source_anchor") or "").strip()
        if not (semantic_statement and given and when and then and anchor):
            raise ValueError("re-segmentation leaf has incomplete semantic fields")
        grounding = _semantic._ground_source_anchor(clause, anchor)
        if grounding is None:
            raise ValueError(f"re-segmentation anchor is not grounded: {anchor!r}")
        start = int(grounding["source_start"])
        end = int(grounding["source_end"])
        if not (parent_start <= start < end <= parent_end):
            raise ValueError("re-segmentation leaf escaped its parent source span")
        semantic_type = str(raw.get("semantic_type") or "gameplay_mechanic").casefold()
        if semantic_type not in {"gameplay_mechanic", "software_quality"}:
            semantic_type = "gameplay_mechanic"
        normalized.append(
            {
                "source_clause_index": clause_index,
                "source_anchor": anchor,
                "semantic_statement": semantic_statement,
                "given": given,
                "when": when,
                "then": then,
                "semantic_type": semantic_type,
                **grounding,
            }
        )

    normalized, provenance_residue = _complete_parent_provenance(
        normalized,
        clause=clause,
        parent_start=parent_start,
        parent_end=parent_end,
    )
    clause_start = int(clause["char_start"])
    parent_text = str(clause["text"])[
        parent_start - clause_start : parent_end - clause_start
    ]
    parent_clause = {
        "clause_index": clause_index,
        "char_start": parent_start,
        "char_end": parent_end,
        "text": parent_text,
    }
    partition_diagnostics = validate_semantic_source_partition(
        [*normalized, *provenance_residue], [parent_clause]
    )
    if partition_diagnostics:
        raise ValueError(
            "host provenance partition invariant failed after re-segmentation: "
            + _semantic._canonical(list(partition_diagnostics))
        )

    atomic: list[dict[str, Any]] = []
    non_executable: list[dict[str, Any]] = list(provenance_residue)
    for leaf in normalized:
        status, reason = validate_leaf_atomicity(leaf, parent_text)
        if status == LeafAtomicityStatus.COMPOUND:
            raise ValueError("re-segmentation remained compound: " + reason)
        if status in {LeafAtomicityStatus.CONTEXT, LeafAtomicityStatus.CATCH_ALL}:
            non_executable.append(leaf)
        else:
            atomic.append(leaf)
    if not atomic:
        raise ValueError("compound re-segmentation produced no executable atomic leaf")
    return atomic, non_executable


def resegment_compound_leaves(
    router: Any,
    compounds: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Re-segment compound leaves once per bounded group; never retry source partitioning."""
    if not compounds:
        return [], [], 0

    from . import semantic_leaf_pipeline as _pipeline

    atomic: list[dict[str, Any]] = []
    non_executable: list[dict[str, Any]] = []
    model_calls = 0

    for chunk_start in range(0, len(compounds), _MAX_RESEGMENT_GROUPS):
        chunk = compounds[chunk_start : chunk_start + _MAX_RESEGMENT_GROUPS]
        items = [
            (f"compound_{chunk_start + index}", leaf, clause)
            for index, (leaf, clause) in enumerate(chunk)
        ]
        group_ids = [group_id for group_id, _, _ in items]
        max_clause_index = max(int(clause["clause_index"]) for _, _, clause in items)
        payload = _pipeline._call_model(
            router,
            operation="resegment_compound_requirements",
            output_tokens=min(4096, 384 + 384 * len(items)),
            messages=_resegment_messages(items),
            parameters=_resegment_schema(group_ids, max_clause_index),
            description=(
                "Re-segment bounded compound semantic groups into source-grounded atomic leaves "
                "without capability or planning authority."
            ),
        )
        model_calls += 1
        if not isinstance(payload, Mapping) or not isinstance(payload.get("groups"), list):
            raise TypeError("compound re-segmentation returned an invalid root object")
        raw_groups = payload["groups"]
        by_id: dict[str, Mapping[str, Any]] = {}
        for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                raise TypeError("compound re-segmentation group must be an object")
            group_id = str(raw_group.get("group_id") or "")
            if group_id not in group_ids or group_id in by_id:
                raise ValueError(f"unknown or repeated compound group id: {group_id!r}")
            by_id[group_id] = raw_group
        if set(by_id) != set(group_ids):
            raise ValueError("compound re-segmentation omitted a host-owned group")

        for group_id, parent_leaf, clause in items:
            group_atomic, group_non_executable = _normalize_resegmented_group(
                by_id[group_id].get("leaves"), parent_leaf, clause
            )
            atomic.extend(group_atomic)
            non_executable.extend(group_non_executable)

    return atomic, non_executable, model_calls


def resegment_compound_leaf(
    router: Any,
    compound_leaf: Mapping[str, Any],
    clause: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper around the bounded batch re-segmentation path."""
    atomic, non_executable, _ = resegment_compound_leaves(
        router, [(compound_leaf, clause)]
    )
    return atomic, non_executable


__all__ = [
    "LeafAtomicityStatus",
    "detected_action_families",
    "filter_and_split_context",
    "is_genre_context",
    "is_pure_catch_all",
    "resegment_compound_leaf",
    "resegment_compound_leaves",
    "validate_leaf_atomicity",
]
