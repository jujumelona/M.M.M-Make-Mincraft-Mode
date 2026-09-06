from __future__ import annotations

"""Authority-neutral semantic atomicity validation.

The host uses these helpers only to identify obvious context, catch-all text, and visibly
compound model leaves. Compound leaves are conservatively demoted by the host semantic
batcher; this module never calls a model, re-segments text, owns capability selection, or
requires exact character partitioning.
"""

import enum
import re
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
    re.compile(
        r"^(?:등\s*여러\s*가지(?:가\s*가능한\s*모드)?|기타\s*활동|기타\s*기능|등등)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:other\s+activities|and\s+more|etc\.?|and\s+other\s+activities|various\s+others)\s*$",
        re.IGNORECASE,
    ),
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
    """Conservatively identify leaves that visibly bundle independent behaviors."""
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


__all__ = [
    "LeafAtomicityStatus",
    "detected_action_families",
    "filter_and_split_context",
    "is_genre_context",
    "is_pure_catch_all",
    "validate_leaf_atomicity",
]
