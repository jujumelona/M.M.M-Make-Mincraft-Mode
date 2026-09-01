from __future__ import annotations

"""Pipeline hardening for semantic seed search and strict evidence provenance."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline_hardening import _base_evidence_ref, _replace_bound_references

_INSTALLED = False
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9.+]+|[가-힣]{2,}")
_SEED_NOISE = frozenset(
    {
        "minecraft",
        "mod",
        "mods",
        "make",
        "create",
        "system",
        "semantic",
        "implementation",
        "implement",
        "task",
        "feature",
        "mechanic",
        "module",
        "generated",
        "generator",
        "interaction",
        "logic",
        "code",
        "design",
        "plan",
        "planning",
        "research",
        "fabric",
        "forge",
        "neoforge",
        "version",
        "java",
        "with",
        "that",
        "this",
        "the",
        "and",
        "for",
        "please",
    }
)


def _query_tokens(value: Any) -> list[str]:
    text = _CAMEL_BOUNDARY.sub(" ", str(value or ""))
    text = re.sub(r"[_/:\-]+", " ", text)
    return [
        token
        for token in _QUERY_TOKEN.findall(text)
        if len(token.strip()) >= 2
        and token.casefold() not in _SEED_NOISE
    ]


def bounded_seed_query(prompt: str, game_design: Mapping[str, Any]) -> str:
    """Build a short, high-signal ecosystem query instead of serializing the plan."""

    sources: list[Any] = [game_design.get("title", "")]

    modules = game_design.get("modules")
    if isinstance(modules, Sequence) and not isinstance(modules, (str, bytes, bytearray)):
        for item in modules:
            if not isinstance(item, Mapping):
                continue
            sources.extend(
                (
                    item.get("name", ""),
                    item.get("kind", ""),
                    item.get("plugin_id", ""),
                    item.get("reason", ""),
                )
            )

    capabilities = game_design.get("capabilities")
    if isinstance(capabilities, Sequence) and not isinstance(
        capabilities, (str, bytes, bytearray)
    ):
        sources.extend(capabilities)

    sources.extend((game_design.get("pitch", ""), prompt))

    tokens: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for token in _query_tokens(source):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
            if len(tokens) >= 16:
                break
        if len(tokens) >= 16:
            break

    if not tokens:
        fallback = " ".join(str(prompt or "").split()).strip()
        return fallback[:240] or "gameplay"

    query = " ".join(tokens)
    return query[:320].rstrip()


def _strict_provenance_repair(
    note: Mapping[str, Any],
    *,
    allowed_refs: Sequence[str],
) -> dict[str, Any]:
    """Filter provenance without fabricating evidence references."""

    result = dict(note)
    allowed = tuple(
        dict.fromkeys(
            _base_evidence_ref(ref)
            for ref in allowed_refs
            if _base_evidence_ref(ref)
        )
    )
    allowed_set = set(allowed)

    claims: list[Any] = []
    dropped = 0
    for claim in result.get("claims", ()):
        if not isinstance(claim, Mapping):
            dropped += 1
            continue
        item = dict(claim)
        refs = [
            _base_evidence_ref(ref)
            for ref in item.get("evidence_refs", ())
            if _base_evidence_ref(ref) in allowed_set
        ]
        refs = list(dict.fromkeys(refs))
        if not refs:
            dropped += 1
            continue
        item["evidence_refs"] = refs
        claims.append(item)
    if "claims" in result:
        result["claims"] = claims

    procedures: list[Any] = []
    for procedure in result.get("procedures", ()):
        if not isinstance(procedure, Mapping):
            continue
        item = dict(procedure)
        if "evidence_refs" in item:
            refs = [
                _base_evidence_ref(ref)
                for ref in item.get("evidence_refs", ())
                if _base_evidence_ref(ref) in allowed_set
            ]
            refs = list(dict.fromkeys(refs))
            if not refs:
                continue
            item["evidence_refs"] = refs
        procedures.append(item)
    if "procedures" in result:
        result["procedures"] = procedures

    if dropped:
        gaps = [str(value) for value in result.get("gaps", ()) if str(value).strip()]
        gaps.append(
            f"{dropped} synthesized claim(s) were omitted because no host-issued "
            "evidence reference survived provenance validation."
        )
        result["gaps"] = gaps
        if not claims:
            result["sufficient"] = False
    return result


def _install_semantic_seed_search() -> None:
    from . import ecosystem_discovery as ecosystem

    original = ecosystem._seed_query
    if getattr(original, "_mmm_bounded_semantic_seed_query", False):
        return

    def seed_query(prompt: str, game_design: dict[str, Any]) -> str:
        return bounded_seed_query(prompt, game_design)

    seed_query._mmm_bounded_semantic_seed_query = True  # type: ignore[attr-defined]
    ecosystem._seed_query = seed_query
    _replace_bound_references(original, seed_query)


def _install_strict_provenance_filter() -> None:
    from . import pipeline_hardening as v1

    v1._repair_note_provenance = _strict_provenance_repair




def install_pipeline_hardening_v4() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_semantic_seed_search()
    _install_strict_provenance_filter()
    _INSTALLED = True


__all__ = [
    "_strict_provenance_repair",
    "bounded_seed_query",
    "install_pipeline_hardening_v4",
]
