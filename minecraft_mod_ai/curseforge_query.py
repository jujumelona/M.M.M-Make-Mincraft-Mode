"""Target-aware CurseForge search terms and evidence relevance.

This module never performs source-repository discovery. A catalogue hit is
research metadata, not evidence that Java source can be transplanted.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .source_intent_aliases import source_intent_aliases, source_intent_terms

_GENERIC = frozenset((
    "minecraft", "fabric", "forge", "neoforge", "quilt", "mode", "mod", "mods",
    "game", "games", "system", "feature", "features", "code", "source",
    "java", "plugin", "implementation", "create", "creating", "make",
    "making", "player", "players", "custom", "add", "adding", "and", "or",
    "with", "into", "from", "there", "the", "for", "this", "that",
    "using", "more", "all", "some", "new", "allows", "enable", "enables",
    "gameplay", "worldbuilding", "request", "requests", "long",
    "requirement", "requirements", "story", "narrative", "wanted",
    "should", "would", "could", "needs", "need", "stuff",
    "modular", "sections", "upgrade", "upgrades", "purchase", "purchases",
    "special", "accumulation", "build", "building", "explore", "exploring",
    "fight", "fighting", "launch", "mode", "mechanics", "interaction",
))

# Search is intentionally broad enough for NAME/AUTHOR matching. Descriptions
# are evaluated after results arrive; full feature sentences are not sent to
# searchFilter because the API does not search descriptions.
_COMPOUNDS = (
    (("space", "ship"), "spaceship"),
    (("space", "craft"), "spacecraft"),
    (("alien", "combat"), "alien"),
)


def _words(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z][a-z0-9]{2,}", str(value).casefold()))


def curseforge_search_queries(
    capability: str, *,
    search_terms: Sequence[str] = (),
    max_variants: int = 4,
) -> tuple[str, ...]:
    """Convert a user capability into bounded nonempty name-search variants.

    Capability terms take precedence; graph hints broaden recall only. Unknown
    non-English text without a supported retrieval translation is not silently
    replaced with an unrelated global catalogue browse.
    """
    if max_variants < 1:
        raise ValueError("CurseForge search variant budget must be positive.")
    phrase = " ".join(str(capability or "").split())
    if not phrase:
        return ()
    aliases = list(source_intent_aliases(phrase))
    raw = list(_words(phrase))
    terms = list(dict.fromkeys(
        word for word in (*aliases, *raw)
        if word not in _GENERIC and len(word) >= 3
    ))
    for hint in search_terms:
        for word in (*source_intent_aliases(str(hint)), *_words(str(hint))):
            if word not in _GENERIC and len(word) >= 3 and word not in terms:
                terms.append(word)
    if not terms:
        return ()
    compounds = [
        token for needles, token in _COMPOUNDS if all(x in terms for x in needles)
    ]
    # Provider search matches NAME/AUTHOR, so one subject per request. The
    # title/summary reranker subsequently checks the full original capability.
    queries = list(dict.fromkeys((*compounds, *terms)))
    return tuple(queries[:max_variants])


def curseforge_catalogue_score(
    capability: str, mod: Mapping[str, Any], *,
    search_terms: Sequence[str] = (),
) -> int:
    """Exact token/synonym overlaps weighted by title > summary > category."""
    wanted = set(source_intent_terms(capability))
    for hint in search_terms:
        wanted.update(source_intent_terms(str(hint)))
    wanted.difference_update(_GENERIC)
    if not wanted:
        return 0
    title = set(source_intent_terms(str(mod.get("name") or "")))
    summary = set(source_intent_terms(str(mod.get("summary") or "")))
    raw_categories = mod.get("categories")
    categories = " ".join(
        str(row.get("name") or "") for row in raw_categories
        if isinstance(row, Mapping)
    ) if isinstance(raw_categories, list) else ""
    cat_terms = set(source_intent_terms(categories))
    return (6 * len(wanted & title)
            + 2 * len(wanted & summary)
            + len(wanted & cat_terms))


__all__ = ["curseforge_search_queries", "curseforge_catalogue_score"]
