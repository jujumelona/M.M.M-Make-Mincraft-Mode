"""Search-only bilingual Minecraft capability aliases.

These aliases expand code-source recall, NOT gameplay scope, trust,
license authority or target compilation evidence. The same vocabulary
must be used when finding repositories and when locating code within
their immutable source trees.
"""
from __future__ import annotations

import re

_SOURCE_INTENT_ALIASES = (
    ("외계", "alien"), ("우주선", "spaceship"), ("우주", "space"),
    ("행성", "planet"), ("식민", "colony"), ("전투", "combat"),
    ("싸움", "combat"), ("공격", "combat"), ("채굴", "mining"),
    ("광물", "ore"), ("광석", "ore"), ("거래", "trading"),
    ("구매", "trading"), ("돈", "economy"), ("선원", "crew"),
    ("제작", "crafting"), ("조립", "assembly"), ("탐험", "exploration"),
    ("이동", "travel"), ("포털", "portal"), ("포탈", "portal"),
    ("차원", "dimension"), ("엔티티", "entity"), ("몹", "mob"),
    ("싸우", "combat"), ("싸움", "combat"), ("전쟁", "combat"),
    ("전투", "combat"), ("교전", "combat"), ("침공", "combat"),
    ("침략", "combat"), ("로켓", "rocket"), ("함선", "spaceship"),
    ("우주선", "spaceship"), ("탐사", "exploration"),
)


def source_intent_aliases(capability: str) -> tuple[str, ...]:
    from .minecraft_knowledge_nodes import FEATURE_TERMS

    phrase = str(capability).casefold()
    values = [alias for korean, alias in _SOURCE_INTENT_ALIASES if korean in phrase]
    for terms in FEATURE_TERMS.values():
        if any(str(term) in phrase for term in terms if re.search(r"[가-힣]", str(term))):
            values.extend(str(term).lower() for term in terms
                          if re.fullmatch(r"[a-z][a-z -]{2,50}", str(term).lower()))
    return tuple(dict.fromkeys(values))


# These are retrieval equivalences, not assertions that projects implement the
# requested mechanic. Canonical labels must not grant donor proof authority.
_SOURCE_CANONICAL_WORDS = {
    "spacecraft": ("space", "ship", "spaceship"),
    "starship": ("space", "ship", "spaceship"),
    "spaceship": ("space", "ship"),
    "spaceships": ("space", "ship", "spaceship"),
    "rocket": ("space", "ship"),
    "battle": ("combat",),
    "battles": ("combat",),
    "fight": ("combat",),
    "fighting": ("combat",),
    "warfare": ("combat",),
    "attack": ("combat",),
    "attacking": ("combat",),
    "invasion": ("combat",),
    "extraterrestrial": ("alien",),
    "aliens": ("alien",),
    "planetary": ("planet",),
    "planets": ("planet",),
    "explore": ("exploration",),
    "exploring": ("exploration",),
    "mines": ("mining",),
    "mine": ("mining",),
    "trade": ("trading",),
    "trades": ("trading",),
}


def source_intent_terms(value: str) -> frozenset[str]:
    """Deterministic Korean/English retrieval vocabulary for source ranking.

    The same normalization is applied to the requested mechanic AND actual
    provider evidence, so bilingual requests cannot be reduced to result order.
    """
    phrase = str(value or "").casefold()
    words = set(re.findall(r"[a-z][a-z0-9_]*|[가-힣]{2,}", phrase))
    words.update(source_intent_aliases(phrase))
    for word in tuple(words):
        words.update(_SOURCE_CANONICAL_WORDS.get(word, ()))
    return frozenset(words)


def source_query_variants(value: str) -> tuple[str, ...]:
    """Find English-language mod projects for Korean requests.

    Reuse the authored intent; never fabricate new gameplay requirements.
    English queries retain their existing search and pagination semantics.
    """
    phrase = " ".join(str(value or "").split())
    if not re.search(r"[가-힣]", phrase):
        return (phrase,) if phrase else ()
    aliases = tuple(dict.fromkeys(source_intent_aliases(phrase)))
    if not aliases:
        return (phrase,)
    variants = [" ".join(aliases[:4])]
    if len(aliases) >= 3:
        variants.extend((f"{aliases[0]} {aliases[-1]}", f"{aliases[1]} {aliases[-1]}"))
    elif len(aliases) == 2:
        variants.extend(aliases)
    else:
        variants.append(phrase)
    return tuple(dict.fromkeys(v for v in variants if v))
