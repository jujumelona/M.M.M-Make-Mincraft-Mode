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
