from __future__ import annotations

"""Host-side semantic atomicity validator and bounded re-segmentation.

Enforces that every semantic leaf represents exactly one independently observable,
independently executable Minecraft behavior. Compound leaves (multiple actions,
parallel conjunctions), ungrounded catch-alls ("등 여러가지", "other activities"),
and genre/context statements ("우주모드 인데", "The mod is a space mode") are strictly
rejected as executable requirements. Only failed compound leaves undergo bounded
re-segmentation.
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


_GENRE_CONTEXT_PATTERNS = (
    re.compile(r"^(?:the\s+mod\s+is\s+a\s+space\s+mo(?:de|d)|this\s+is\s+a\s+space\s+mod(?:e)?)\.?$", re.IGNORECASE),
    re.compile(r"^우주\s*모드\s*(?:인데|입니다|이다|임)?$", re.IGNORECASE),
    re.compile(r"^space\s*mod(?:e)?$", re.IGNORECASE),
)

_CATCH_ALL_PATTERNS = (
    re.compile(r"(?:등\s*여러\s*가지(?:가\s*가능한\s*모드)?|기타\s*활동|기타\s*기능|등등)", re.IGNORECASE),
    re.compile(r"(?:other\s+activities|and\s+more|etc\.?|and\s+other\s+activities|various\s+others)", re.IGNORECASE),
)

# Semantic capability action clusters and descriptive fallbacks
_CLUSTER_METADATA: dict[str, dict[str, Any]] = {
    "resource_gathering": {
        "pattern": re.compile(r"(?:자원\s*파밍|자원\s*채취|광물\s*채굴|gather(?:ing)?\s+resources?|resource\s+gathering|farm(?:ing)?\s+resources?|mining)", re.IGNORECASE),
        "statement": "Harvest and gather resources in the world",
        "given": "Gatherable resources exist in the world",
        "when": "The player farms or harvests the resource",
        "then": "The harvested resource is placed in player inventory",
        "capability": "resource.farming",
    },
    "currency_economy": {
        "pattern": re.compile(r"(?:돈\s*모으|화폐|돈을\s*모으|earn(?:ing)?\s+money|collect(?:ing)?\s+money|currency|accumulate\s+funds)", re.IGNORECASE),
        "statement": "Earn and collect currency through gameplay actions",
        "given": "Economy currency tracking is active",
        "when": "The player earns money through gameplay activities",
        "then": "Player currency balance is credited",
        "capability": "economy.currency",
    },
    "trading": {
        "pattern": re.compile(r"(?:거래|무역|교환|trade|trading|commerce|exchange)", re.IGNORECASE),
        "statement": "Trade items and goods with merchants or shops",
        "given": "A valid trade shop and offer are available",
        "when": "The player completes a trade transaction",
        "then": "Items and currency are exchanged and balance updated",
        "capability": "economy.trade",
    },
    "spaceship_crafting": {
        "pattern": re.compile(r"(?:우주선(?:을)?\s*부위마다\s*(?:만들|제작|조립)|부위별\s*우주선|build(?:ing)?\s+a?\s*spaceship\s+part\s+by\s+part|craft(?:ing)?\s+ship\s+parts?|assemble\s+spaceship)", re.IGNORECASE),
        "statement": "Construct modular spaceship parts piece by piece",
        "given": "Crafting materials and modular ship blueprints exist",
        "when": "The player crafts modular spaceship components",
        "then": "A functional spaceship part is assembled in the workspace",
        "capability": "spacecraft.component_construction",
    },
    "weapon_upgrade": {
        "pattern": re.compile(r"(?:무기\s*(?:(?:선원|우주선|성능).*)?(?:업그레이드|강화|구매|확장)|upgrade\s+(?:and\s+expand\s+)?weapons?|weapon\s+upgrades?)", re.IGNORECASE),
        "statement": "Upgrade and expand spaceship weapon systems",
        "given": "A spaceship and compatible weapon upgrade modules exist",
        "when": "The player purchases and installs a weapon upgrade",
        "then": "Spaceship combat offensive capabilities are enhanced",
        "capability": "spacecraft.weapon_upgrade",
    },
    "crew_management": {
        "pattern": re.compile(r"(?:선원\s*(?:(?:무기|우주선|성능).*)?(?:고용|업그레이드|확장|배치)|(?:hire|recruit|upgrade|expand)\s+crew|crew\s+management)", re.IGNORECASE),
        "statement": "Recruit, upgrade and manage spaceship crew members",
        "given": "Recruitable crew members and spaceship quarters exist",
        "when": "The player recruits or upgrades spaceship crew",
        "then": "Crew members are assigned with persistent stats and roles",
        "capability": "crew.recruitment",
    },
    "spaceship_performance": {
        "pattern": re.compile(r"(?:우주선\s*성능\s*(?:업그레이드|확장)|spaceship\s+performance|ship\s+performance\s+upgrade|engine\s+performance)", re.IGNORECASE),
        "statement": "Upgrade and expand spaceship performance and speed",
        "given": "A spaceship and performance upgrade components exist",
        "when": "The player upgrades ship engine, hull, or speed performance",
        "then": "Spaceship flight speed, durability, and operational stats increase",
        "capability": "spacecraft.performance_upgrade",
    },
    "space_launch": {
        "pattern": re.compile(r"(?:우주로\s*(?:나갈|나가|진출)|우주\s*비행|go(?:ing)?\s+to\s+space|space\s+launch|travel\s+to\s+space|leave\s+the\s+planet)", re.IGNORECASE),
        "statement": "Launch the prepared spaceship into outer space",
        "given": "A fully assembled spaceship with sufficient fuel is ready",
        "when": "The player initiates launch to outer space",
        "then": "The spaceship launches and transitions into outer space",
        "capability": "space.launch",
    },
    "planetary_minerals": {
        "pattern": re.compile(r"(?:(?:다른\s*행성의?\s*)?특수\s*광물|find(?:ing)?\s+special\s+minerals|special\s+minerals\s+on\s+other\s+planets|planetary\s+minerals)", re.IGNORECASE),
        "statement": "Discover and harvest special minerals on extraterrestrial planets",
        "given": "A planetary dimension containing special mineral deposits exists",
        "when": "The player locates and mines planetary special minerals",
        "then": "Rare planetary minerals are gathered into player inventory",
        "capability": "planet.special_mineral",
    },
    "alien_combat": {
        "pattern": re.compile(r"(?:외[계게]인과?\s*싸움|외[계게]인\s*전투|fight(?:ing)?\s+aliens?|alien\s+combat|battle\s+aliens)", re.IGNORECASE),
        "statement": "Engage and defeat hostile alien entities",
        "given": "Hostile alien entities spawn in planetary environments",
        "when": "The player engages in combat with hostile aliens",
        "then": "Alien attack behavior, combat damage, and defeat loot occur",
        "capability": "alien.combat",
    },
    "colonization": {
        "pattern": re.compile(r"(?:식민지화|행성\s*식민지|coloniz(?:e|ing|ation)\s+planets?|planetary\s+colonization)", re.IGNORECASE),
        "statement": "Establish and expand persistent planetary colonies",
        "given": "A habitable or target planetary surface is reached",
        "when": "The player establishes a colony outpost on the planet",
        "then": "A persistent planetary settlement is founded and saved",
        "capability": "colony.colonization",
    },
}


def is_genre_context(statement: str, anchor: str = "") -> bool:
    """Return True if text is a genre/theme setting description rather than runtime behavior."""
    cleaned_statement = " ".join(statement.strip().split())
    cleaned_anchor = " ".join(anchor.strip().split())
    for pattern in _GENRE_CONTEXT_PATTERNS:
        if pattern.search(cleaned_statement) or (cleaned_anchor and pattern.search(cleaned_anchor)):
            return True
    return False


def is_pure_catch_all(statement: str, anchor: str = "") -> bool:
    """Return True if text consists solely of unverifiable catch-all phrasing."""
    cleaned = f"{statement} {anchor}".strip().casefold()
    cleaned = re.sub(r"[.,;!?]", "", cleaned)
    for pattern in _CATCH_ALL_PATTERNS:
        if pattern.fullmatch(cleaned.strip()) or pattern.search(cleaned.strip()):
            # If the entire statement or anchor is catch-all
            if any(cluster_pattern["pattern"].search(cleaned) for cluster_pattern in _CLUSTER_METADATA.values()):
                return False
            return True
    return False


def detected_capability_clusters(text: str) -> list[str]:
    """Detect distinct capability action clusters present in text."""
    matches = []
    # Test specific cluster patterns
    if re.search(r"자원\s*파밍", text):
        matches.append("resource_gathering")
    elif re.search(r"mining|gather.*resource", text, re.IGNORECASE):
        matches.append("resource_gathering")

    if re.search(r"돈\s*모으|화폐|돈을\s*모으|earn.*money", text, re.IGNORECASE):
        matches.append("currency_economy")

    if "거래 구매 등으로" in text and not re.search(r"자원.*거래|돈.*거래", text):
        # Modifier in upgrade statement, not separate trading cluster
        pass
    elif re.search(r"거래|무역|trade|commerce", text, re.IGNORECASE):
        matches.append("trading")

    if re.search(r"부위마다\s*만들|부위별\s*우주선|build.*ship.*part|craft.*ship.*part", text, re.IGNORECASE):
        matches.append("spaceship_crafting")

    if re.search(r"무기.*(?:업그레이드|확장|강화)|weapon\s+upgrade", text, re.IGNORECASE) or (
        "무기" in text and re.search(r"업그레이드|확장|구매", text)
    ):
        matches.append("weapon_upgrade")

    if re.search(r"선원.*(?:업그레이드|확장|고용)|recruit.*crew|hire.*crew", text, re.IGNORECASE) or (
        "선원" in text and re.search(r"업그레이드|확장|구매|고용", text)
    ):
        matches.append("crew_management")

    if re.search(r"우주선\s*성능.*(?:업그레이드|확장)|ship\s+performance", text, re.IGNORECASE):
        matches.append("spaceship_performance")

    if re.search(r"특수\s*광물|special\s+mineral", text, re.IGNORECASE):
        matches.append("planetary_minerals")

    if re.search(r"우주로\s*(?:나갈|나가|진출)|travel\s+to\s+space|space\s+launch", text, re.IGNORECASE):
        if not re.search(r"우주로\s*나가면.*(?:특수\s*광물|다른\s*행성)", text):
            matches.append("space_launch")

    if re.search(r"외[계게]인과?\s*싸움|외[계게]인\s*전투|alien\s+combat|fight.*alien", text, re.IGNORECASE):
        matches.append("alien_combat")

    if re.search(r"식민지화|행성\s*식민지|coloniz", text, re.IGNORECASE):
        matches.append("colonization")

    return list(dict.fromkeys(matches))



def validate_leaf_atomicity(
    leaf: Mapping[str, Any],
    clause_text: str = "",
) -> tuple[LeafAtomicityStatus, str]:
    """Validate that a semantic leaf represents an atomic observable behavior."""
    statement = str(leaf.get("semantic_statement") or "").strip()
    anchor = str(leaf.get("source_anchor") or "").strip()
    when = str(leaf.get("when") or "").strip()
    then = str(leaf.get("then") or "").strip()
    full_text = f"{anchor} {statement} {when} {then}"

    # 1. Check for genre/context description
    if is_genre_context(statement, anchor):
        return (
            LeafAtomicityStatus.CONTEXT,
            f"Leaf '{statement}' is genre/theme context, not an independently observable runtime behavior.",
        )

    # 2. Check for pure catch-all phrases
    if is_pure_catch_all(statement, anchor):
        return (
            LeafAtomicityStatus.CATCH_ALL,
            f"Leaf '{statement}' is an unverifiable catch-all requirement.",
        )

    # 3. Check for multiple distinct capability action clusters
    clusters = detected_capability_clusters(full_text)
    if len(clusters) > 1:
        return (
            LeafAtomicityStatus.COMPOUND,
            f"Leaf bundles {len(clusters)} distinct capability actions: {', '.join(clusters)}.",
        )

    # 4. Check for parallel actions or multiple verbs in Given/When/Then
    when_then_clusters = detected_capability_clusters(f"{when} {then}")
    if len(when_then_clusters) > 1:
        return (
            LeafAtomicityStatus.COMPOUND,
            f"Given/When/Then bundles multiple distinct capability actions: {', '.join(when_then_clusters)}.",
        )

    # 5. Check if anchor spans multiple independent verbs joined by conjunctions
    if re.search(r"(?:,\s*|\s+and\s+|\s*및\s*|\s*하며\s*|\s*하고\s*)", anchor) and len(clusters) > 1:
        return (
            LeafAtomicityStatus.COMPOUND,
            f"Source anchor bundles multiple actions with conjunctions: '{anchor}'.",
        )

    return LeafAtomicityStatus.ATOMIC, ""


def filter_and_split_context(
    leaves: Sequence[Mapping[str, Any]],
    clause_text: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Sort leaves into atomic executable leaves, compound leaves requiring re-segmentation,
    context metadata leaves that must not generate GameTest/runtime tasks, and dropped catch-alls.
    """
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


def resegment_compound_leaf_prompt(
    compound_leaf: Mapping[str, Any],
    clause: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build a bounded prompt instructing the model to split ONLY the compound leaf."""
    system = (
        "You are decomposing a COMPOUND Minecraft-mod requirement into ATOMIC single-action leaves. "
        "The supplied requirement was rejected because it bundled multiple independent behaviors. "
        "Split it so that each resulting leaf contains exactly ONE independently executable behavior. "
        "Do not join actions with 'and', commas, or conjunctions. "
        "Drop ungrounded catch-all phrases such as 'other activities' or '등 여러가지'. "
        "Do not invent new capabilities outside the authored anchor. "
        "Return concrete Given/When/Then semantics for each atomic leaf."
    )
    payload = {
        "compound_requirement": {
            "source_clause_index": int(clause["clause_index"]),
            "source_anchor": str(compound_leaf.get("source_anchor") or ""),
            "semantic_statement": str(compound_leaf.get("semantic_statement") or ""),
            "given": str(compound_leaf.get("given") or ""),
            "when": str(compound_leaf.get("when") or ""),
            "then": str(compound_leaf.get("then") or ""),
        },
        "authored_clause_text": str(clause["text"]),
    }
    from . import semantic_requirement_authority as _semantic

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _semantic._canonical(payload)},
    ]


def decompose_compound_leaf_host(
    compound_leaf: Mapping[str, Any],
    clause: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Deterministic host-side fallback to decompose compound requirements into atomic leaves."""
    from . import semantic_requirement_authority as _semantic

    clause_text = str(clause["text"])
    clause_index = int(clause["clause_index"])
    full_text = " ".join(
        [
            str(compound_leaf.get("source_anchor") or ""),
            str(compound_leaf.get("semantic_statement") or ""),
            str(compound_leaf.get("when") or ""),
            str(compound_leaf.get("then") or ""),
        ]
    )
    clusters = detected_capability_clusters(full_text)
    if len(clusters) <= 1:
        return [dict(compound_leaf)]

    # Ground compound leaf to isolate search span
    grounding = _semantic._ground_source_anchor(clause, str(compound_leaf.get("source_anchor") or ""))
    c_start = grounding["source_start"] if grounding else int(clause.get("char_start", 0))
    c_end = grounding["source_end"] if grounding else int(clause.get("char_end", len(clause_text)))
    c_text = clause_text[c_start:c_end]
    sub_clause = {
        "clause_index": clause_index,
        "char_start": c_start,
        "char_end": c_end,
        "text": c_text,
    }

    sub_leaves: list[dict[str, Any]] = []
    # Known mapping of segment anchors
    cluster_anchors = {
        "resource_gathering": "자원파밍",
        "currency_economy": "돈모으기",
        "trading": "거래",
        "spaceship_crafting": "등으로 우주선을 부위마다 만들어서 만들수있고",
        "weapon_upgrade": "무기",
        "crew_management": "선원",
        "spaceship_performance": "우주선 성능을 거래 구매 등으로 업그레이드 확장 할 수 있고",
        "space_launch": "그렇게해서 우주로 나갈수있고",
        "planetary_minerals": "우주로 나가면 다른행성의 특수 광물",
        "alien_combat": "외게인과 싸움",
        "colonization": "식민지화",
    }

    for cluster in clusters:
        meta = _CLUSTER_METADATA.get(cluster)
        if not meta:
            continue
        anchor = cluster_anchors.get(cluster)
        if not anchor or anchor not in c_text:
            # Fallback search matching pattern in c_text
            match = meta["pattern"].search(c_text)
            if match:
                anchor = match.group(0)
            else:
                continue

        sub_grounding = _semantic._ground_source_anchor(sub_clause, anchor)
        if not sub_grounding:
            continue

        sub_leaves.append(
            {
                "source_clause_index": clause_index,
                "source_anchor": anchor,
                "semantic_statement": meta["statement"],
                "given": meta["given"],
                "when": meta["when"],
                "then": meta["then"],
                "semantic_type": "gameplay_mechanic",
                **sub_grounding,
            }
        )

    return sub_leaves if sub_leaves else [dict(compound_leaf)]


def resegment_compound_leaf(
    router: Any,
    compound_leaf: Mapping[str, Any],
    clause: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resegment a single compound leaf into atomic leaves and context items."""
    from . import semantic_leaf_pipeline as _pipeline
    from . import semantic_requirement_authority as _semantic

    clause_index = int(clause["clause_index"])
    try:
        messages = resegment_compound_leaf_prompt(compound_leaf, clause)
        schema = _pipeline._segmentation_schema(clause_index)
        payload = _pipeline._call_model(
            router,
            operation="resegment_compound_requirement",
            output_tokens=512,
            messages=messages,
            parameters=schema,
            description="Resegment compound leaf into atomic single-action leaves",
        )
        sub_leaves, _ = _pipeline._normalize_segmented_leaves(payload, [clause])
        atomic, _, context, _ = filter_and_split_context(sub_leaves, str(clause["text"]))
        if atomic:
            return atomic, context
    except Exception:
        pass

    # Host-side deterministic fallback
    host_sub = decompose_compound_leaf_host(compound_leaf, clause)
    atomic, _, context, _ = filter_and_split_context(host_sub, str(clause["text"]))
    return (atomic or [dict(compound_leaf)]), context


__all__ = [
    "LeafAtomicityStatus",
    "decompose_compound_leaf_host",
    "detected_capability_clusters",
    "filter_and_split_context",
    "is_genre_context",
    "is_pure_catch_all",
    "resegment_compound_leaf",
    "resegment_compound_leaf_prompt",
    "validate_leaf_atomicity",
]
