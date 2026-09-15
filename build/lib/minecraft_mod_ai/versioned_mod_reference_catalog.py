from __future__ import annotations

"""Curated large-mod reference families for exact-target code exemplars.

The catalogue is intentionally small and quality-biased.  It does not claim that a
repository is compatible with a target merely because it appears here: the runtime
resolver must prove the exact Minecraft/loader target from the selected immutable
revision before any source excerpt is admitted.
"""

import re
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ReferenceFamily:
    repository: str
    loaders: frozenset[str]
    baseline_priority: int
    capabilities: tuple[str, ...]
    architecture_terms: tuple[str, ...] = (
        "initializer",
        "registry",
        "registration",
        "datagen",
        "network",
        "packet",
        "config",
        "component",
        "blockentity",
        "block entity",
        "screen",
        "menu",
        "event",
        "resource",
        "recipe",
    )


_REFERENCE_FAMILIES: Final[tuple[ReferenceFamily, ...]] = (
    ReferenceFamily(
        repository="TechReborn/TechReborn",
        loaders=frozenset({"fabric"}),
        baseline_priority=100,
        capabilities=(
            "machine", "automation", "industrial", "energy", "power", "inventory",
            "recipe", "processing", "storage", "fluid", "cable", "block entity",
            "persistence", "network", "기계", "자동화", "산업", "에너지", "인벤토리",
            "레시피", "저장", "유체", "케이블",
        ),
    ),
    ReferenceFamily(
        repository="AztechMC/Modern-Industrialization",
        loaders=frozenset({"fabric"}),
        baseline_priority=96,
        capabilities=(
            "machine", "automation", "industrial", "energy", "inventory", "recipe",
            "processing", "storage", "fluid", "network", "gui", "screen", "datagen",
            "기계", "자동화", "산업", "에너지", "인벤토리", "레시피", "유체", "화면",
        ),
    ),
    ReferenceFamily(
        repository="shedaniel/RoughlyEnoughItems",
        loaders=frozenset({"fabric", "neoforge", "forge"}),
        baseline_priority=88,
        capabilities=(
            "gui", "screen", "menu", "inventory", "recipe", "render", "search",
            "widget", "network", "serialization", "client", "화면", "인벤토리",
            "레시피", "렌더링", "검색", "위젯", "네트워크",
        ),
    ),
    ReferenceFamily(
        repository="TerraformersMC/ModMenu",
        loaders=frozenset({"fabric"}),
        baseline_priority=76,
        capabilities=(
            "gui", "screen", "config", "client", "render", "integration",
            "metadata", "widget", "화면", "설정", "클라이언트", "렌더링", "위젯",
        ),
    ),
    ReferenceFamily(
        repository="FabricMC/fabric-api",
        loaders=frozenset({"fabric"}),
        baseline_priority=72,
        capabilities=(
            "api", "event", "registry", "network", "worldgen", "render", "datagen",
            "block", "item", "entity", "server", "client", "이벤트", "레지스트리",
            "네트워크", "월드젠", "렌더링", "데이터젠", "블록", "아이템", "엔티티",
        ),
    ),
    ReferenceFamily(
        repository="mekanism/Mekanism",
        loaders=frozenset({"forge", "neoforge"}),
        baseline_priority=100,
        capabilities=(
            "machine", "automation", "industrial", "energy", "power", "inventory",
            "recipe", "processing", "storage", "fluid", "chemical", "network",
            "serialization", "gui", "screen", "기계", "자동화", "산업", "에너지",
            "인벤토리", "레시피", "유체", "네트워크", "화면",
        ),
    ),
)


def reference_families(loader: str) -> tuple[ReferenceFamily, ...]:
    normalized = str(loader or "").strip().casefold()
    return tuple(
        sorted(
            (family for family in _REFERENCE_FAMILIES if normalized in family.loaders),
            key=lambda family: (-family.baseline_priority, family.repository.casefold()),
        )
    )


def exact_version_in_text(text: str, minecraft_version: str) -> bool:
    """Match one MC version without confusing 1.21.1 with 1.21.10."""

    version = re.escape(str(minecraft_version or "").strip())
    if not version:
        return False
    return re.search(rf"(?<![0-9.]){version}(?![0-9.])", str(text or "")) is not None


def capability_score(family: ReferenceFamily, query: str) -> float:
    haystack = str(query or "").casefold()
    if not haystack:
        return 0.0
    hits = 0
    weighted = 0
    for capability in family.capabilities:
        needle = capability.casefold()
        if needle and needle in haystack:
            hits += 1
            weighted += max(1, min(4, len(needle.split())))
    if not hits:
        return 0.0
    return min(1.0, 0.22 * hits + 0.08 * weighted)


def baseline_families(loader: str, *, limit: int) -> tuple[ReferenceFamily, ...]:
    return reference_families(loader)[: max(0, int(limit))]


def task_families(
    loader: str,
    query: str,
    *,
    exclude: frozenset[str] = frozenset(),
    limit: int,
) -> tuple[ReferenceFamily, ...]:
    ranked = [
        (capability_score(family, query), family)
        for family in reference_families(loader)
        if family.repository not in exclude
    ]
    ranked = [item for item in ranked if item[0] > 0.0]
    ranked.sort(key=lambda item: (-item[0], -item[1].baseline_priority, item[1].repository))
    return tuple(family for _score, family in ranked[: max(0, int(limit))])


__all__ = [
    "ReferenceFamily",
    "baseline_families",
    "capability_score",
    "exact_version_in_text",
    "reference_families",
    "task_families",
]
