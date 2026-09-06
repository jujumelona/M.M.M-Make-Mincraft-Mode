from __future__ import annotations

from minecraft_mod_ai.versioned_reference_catalog import (
    ReferenceFamily,
    _version_boundary_match,
    reference_families,
)


def test_builtin_reference_catalog_has_distinct_capability_families() -> None:
    families = reference_families()
    repositories = {item.repository for item in families}
    assert "FabricMC/fabric-api" in repositories
    assert "TechReborn/TechReborn" in repositories
    assert "shedaniel/RoughlyEnoughItems" in repositories
    assert "TerraformersMC/ModMenu" in repositories
    assert len(repositories) == len(families)
    assert all(item.capabilities for item in families)


def test_version_boundary_match_rejects_neighbor_versions() -> None:
    assert _version_boundary_match("19.x-1.21.5", "1.21.5")
    assert _version_boundary_match("1.21.5", "1.21.5")
    assert not _version_boundary_match("1.21.50", "1.21.5")
    assert not _version_boundary_match("1.21.6", "1.21.5")


def test_capability_matching_prefers_relevant_family() -> None:
    machine = ReferenceFamily(
        repository="example/machine",
        capabilities=("machine", "automation", "energy", "기계", "자동화"),
        priority=90,
    )
    api = ReferenceFamily(
        repository="example/api",
        capabilities=("events", "registry", "networking"),
        priority=100,
    )
    assert machine.matches_capability("자동화 기계", "기계") > api.matches_capability(
        "자동화 기계", "기계"
    )

    rei = ReferenceFamily(
        repository="example/rei",
        capabilities=("gui", "screen", "inventory", "recipe"),
        priority=90,
    )
    query = "inventory recipe screen"
    assert rei.matches_capability(query, "inventory") > api.matches_capability(
        query, "inventory"
    )
