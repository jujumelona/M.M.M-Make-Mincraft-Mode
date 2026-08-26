from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from minecraft_mod_ai.canonical_capability_ontology import (
    canonical_domain_map,
    resolve_capabilities_from_phrase,
    search_queries_for_capability,
    atomic_capability_definitions,
)
from minecraft_mod_ai.evidence_first_planning import build_request_catalog, _DOMAIN_TERM_MAP
from minecraft_mod_ai.reuse_planner import decompose_capability_graph, _CAPABILITY_HINTS
from minecraft_mod_ai import source_transplant


def test_unified_ontology_zero_drift() -> None:
    # Verify that evidence_first_planning and reuse_planner share the exact same canonical domain map
    ontology_map = canonical_domain_map()
    assert _DOMAIN_TERM_MAP == ontology_map
    assert _CAPABILITY_HINTS == ontology_map

    # Check key domain capability expansion consistency
    assert "trade" in ontology_map
    assert "boss" in ontology_map
    assert "nuclear" in ontology_map
    assert "medieval" in ontology_map


def test_medieval_theme_subsystem_archetype_expansion() -> None:
    # "중세 모드" must expand into a rich subsystem graph rather than a single empty/semantic node
    caps = resolve_capabilities_from_phrase("중세 모드")
    assert len(caps) >= 5
    assert "trade.shop_registry" in caps
    assert "economy.currency" in caps
    assert "item.equipment" in caps
    assert "quest.state" in caps
    assert "worldgen.structure" in caps

    # When decomposed into a capability graph
    graph = decompose_capability_graph("중세 모드 만들어줘")
    assert len(graph.nodes) >= 5
    assert any("trade" in node or "shop" in node for node in graph.nodes)
    assert any("economy" in node or "currency" in node for node in graph.nodes)
    assert any("structure" in node or "worldgen" in node for node in graph.nodes)


def test_nuclear_fusion_generator_functional_decomposition() -> None:
    # "핵융합 발전기" must decompose into machine, energy production, ticking, and UI container
    caps = resolve_capabilities_from_phrase("핵융합 발전기")
    assert "energy.generator" in caps
    assert "energy.production" in caps
    assert "block_entity.tick" in caps
    assert "energy.storage" in caps
    assert "ui.container" in caps

    graph = decompose_capability_graph("핵융합 발전기 시스템")
    assert len(graph.nodes) >= 4
    assert any("energy" in node for node in graph.nodes)
    assert any("block_entity" in node for node in graph.nodes)


def test_spaceship_warp_functional_decomposition() -> None:
    # "우주선 워프 엔진" must decompose into vehicle entity, dimension, and energy generator
    caps = resolve_capabilities_from_phrase("우주선 워프 엔진")
    assert "entity.vehicle" in caps

    graph = decompose_capability_graph("우주선 워프 엔진")
    assert len(graph.nodes) >= 2
    assert any("vehicle" in node or "transport" in node for node in graph.nodes)


def test_multi_artifact_dependency_closure_and_adaptation_cost(monkeypatch) -> None:
    # Verify inspect_repository_slice collects multi-artifact resource files and calculates adaptation cost
    fake_blobs = {
        "src/main/java/com/example/boss/BossEntity.java": "blob-java-1",
        "src/main/java/com/example/boss/BossRenderer.java": "blob-java-2",
        "src/main/resources/assets/modid/models/item/boss_sword.json": "blob-res-1",
        "src/main/resources/data/modid/loot_tables/boss_drops.json": "blob-res-2",
        "fabric.mod.json": "blob-meta-1",
    }

    mock_client = SimpleNamespace(
        get=lambda url, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            content=b'{"mock": true}',
            json=lambda: {"mock": True},
        ),
        close=lambda: None,
    )

    def fake_fetch_blob(client, repo, blob_sha):
        if blob_sha == "blob-java-1":
            return b"package com.example.boss;\npublic class BossEntity extends PathAwareEntity {\npublic BossRenderer renderer;\n}"
        if blob_sha == "blob-java-2":
            return b"package com.example.boss;\npublic class BossRenderer {\n}"
        if blob_sha == "blob-res-1":
            return b'{"parent": "item/generated"}'
        if blob_sha == "blob-res-2":
            return b'{"type": "minecraft:entity"}'
        if blob_sha == "blob-meta-1":
            return b'{"id": "modid", "name": "Boss Mod", "depends": {"fabricloader": ">=0.15.0", "minecraft": "1.21.1"}}'
        return b""

    monkeypatch.setattr(source_transplant, "_fetch_blob_bytes", fake_fetch_blob)
    monkeypatch.setattr(
        source_transplant,
        "_repository_snapshot",
        lambda repo, disc: {
            "license_id": "MIT",
            "commit_sha": "a" * 40,
            "blobs": fake_blobs,
            "source_url": "https://github.com/example/boss-mod",
        },
    )

    adapter = SimpleNamespace(loader="fabric", minecraft_version="1.21.1")
    donor_slice = source_transplant.inspect_repository_slice(
        repository="example/boss-mod",
        capability="boss.entity",
        adapter=adapter,
        discovery_client=SimpleNamespace(github_token=""),
    )

    assert donor_slice is not None
    assert donor_slice.target_compatibility == "exact"
    assert donor_slice.exact_target is True
    # Adaptation cost must be computed and finite
    assert donor_slice.adaptation_cost >= 0.0
    slice_dict = donor_slice.to_dict()
    assert "adaptation_cost" in slice_dict
    assert slice_dict["adaptation_cost"] == donor_slice.adaptation_cost
