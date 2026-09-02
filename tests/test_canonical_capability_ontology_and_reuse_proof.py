from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import source_transplant
from minecraft_mod_ai.canonical_capability_ontology import (
    canonical_domain_map,
    resolve_capabilities_from_phrase,
    resolve_capabilities_from_phrase_structured,
)
from minecraft_mod_ai.capability_semantic_inference import (
    enrich_resolution_with_semantic_inference,
)
from minecraft_mod_ai.evidence_first_planning import _DOMAIN_TERM_MAP
from minecraft_mod_ai.reuse_adapters import apply_deterministic_adapters
from minecraft_mod_ai.reuse_planner import decompose_capability_graph
from minecraft_mod_ai.reuse_proof_executor import execute_reuse_proof


def test_unified_ontology_zero_drift() -> None:
    ontology_map = canonical_domain_map()
    assert _DOMAIN_TERM_MAP == ontology_map
    assert "trade" in ontology_map
    assert "boss" in ontology_map
    assert "nuclear" in ontology_map
    assert "medieval" in ontology_map


def test_ontology_can_expand_theme_before_request_catalog_approval() -> None:
    caps = resolve_capabilities_from_phrase("medieval mod")
    assert "trade.shop_registry" in caps
    assert "economy.currency" in caps
    assert "quest.state" in caps
    assert "worldgen.structure" in caps


def test_unresolved_concept_requires_explicit_semantic_evidence() -> None:
    res = resolve_capabilities_from_phrase_structured("medieval banking loan system")
    assert res.unresolved_spans
    unevidenced = enrich_resolution_with_semantic_inference(res)
    assert not any(node.capability_id.startswith("provisional:") for node in unevidenced.nodes)

    enriched = enrich_resolution_with_semantic_inference(
        res,
        router=lambda _prompt: (
            {
                "name": "bank.loan",
                "category": "finance",
                "description": "Bank lending system",
                "dependencies": ("persistence.state_store",),
            },
        ),
    )
    assert any(node.capability_id == "bank.loan" for node in enriched.nodes)
    assert any("persistence.state_store" in edge[1] for edge in enriched.edges)


def test_reuse_decomposition_requires_approved_catalog() -> None:
    with pytest.raises(ValueError, match="approved request catalog or explicit module kinds"):
        decompose_capability_graph("fusion generator system")

    graph = decompose_capability_graph(
        "prompt cannot add capabilities after approval",
        design={
            "_evidence_request_catalog": {
                "requirements": [
                    {
                        "requirement_id": "req-energy",
                        "capability": "energy.generator",
                        "provides": ["capability:energy.generator"],
                        "statement": "fusion generator",
                        "depends_on": [],
                    },
                    {
                        "requirement_id": "req-storage",
                        "capability": "energy.storage",
                        "provides": ["capability:energy.storage"],
                        "statement": "energy storage",
                        "depends_on": ["req-energy"],
                    },
                ]
            }
        },
    )
    assert graph.nodes == ("energy.generator", "energy.storage")
    assert graph.edges == (("energy.storage", "energy.generator"),)


def test_multi_artifact_closure_complete_gating(monkeypatch) -> None:
    fake_blobs = {
        "src/main/java/com/example/boss/BossEntity.java": "blob-java-1",
        "src/main/resources/assets/modid/models/item/boss_sword.json": "blob-res-1",
        "fabric.mod.json": "blob-meta-1",
    }

    def fake_fetch_blob(client, repo, blob_sha):
        if blob_sha == "blob-java-1":
            return b'package com.example.boss;\npublic class BossEntity {\nString id = "modid:boss_sword";\n}'
        if blob_sha == "blob-res-1":
            return b'{"parent": "item/generated"}'
        if blob_sha == "blob-meta-1":
            return b'{"id":"modid","depends":{"fabricloader":">=0.15.0","minecraft":"1.21.1"}}'
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

    donor_slice = source_transplant.inspect_repository_slice(
        repository="example/boss-mod",
        capability="boss.entity",
        adapter=SimpleNamespace(loader="fabric", minecraft_version="1.21.1"),
        discovery_client=SimpleNamespace(github_token=""),
    )
    assert donor_slice is not None
    assert donor_slice.closure_complete is True
    assert len(donor_slice.artifact_nodes) >= 2
    assert donor_slice.target_compatibility == "exact"


def test_deterministic_adapters_pipeline() -> None:
    donor_files = {
        "src/main/java/com/donor/boss/BossSword.java": (
            "package com.donor.boss;\n"
            "import net.fabricmc.fabric.api.item.v1.FabricItemSettings;\n"
            "public class BossSword {\n"
            '    public static final String ID = "donor_mod:boss_sword";\n'
            "    public static final Object SETTINGS = new FabricItemSettings();\n"
            "}\n"
        ),
        "src/main/resources/assets/donor_mod/models/item/boss_sword.json": (
            '{"parent":"item/generated","textures":{"layer0":"donor_mod:item/boss_sword"}}\n'
        ),
    }
    adapted_files, receipts = apply_deterministic_adapters(
        donor_files,
        {
            "target_package": "ai.minecraft.generated.boss",
            "target_modid": "my_target_mod",
            "donor_modid": "donor_mod",
        },
    )
    java_code = adapted_files["src/main/java/com/donor/boss/BossSword.java"]
    assert "package ai.minecraft.generated.boss;" in java_code
    assert "FabricItemSettings" not in java_code
    assert "new Item.Settings()" in java_code
    assert "src/main/resources/assets/my_target_mod/models/item/boss_sword.json" in adapted_files
    assert len(receipts) >= 2


def test_non_authoritative_compile_checker_never_grants_compile_proof(monkeypatch) -> None:
    fake_code = b"package com.donor.mod;\npublic class Item {}"
    fake_sha = "sha256:" + hashlib.sha256(fake_code).hexdigest()
    monkeypatch.setattr(
        source_transplant,
        "_fetch_blob_bytes",
        lambda _client, _repo, _blob_sha: fake_code,
    )
    donor = source_transplant.DonorSlice(
        capability="item.equipment",
        repository="example/mod",
        commit_sha="3" * 40,
        license_id="MIT",
        source_url="https://github.com/example/mod",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Item.java",
                blob_sha="3" * 40,
                sha256=fake_sha,
                size_bytes=len(fake_code),
                symbols=("Item",),
            ),
        ),
        seed_files=("src/main/java/Item.java",),
        source_symbols=("Item",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.90,
        adaptation_cost=0.0,
        closure_complete=True,
    )
    receipt = execute_reuse_proof(
        donor,
        target_workspace="",
        target_context={},
        compile_checker=lambda _files, _context: {
            "compile_passed": True,
            "tests_passed": True,
        },
    )
    assert receipt.compile_passed is False
    assert receipt.tests_passed is False
    assert receipt.authoritative_compile is False
    assert receipt.proof_level == "MATERIALIZED"
    assert receipt.failure_code == "NON_AUTHORITATIVE_COMPILE_CHECKER"
