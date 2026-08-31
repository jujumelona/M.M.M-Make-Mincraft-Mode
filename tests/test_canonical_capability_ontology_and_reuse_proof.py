from __future__ import annotations

from types import SimpleNamespace

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
from minecraft_mod_ai.reuse_planner import _CAPABILITY_HINTS, decompose_capability_graph
from minecraft_mod_ai.reuse_proof_executor import (
    execute_candidate_fallback_loop,
    execute_reuse_proof,
)


def test_unified_ontology_zero_drift() -> None:
    ontology_map = canonical_domain_map()
    assert _DOMAIN_TERM_MAP == ontology_map
    assert _CAPABILITY_HINTS == ontology_map

    assert "trade" in ontology_map
    assert "boss" in ontology_map
    assert "nuclear" in ontology_map
    assert "medieval" in ontology_map


def test_medieval_theme_subsystem_archetype_expansion() -> None:
    caps = resolve_capabilities_from_phrase("medieval mod")
    assert len(caps) >= 5
    assert "trade.shop_registry" in caps
    assert "economy.currency" in caps
    assert "item.equipment" in caps
    assert "quest.state" in caps
    assert "worldgen.structure" in caps

    graph = decompose_capability_graph("create medieval mod")
    assert len(graph.nodes) >= 5
    assert any("trade" in node or "shop" in node for node in graph.nodes)
    assert any("economy" in node or "currency" in node for node in graph.nodes)
    assert any("structure" in node or "worldgen" in node for node in graph.nodes)


def test_medieval_banking_unresolved_concept_preservation() -> None:
    res = resolve_capabilities_from_phrase_structured("medieval banking loan system")
    origins = {n.capability_id: n.origin for n in res.nodes}

    assert any(origins.get(k) == "archetype_inferred" for k in origins)
    assert any(origins.get(k) == "unresolved_concept" for k in origins)
    assert len(res.unresolved_spans) >= 1
    assert "banking loan" in res.unresolved_spans[0]

    # Unknown concepts are not promoted without semantic evidence.
    unevidenced = enrich_resolution_with_semantic_inference(res)
    assert not any(node.capability_id.startswith("provisional:") for node in unevidenced.nodes)

    # An explicit semantic router may promote the unresolved concept and declare dependencies.
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


def test_nuclear_fusion_generator_functional_decomposition() -> None:
    caps = resolve_capabilities_from_phrase("fusion generator")
    assert "energy.generator" in caps
    assert "energy.production" in caps
    assert "block_entity.tick" in caps
    assert "energy.storage" in caps
    assert "ui.container" in caps

    graph = decompose_capability_graph("fusion generator system")
    assert len(graph.nodes) >= 4
    assert any("energy" in node for node in graph.nodes)
    assert any("block_entity" in node for node in graph.nodes)
    # Directed requires-edges must exist between generator and storage/tick
    assert len(graph.edges) >= 1


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
            '{"parent": "item/generated", "textures": {"layer0": "donor_mod:item/boss_sword"}}\n'
        ),
    }

    target_context = {
        "target_package": "ai.minecraft.generated.boss",
        "target_modid": "my_target_mod",
        "donor_modid": "donor_mod",
    }

    adapted_files, receipts = apply_deterministic_adapters(donor_files, target_context)

    # 1. Check package relocation
    java_code = adapted_files["src/main/java/com/donor/boss/BossSword.java"]
    assert "package ai.minecraft.generated.boss;" in java_code
    assert "FabricItemSettings" not in java_code
    assert "new Item.Settings()" in java_code

    # 2. Check modid rewriting and folder relocation
    assert "src/main/resources/assets/my_target_mod/models/item/boss_sword.json" in adapted_files
    json_model = adapted_files["src/main/resources/assets/my_target_mod/models/item/boss_sword.json"]
    assert "my_target_mod:item/boss_sword" in json_model
    assert "donor_mod" not in json_model

    assert len(receipts) >= 2


def test_reuse_proof_executor_fallback_loop(monkeypatch) -> None:
    import hashlib

    broken_code = b"package test; public class Broken {}"
    clean_code = b"package test; public class Clean {}"
    broken_blob = "1" * 40
    clean_blob = "2" * 40

    def fake_fetch(_client, _repo, blob_sha):
        return broken_code if blob_sha == broken_blob else clean_code

    monkeypatch.setattr(source_transplant, "_fetch_blob_bytes", fake_fetch)

    donor_a = source_transplant.DonorSlice(
        capability="combat.damage",
        repository="example/broken-mod",
        commit_sha="1" * 40,
        license_id="MIT",
        source_url="https://github.com/example/broken-mod",
        target_compatibility="adapt",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Broken.java",
                blob_sha=broken_blob,
                sha256="sha256:" + hashlib.sha256(broken_code).hexdigest(),
                size_bytes=len(broken_code),
                symbols=("Broken",),
            ),
        ),
        seed_files=("src/main/java/Broken.java",),
        source_symbols=("Broken",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.95,
        adaptation_cost=45.0,
        closure_complete=True,
    )
    donor_b = source_transplant.DonorSlice(
        capability="combat.damage",
        repository="example/clean-mod",
        commit_sha="2" * 40,
        license_id="MIT",
        source_url="https://github.com/example/clean-mod",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Clean.java",
                blob_sha=clean_blob,
                sha256="sha256:" + hashlib.sha256(clean_code).hexdigest(),
                size_bytes=len(clean_code),
                symbols=("Clean",),
            ),
        ),
        seed_files=("src/main/java/Clean.java",),
        source_symbols=("Clean",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.85,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    def diagnostic_checker(files, _context):
        if "src/main/java/Broken.java" in files:
            return {"compile_passed": False, "unresolved_symbols": ["MissingDep"]}
        return {"compile_passed": True}

    selected_donor, receipts = execute_candidate_fallback_loop(
        candidates=[donor_a, donor_b],
        capability="combat.damage",
        target_workspace="",
        target_context={"target_package": "ai.test"},
        compile_checker=diagnostic_checker,
    )

    assert selected_donor is None
    assert len(receipts) == 2
    assert all(receipt.compile_passed is False for receipt in receipts)
    assert all(receipt.authoritative_compile is False for receipt in receipts)
    assert all(receipt.proof_level == "MATERIALIZED" for receipt in receipts)
    assert all(
        receipt.failure_code == "NON_AUTHORITATIVE_COMPILE_CHECKER"
        for receipt in receipts
    )


def test_unpunctuated_multiroot_semantic_prompt_regression() -> None:
    # Verbatim exact user failure prompt without punctuation or explicit arrow separators
    prompt = "Create an RPG mod with common mobs bosses equipment leveling and item upgrades"
    capabilities = (
        "mob.spawning",
        "boss.entity",
        "item.equipment",
        "progression.level",
        "item.upgrade",
    )

    # 1. The semantic-model boundary, not the structural parser, owns meaning in an
    # unpunctuated clause. Supply explicit semantic evidence for the complete clause.
    from minecraft_mod_ai.evidence_first_planning import build_request_catalog
    semantic_router = SimpleNamespace(
        generate_text=lambda *_args, **_kwargs: (
            '{"intent":"' + prompt + '",'
            '"gameplay_capability_candidates":['
            '"mob.spawning","boss.entity","item.equipment",'
            '"progression.level","item.upgrade"],'
            '"unresolved":false}'
        )
    )
    catalog = build_request_catalog(prompt, {}, router=semantic_router)
    req_caps = {req["capability"] for req in catalog["requirements"]}

    assert "mob.spawning" in req_caps
    assert "boss.entity" in req_caps
    assert "item.equipment" in req_caps or "item.weapon" in req_caps
    assert "progression.level" in req_caps
    assert "item.upgrade" in req_caps

    # 2. Reuse decomposition receives equivalent explicit semantic evidence.
    graph = decompose_capability_graph(
        prompt,
        semantic_router=lambda _prompt: tuple({"name": cap} for cap in capabilities),
    )
    assert len(graph.nodes) >= 5
    assert any("mob" in n for n in graph.nodes)
    assert any("boss" in n for n in graph.nodes)
    assert any("item" in n or "equipment" in n for n in graph.nodes)
    assert any("level" in n or "progression" in n for n in graph.nodes)
    assert any("upgrade" in n for n in graph.nodes)


def test_package_prefix_relocation_preserves_subpackages() -> None:
    donor_files = {
        "src/main/java/com/donor/mod/client/BossRenderer.java": (
            "package com.donor.mod.client;\n"
            "import com.donor.mod.common.BossEntity;\n"
            "public class BossRenderer {}\n"
        ),
        "src/main/java/com/donor/mod/common/BossEntity.java": (
            "package com.donor.mod.common;\n"
            "public class BossEntity {}\n"
        ),
    }

    target_context = {
        "target_package": "ai.minecraft.generated.maple",
        "target_modid": "maple",
    }

    adapted, _ = apply_deterministic_adapters(donor_files, target_context)

    client_code = adapted["src/main/java/com/donor/mod/client/BossRenderer.java"]
    common_code = adapted["src/main/java/com/donor/mod/common/BossEntity.java"]

    assert "package ai.minecraft.generated.maple.client;" in client_code
    assert "package ai.minecraft.generated.maple.common;" in common_code
    assert "import ai.minecraft.generated.maple.common.BossEntity;" in client_code


def test_strict_materialized_proof_level_without_compile(monkeypatch) -> None:
    import hashlib

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


def test_real_blob_byte_materialization_and_sandbox_isolation(monkeypatch, tmp_path) -> None:
    fake_code = b"package com.donor.mod;\npublic class RealItem {\n    public static void test() {}\n}"
    import hashlib
    fake_sha = "sha256:" + hashlib.sha256(fake_code).hexdigest()

    def mock_fetch(client, repo, blob_sha):
        if blob_sha == "blob-real-1":
            return fake_code
        return b""

    monkeypatch.setattr(source_transplant, "_fetch_blob_bytes", mock_fetch)

    donor = source_transplant.DonorSlice(
        capability="item.equipment",
        repository="example/real-mod",
        commit_sha="4444444444444444444444444444444444444444",
        license_id="MIT",
        source_url="https://github.com/example/real-mod",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/com/donor/mod/RealItem.java",
                blob_sha="blob-real-1",
                sha256=fake_sha,
                size_bytes=len(fake_code),
                symbols=("RealItem",),
            ),
        ),
        seed_files=("src/main/java/com/donor/mod/RealItem.java",),
        source_symbols=("RealItem",),
        required_dependencies=(),
        donor_tests=("RealItemRegistryTest", "RealItemUsageTest"),
        confidence=0.95,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    evaluated_files = {}

    def mock_compile(files, context):
        evaluated_files.update(files)
        return {
            "compile_passed": True,
            "tests_passed": True,
            "tests_executed": 2,
            "tests_passed_count": 2,
            "executed_test_ids": ["RealItemRegistryTest", "RealItemUsageTest"],
        }

    from minecraft_mod_ai.reuse_proof_executor import execute_reuse_proof
    receipt = execute_reuse_proof(
        donor,
        target_workspace=tmp_path,
        target_context={"target_package": "ai.target.mod"},
        compile_checker=mock_compile,
    )

    # 1. Real bytes were evaluated, not placeholders
    assert "src/main/java/com/donor/mod/RealItem.java" in evaluated_files
    assert "public class RealItem" in evaluated_files["src/main/java/com/donor/mod/RealItem.java"]

    # 2. Both compile and tests passed -> BEHAVIOR_VERIFIED
    assert receipt.compile_passed is True
    assert receipt.tests_passed is True
    assert receipt.proof_level == "BEHAVIOR_VERIFIED"

    # 3. Caller workspace remained unpolluted by ephemeral compilation
    assert not (tmp_path / "src/main/java/com/donor/mod/RealItem.java").exists()


def test_reuse_ledger_status_strictly_maps_proof_level() -> None:
    from minecraft_mod_ai.platform_catalog import adapter_for_target
    from minecraft_mod_ai.reuse_planner import ReuseDecision, TargetImplementationPlan

    adapter = adapter_for_target("1.21.1", "fabric")

    d_verified = ReuseDecision(
        capability="boss.entity",
        mode="source_transplant",
        confidence=0.9,
        fresh_implementation_cost=20.0,
        fresh_verification_cost=8.0,
        proof_level="COMPILE_VERIFIED",
    )
    d_mat = ReuseDecision(
        capability="item.upgrade",
        mode="adapt",
        confidence=0.8,
        fresh_implementation_cost=15.0,
        fresh_verification_cost=6.0,
        proof_level="MATERIALIZED",
    )
    d_fresh = ReuseDecision(
        capability="magic.spell",
        mode="fresh",
        confidence=1.0,
        fresh_implementation_cost=25.0,
        fresh_verification_cost=10.0,
        proof_level="DISCOVERED",
    )

    plan = TargetImplementationPlan(
        adapter=adapter,
        capabilities=(d_verified, d_mat, d_fresh),
        platform_evidence=None,
        cross_component_integration_cost=0.0,
        platform_verification_cost=1.0,
        maintenance_risk=0.0,
        total_expected_cost=50.0,
        weighted_verified_reuse=15.0,
        fresh_work=25.0,
        adaptation_work=5.0,
        verification_work=10.0,
        uncertainty=0.0,
        reusable_registry_candidates=0,
    )

    ledger = {item["capability"]: item["status"] for item in plan.to_dict()["reuse_ledger"]}
    scope = {item["capability"]: item["fresh_generation_scope"] for item in plan.to_dict()["reuse_ledger"]}

    assert ledger["boss.entity"] == "VERIFIED_REUSE"
    assert scope["boss.entity"] == "forbidden"

    assert ledger["item.upgrade"] == "MATERIALIZED"
    assert scope["item.upgrade"] == "full"  # Unverified donor must not block fresh implementation!

    assert ledger["magic.spell"] == "FRESH_REQUIRED"
    assert scope["magic.spell"] == "full"

    assert plan.unresolved_capabilities == 2  # boss.entity is verified, item.upgrade and magic.spell are unresolved!


def test_hard_fallback_to_fresh_when_all_candidates_fail_compile() -> None:
    donor_fail_1 = source_transplant.DonorSlice(
        capability="combat.damage",
        repository="example/fail1",
        commit_sha="5555555555555555555555555555555555555555",
        license_id="MIT",
        source_url="https://github.com/example/fail1",
        target_compatibility="metadata_exact",
        files=(),
        seed_files=(),
        source_symbols=(),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )
    donor_fail_2 = source_transplant.DonorSlice(
        capability="combat.damage",
        repository="example/fail2",
        commit_sha="6666666666666666666666666666666666666666",
        license_id="MIT",
        source_url="https://github.com/example/fail2",
        target_compatibility="metadata_exact",
        files=(),
        seed_files=(),
        source_symbols=(),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.8,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    def always_fail(files, context):
        return {"compile_passed": False}

    selected_donor, receipts = execute_candidate_fallback_loop(
        candidates=[donor_fail_1, donor_fail_2],
        capability="combat.damage",
        target_workspace="/tmp/sandbox",
        target_context={},
        compile_checker=always_fail,
    )

    # When all candidates fail, best_donor MUST be None!
    assert selected_donor is None
    assert len(receipts) == 2
    assert receipts[0].compile_passed is False
    assert receipts[1].compile_passed is False


def test_materialize_pinned_donor_rejects_hash_mismatch(monkeypatch) -> None:
    fake_code = b"corrupted content"
    expected_sha = "sha256:7777777777777777777777777777777777777777777777777777777777777777"

    def mock_fetch(client, repo, blob_sha):
        return fake_code

    monkeypatch.setattr(source_transplant, "_fetch_blob_bytes", mock_fetch)

    donor = source_transplant.DonorSlice(
        capability="item.equipment",
        repository="example/mismatch",
        commit_sha="7777777777777777777777777777777777777777",
        license_id="MIT",
        source_url="https://github.com/example/mismatch",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Item.java",
                blob_sha="7777777777777777777777777777777777777777",
                sha256=expected_sha,
                size_bytes=len(fake_code),
                symbols=("Item",),
            ),
        ),
        seed_files=("src/main/java/Item.java",),
        source_symbols=("Item",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.95,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    # Must raise SourceTransplantError on hash mismatch without using corrupted bytes
    import pytest
    with pytest.raises(source_transplant.SourceTransplantError):
        source_transplant.materialize_pinned_donor(donor)


def test_host_verified_library_state() -> None:
    from minecraft_mod_ai.reuse_planner import ReuseDecision
    d_lib = ReuseDecision(
        capability="block.basic",
        mode="library",
        confidence=0.95,
        fresh_implementation_cost=10.0,
        fresh_verification_cost=4.0,
        proof_level="HOST_VERIFIED",
    )

    assert d_lib.verified_reuse is True
    assert d_lib.proof_level == "HOST_VERIFIED"


def test_dependency_adaptation_plan_injection() -> None:
    from minecraft_mod_ai.reuse_adapters import DependencyAdaptationPlan

    sample_bg = """plugins {
    id 'fabric-loom' version '1.7-SNAPSHOT'
}

repositories {
    mavenCentral()
}

dependencies {
    minecraft 'com.mojang:minecraft:1.21.1'
}
"""
    updated_bg, applied = DependencyAdaptationPlan.inject_dependencies_into_build_gradle(
        sample_bg,
        required_dependencies=["cloth-config", "geckolib"],
    )

    assert applied is True
    assert "https://maven.shedaniel.me/" in updated_bg
    assert "me.shedaniel.cloth:cloth-config-fabric" in updated_bg
    assert "https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/" in updated_bg
    assert "software.bernie.geckolib:geckolib-fabric" in updated_bg


def test_residual_symbol_analyzer_extracts_symbols_honestly() -> None:
    from minecraft_mod_ai.reuse_adapters import ResidualSymbolAnalyzer

    residuals = ResidualSymbolAnalyzer.analyze_unresolved_symbols(
        ["CustomBossEntity", "SpecialWeaponItem", "CustomBossEntity"]
    )

    assert residuals == ("CustomBossEntity", "SpecialWeaponItem")


def test_closure_incomplete_capped_at_partial_reuse(monkeypatch) -> None:
    import hashlib

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
        closure_complete=False,
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

    assert receipt.proof_level == "MATERIALIZED"
    assert receipt.compile_passed is False
    assert receipt.authoritative_compile is False
    assert receipt.failure_code == "NON_AUTHORITATIVE_COMPILE_CHECKER"


def test_loader_aware_scaffold_neoforge_and_wrapper(tmp_path) -> None:
    import pytest

    from minecraft_mod_ai.reuse_proof_executor import (
        scaffold_minimal_ephemeral_workspace,
    )

    with pytest.raises(ValueError, match="No executable platform provider"):
        scaffold_minimal_ephemeral_workspace(
            tmp_path,
            target_context={
                "loader": "neoforge",
                "minecraft_version": "1.21.1",
                "target_modid": "maple_mod",
                "java_version": "21",
            },
        )

    assert not (tmp_path / "build.gradle").exists()


def test_kotlin_dsl_dependency_injection() -> None:
    from minecraft_mod_ai.reuse_adapters import DependencyAdaptationPlan

    sample_kts = """plugins {
    kotlin("jvm") version "2.0.0"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.example:lib:1.0")
}
"""
    updated_kts, applied = DependencyAdaptationPlan.inject_dependencies_into_build_gradle(
        sample_kts,
        required_dependencies=["cloth-config", "geckolib"],
        loader="fabric",
        is_kotlin_dsl=True,
    )

    assert applied is True
    assert 'maven("https://maven.shedaniel.me/")' in updated_kts
    assert 'modImplementation("me.shedaniel.cloth:cloth-config-fabric:15.0.127")' in updated_kts


def test_behavior_verified_requires_nonzero_tests_executed() -> None:
    fake_code = b"package com.donor.mod;\npublic class Item {}"
    import hashlib
    fake_sha = "sha256:" + hashlib.sha256(fake_code).hexdigest()

    donor = source_transplant.DonorSlice(
        capability="item.equipment",
        repository="example/mod",
        commit_sha="4444444444444444444444444444444444444444",
        license_id="MIT",
        source_url="https://github.com/example/mod",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Item.java",
                blob_sha="4444444444444444444444444444444444444444",
                sha256=fake_sha,
                size_bytes=len(fake_code),
                symbols=("Item",),
            ),
        ),
        seed_files=("src/main/java/Item.java",),
        source_symbols=("Item",),
        required_dependencies=(),
        donor_tests=("ItemRegistryTest", "ItemUsageTest"),
        confidence=0.95,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    # 1. 0 executed tests: Must NOT be promoted to BEHAVIOR_VERIFIED
    def mock_no_tests(files, context):
        return {"compile_passed": True, "tests_passed": False, "tests_executed": 0, "tests_passed_count": 0}

    receipt_zero = execute_reuse_proof(donor, target_workspace="", target_context={}, compile_checker=mock_no_tests)
    assert receipt_zero.proof_level == "COMPILE_VERIFIED"
    assert receipt_zero.tests_passed is False

    # 2. >= 1 executed test and passed: Promotes to BEHAVIOR_VERIFIED
    def mock_has_tests(files, context):
        return {
            "compile_passed": True,
            "tests_passed": True,
            "tests_executed": 2,
            "tests_passed_count": 2,
            "executed_test_ids": ["ItemRegistryTest", "ItemUsageTest"],
        }

    receipt_passed = execute_reuse_proof(donor, target_workspace="", target_context={}, compile_checker=mock_has_tests)
    assert receipt_passed.proof_level == "BEHAVIOR_VERIFIED"
    assert receipt_passed.tests_passed is True
    assert receipt_passed.tests_executed == 2


def test_artifact_level_partial_reuse_slicing(monkeypatch, tmp_path) -> None:
    import hashlib

    code_a = b"package com.donor.mod;\npublic class CleanClass {}"
    code_b = b"package com.donor.mod;\npublic class BrokenClass { MissingSymbol field; }"
    blob_a = "a" * 40
    blob_b = "b" * 40

    monkeypatch.setattr(
        source_transplant,
        "_fetch_blob_bytes",
        lambda _client, _repo, sha: code_a if sha == blob_a else code_b,
    )

    donor = source_transplant.DonorSlice(
        capability="combat.damage",
        repository="example/slice",
        commit_sha="5" * 40,
        license_id="MIT",
        source_url="https://github.com/example/slice",
        target_compatibility="adapt",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/CleanClass.java",
                blob_sha=blob_a,
                sha256="sha256:" + hashlib.sha256(code_a).hexdigest(),
                size_bytes=len(code_a),
                symbols=("CleanClass",),
            ),
            source_transplant.DonorFile(
                path="src/main/java/BrokenClass.java",
                blob_sha=blob_b,
                sha256="sha256:" + hashlib.sha256(code_b).hexdigest(),
                size_bytes=len(code_b),
                symbols=("BrokenClass",),
            ),
        ),
        seed_files=("src/main/java/CleanClass.java", "src/main/java/BrokenClass.java"),
        source_symbols=("CleanClass", "BrokenClass"),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.85,
        adaptation_cost=20.0,
        closure_complete=True,
    )

    def diagnostic_partial(files, _context):
        if len(files) == 1 and "src/main/java/CleanClass.java" in files:
            return {"compile_passed": True, "tests_passed": False}
        return {"compile_passed": False, "unresolved_symbols": ["MissingSymbol"]}

    existing = tmp_path / "src/main/java/BrokenClass.java"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        "package target; final class BrokenClass {}\n",
        encoding="utf-8",
    )
    existing_sha256 = "sha256:" + hashlib.sha256(existing.read_bytes()).hexdigest()

    receipt = execute_reuse_proof(
        donor,
        target_workspace=tmp_path,
        target_context={},
        compile_checker=diagnostic_partial,
    )

    assert receipt.proof_level == "MATERIALIZED"
    assert receipt.verified_artifacts == ()
    assert "src/main/java/CleanClass.java" in receipt.residual_artifacts
    assert "src/main/java/BrokenClass.java" in receipt.residual_artifacts
    assert receipt.verified_symbols == ()
    assert "MissingSymbol" in receipt.residual_symbols
    assert receipt.failure_code == "NON_AUTHORITATIVE_COMPILE_CHECKER"
    assert receipt.contract.allowed_write_paths == ("src/main/java/BrokenClass.java",)
    assert receipt.contract.expected_old_sha256 == {
        "src/main/java/BrokenClass.java": existing_sha256,
    }
    assert "src/main/java/CleanClass.java" in receipt.contract.required_new_artifacts
    assert "src/main/java/BrokenClass.java" not in receipt.contract.required_new_artifacts


def test_two_stage_compile_and_test_separation(monkeypatch) -> None:
    import hashlib

    code = b"package com.donor.mod;\npublic class BossEntity {}"
    blob = "6" * 40
    monkeypatch.setattr(
        source_transplant,
        "_fetch_blob_bytes",
        lambda _client, _repo, _sha: code,
    )
    donor = source_transplant.DonorSlice(
        capability="boss.entity",
        repository="example/boss",
        commit_sha="6" * 40,
        license_id="MIT",
        source_url="https://github.com/example/boss",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/BossEntity.java",
                blob_sha=blob,
                sha256="sha256:" + hashlib.sha256(code).hexdigest(),
                size_bytes=len(code),
                symbols=("BossEntity",),
            ),
        ),
        seed_files=("src/main/java/BossEntity.java",),
        source_symbols=("BossEntity",),
        required_dependencies=(),
        donor_tests=("BossEntityTest",),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    receipt = execute_reuse_proof(
        donor,
        target_workspace="",
        target_context={},
        compile_checker=lambda _files, _context: {
            "compile_passed": True,
            "tests_passed": False,
            "tests_executed": 1,
            "tests_passed_count": 0,
            "executed_test_ids": ["BossEntityTest"],
        },
    )

    assert receipt.compile_passed is False
    assert receipt.tests_passed is False
    assert receipt.proof_level == "MATERIALIZED"
    assert receipt.failure_code == "NON_AUTHORITATIVE_COMPILE_CHECKER"


def test_capability_acceptance_test_matching() -> None:
    donor = source_transplant.DonorSlice(
        capability="boss.entity",
        repository="example/boss",
        commit_sha="6666666666666666666666666666666666666666",
        license_id="MIT",
        source_url="https://github.com/example/boss",
        target_compatibility="metadata_exact",
        files=(),
        seed_files=(),
        source_symbols=("BossEntity",),
        required_dependencies=(),
        donor_tests=("BossSpawnTest", "BossHealthStateTest", "BossPhaseAITest", "BossLootDeathTest"),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    # 1. Unrelated test passed: MathUtilTest should NOT yield BEHAVIOR_VERIFIED
    def mock_unrelated_test(files, context):
        return {
            "compile_passed": True,
            "tests_passed": True,
            "tests_executed": 1,
            "tests_passed_count": 1,
            "executed_test_ids": ["MathUtilTest"],
        }

    receipt_unrelated = execute_reuse_proof(donor, target_workspace="", target_context={}, compile_checker=mock_unrelated_test)
    assert receipt_unrelated.proof_level == "COMPILE_VERIFIED"
    assert len(receipt_unrelated.matched_capability_tests) == 0

    # 2. Matching capability contracts passed: All 4 REQ-BOSS contracts satisfied -> BEHAVIOR_VERIFIED
    def mock_matching_test(files, context):
        return {
            "compile_passed": True,
            "tests_passed": True,
            "tests_executed": 4,
            "tests_passed_count": 4,
            "executed_test_ids": ["BossSpawnTest", "BossHealthStateTest", "BossPhaseAITest", "BossLootDeathTest"],
        }

    receipt_matching = execute_reuse_proof(donor, target_workspace="", target_context={}, compile_checker=mock_matching_test)
    assert receipt_matching.proof_level == "BEHAVIOR_VERIFIED"
    assert len(receipt_matching.matched_capability_tests) == 4


def test_dynamic_minecraft_version_dependency_injection() -> None:
    from minecraft_mod_ai.reuse_adapters import DependencyAdaptationPlan

    sample_bg = """repositories {
}
dependencies {
}
"""
    updated_bg, applied = DependencyAdaptationPlan.inject_dependencies_into_build_gradle(
        sample_bg,
        required_dependencies=["geckolib"],
        loader="fabric",
        minecraft_version="1.20.1",
    )

    assert applied is True
    assert "https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/" in updated_bg


def test_donor_without_tests_capped_at_compile_verified(monkeypatch) -> None:
    import hashlib

    code = b"package com.donor.mod;\npublic class OreFeature {}"
    blob = "7" * 40
    monkeypatch.setattr(
        source_transplant,
        "_fetch_blob_bytes",
        lambda _client, _repo, _sha: code,
    )
    donor = source_transplant.DonorSlice(
        capability="worldgen.ore",
        repository="example/ores",
        commit_sha="7" * 40,
        license_id="MIT",
        source_url="https://github.com/example/ores",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/OreFeature.java",
                blob_sha=blob,
                sha256="sha256:" + hashlib.sha256(code).hexdigest(),
                size_bytes=len(code),
                symbols=("OreFeature",),
            ),
        ),
        seed_files=("src/main/java/OreFeature.java",),
        source_symbols=("OreFeature",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.9,
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
            "tests_executed": 3,
            "tests_passed_count": 3,
            "executed_test_ids": ["SomeTest", "OtherTest"],
        },
    )

    assert receipt.proof_level == "MATERIALIZED"
    assert receipt.compile_passed is False
    assert receipt.tests_passed is False
    assert receipt.authoritative_compile is False
    assert receipt.failure_code == "NON_AUTHORITATIVE_COMPILE_CHECKER"


def test_requirement_acceptance_contract_mapping() -> None:
    donor = source_transplant.DonorSlice(
        capability="magic.spell",
        repository="example/magic",
        commit_sha="8888888888888888888888888888888888888888",
        license_id="MIT",
        source_url="https://github.com/example/magic",
        target_compatibility="metadata_exact",
        files=(),
        seed_files=(),
        source_symbols=("SpellCast",),
        required_dependencies=(),
        donor_tests=("SpellCastAcceptanceTest", "ManaDrainAcceptanceTest"),
        confidence=0.95,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    # 1. Partial acceptance pass (1 out of 2 tests passed) -> COMPILE_VERIFIED
    def mock_partial_pass(files, context):
        return {
            "compile_passed": True,
            "tests_passed": True,
            "tests_executed": 1,
            "tests_passed_count": 1,
            "executed_test_ids": ["SpellCastAcceptanceTest"],
        }

    receipt_partial = execute_reuse_proof(donor, target_workspace="", target_context={}, compile_checker=mock_partial_pass)
    assert receipt_partial.proof_level == "COMPILE_VERIFIED"
    assert any(item[0] == "REQ-MAGIC-001" and item[3] is True for item in receipt_partial.requirement_acceptance_map)
    assert any(item[0] == "REQ-MAGIC-002" and item[3] is False for item in receipt_partial.requirement_acceptance_map)

    # 2. Complete acceptance pass (both tests passed) -> BEHAVIOR_VERIFIED
    def mock_complete_pass(files, context):
        return {
            "compile_passed": True,
            "tests_passed": True,
            "tests_executed": 2,
            "tests_passed_count": 2,
            "executed_test_ids": ["SpellCastAcceptanceTest", "ManaDrainAcceptanceTest"],
        }

    receipt_complete = execute_reuse_proof(donor, target_workspace="", target_context={}, compile_checker=mock_complete_pass)
    assert receipt_complete.proof_level == "BEHAVIOR_VERIFIED"
    assert all(item[3] is True for item in receipt_complete.requirement_acceptance_map)


def test_multi_layer_typed_artifact_closure_slicing() -> None:
    from minecraft_mod_ai.reuse_proof_executor import (
        _compute_dependency_closed_subgraphs,
    )

    adapted_files = {
        "src/main/java/BossEntity.java": "package com.mod;\npublic class BossEntity {\n    Identifier ID = Identifier.of(\"modid\", \"boss\");\n}",
        "src/main/java/BossRenderer.java": "package com.mod;\nimport com.mod.BossEntity;\npublic class BossRenderer {}",
        "assets/modid/models/entity/boss.json": '{"textures": {"layer0": "modid:entity/boss"}}',
        "assets/modid/textures/entity/boss.png": b"fake png bytes",
        "src/main/java/UnrelatedItem.java": "package com.mod;\npublic class UnrelatedItem {}",
    }

    donor = source_transplant.DonorSlice(
        capability="boss.entity",
        repository="example/boss-full",
        commit_sha="9999999999999999999999999999999999999999",
        license_id="MIT",
        source_url="https://github.com/example/boss-full",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile("src/main/java/BossEntity.java", "s1", "sha:1", 100, ("BossEntity",)),
            source_transplant.DonorFile("src/main/java/BossRenderer.java", "s2", "sha:2", 100, ("BossRenderer",)),
            source_transplant.DonorFile("assets/modid/models/entity/boss.json", "s3", "sha:3", 50, ()),
            source_transplant.DonorFile("assets/modid/textures/entity/boss.png", "s4", "sha:4", 50, ()),
            source_transplant.DonorFile("src/main/java/UnrelatedItem.java", "s5", "sha:5", 100, ("UnrelatedItem",)),
        ),
        seed_files=("src/main/java/BossEntity.java",),
        source_symbols=("BossEntity", "BossRenderer", "UnrelatedItem"),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    subgraphs = _compute_dependency_closed_subgraphs(adapted_files, donor)
    # BossEntity, BossRenderer, boss.json, and boss.png must be grouped into one closed component!
    boss_comp = next((comp for comp in subgraphs if "src/main/java/BossEntity.java" in comp), None)
    assert boss_comp is not None
    assert "src/main/java/BossRenderer.java" in boss_comp
    assert "assets/modid/models/entity/boss.json" in boss_comp
    assert "assets/modid/textures/entity/boss.png" in boss_comp
    # UnrelatedItem must be in its own separate subgraph
    assert "src/main/java/UnrelatedItem.java" not in boss_comp


def test_individual_requirement_test_verification() -> None:
    donor = source_transplant.DonorSlice(
        capability="boss.entity",
        repository="example/boss-contract",
        commit_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        license_id="MIT",
        source_url="https://github.com/example/boss-contract",
        target_compatibility="metadata_exact",
        files=(),
        seed_files=(),
        source_symbols=("BossEntity",),
        required_dependencies=(),
        donor_tests=("BossSpawnTest", "BossHealthStateTest", "BossPhaseAITest", "BossLootDeathTest"),
        confidence=0.95,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    # 3 out of 4 tests pass individually; BossPhaseAITest fails individually in test XML
    def mock_per_test_results(files, context):
        return {
            "compile_passed": True,
            "tests_passed": False,
            "tests_executed": 4,
            "tests_passed_count": 3,
            "executed_test_ids": ["BossSpawnTest", "BossHealthStateTest", "BossPhaseAITest", "BossLootDeathTest"],
            "individual_test_results": {
                "BossSpawnTest": True,
                "BossHealthStateTest": True,
                "BossPhaseAITest": False,  # INDIVIDUAL FAILURE
                "BossLootDeathTest": True,
            },
        }

    receipt = execute_reuse_proof(donor, target_workspace="", target_context={}, compile_checker=mock_per_test_results)
    assert receipt.proof_level == "COMPILE_VERIFIED"
    assert any(item[0] == "REQ-BOSS-001" and item[3] is True for item in receipt.requirement_acceptance_map)
    assert any(item[0] == "REQ-BOSS-002" and item[3] is True for item in receipt.requirement_acceptance_map)
    assert any(item[0] == "REQ-BOSS-003" and item[3] is False for item in receipt.requirement_acceptance_map)
    assert any(item[0] == "REQ-BOSS-004" and item[3] is True for item in receipt.requirement_acceptance_map)


def test_proof_level_fail_closed_validation() -> None:
    from minecraft_mod_ai.proof_level import ProofLevel

    # 1. Valid conversions
    assert ProofLevel.from_value("COMPILE_VERIFIED") == ProofLevel.COMPILE_VERIFIED
    assert ProofLevel.from_value("BEHAVIOR_VERIFIED") == ProofLevel.BEHAVIOR_VERIFIED
    assert ProofLevel.from_value("PARTIAL_REUSE") == ProofLevel.PARTIAL_REUSE

    # 2. Unknown/invalid values evaluate strictly to UNVERIFIED (fail-closed)
    assert ProofLevel.from_value("UNKNOWN_PROOF_STATE") == ProofLevel.UNVERIFIED
    assert ProofLevel.from_value(None) == ProofLevel.UNVERIFIED
    assert ProofLevel.from_value(123) == ProofLevel.UNVERIFIED

    # 3. Method checks
    assert ProofLevel.COMPILE_VERIFIED.is_verified() is True
    assert ProofLevel.PARTIAL_REUSE.is_verified() is False
    assert ProofLevel.PARTIAL_REUSE.is_partial() is True
    assert ProofLevel.UNVERIFIED.allows_reuse() is False


def test_requirement_catalog_and_capability_specs() -> None:
    from minecraft_mod_ai.requirement_catalog import build_requirement_catalog
    from minecraft_mod_ai.canonical_capability_ontology import resolve_capabilities_from_phrase_structured

    prompt = "common mobs through bosses scale with level\nbosses drop dedicated loot when defeated"
    resolution = resolve_capabilities_from_phrase_structured(prompt)
    catalog = build_requirement_catalog(prompt, resolution)

    assert len(catalog.requirements) == 2
    assert catalog.requirements[0].id == "REQ-001"
    assert "common mobs through bosses" in catalog.requirements[0].statement
    assert catalog.requirements[0].mandatory is True

    assert len(catalog.capabilities) >= 2
    for cap in catalog.capabilities:
        assert cap.id != ""
        assert len(cap.source_requirement_ids) > 0


def test_artifact_dependency_graph_scc_and_directional_closure() -> None:
    from minecraft_mod_ai.artifact_dependency_graph import ArtifactDependencyGraph

    # Directed dependency chain: UI -> TradeCore -> PersistenceStore
    # Plus a cycle: BossEntity <-> BossGoal
    # Plus an isolated StandaloneUtil
    files = {
        "src/main/java/UI.java": "package com.mod;\nimport com.mod.TradeCore;\npublic class UI { TradeCore core; }",
        "src/main/java/TradeCore.java": "package com.mod;\nimport com.mod.PersistenceStore;\npublic class TradeCore { PersistenceStore store; }",
        "src/main/java/PersistenceStore.java": "package com.mod;\npublic class PersistenceStore {}",
        "src/main/java/BossEntity.java": "package com.mod;\nimport com.mod.BossGoal;\npublic class BossEntity { BossGoal goal; }",
        "src/main/java/BossGoal.java": "package com.mod;\nimport com.mod.BossEntity;\npublic class BossGoal { BossEntity boss; }",
        "src/main/java/StandaloneUtil.java": "package com.mod;\npublic class StandaloneUtil {}",
    }

    graph = ArtifactDependencyGraph.build_from_files(files)
    sccs = graph.compute_scc()

    # 1. BossEntity and BossGoal must form a single cyclic SCC
    boss_scc = next((scc for scc in sccs if "src/main/java/BossEntity.java" in scc), None)
    assert boss_scc is not None
    assert "src/main/java/BossGoal.java" in boss_scc
    assert "src/main/java/StandaloneUtil.java" not in boss_scc

    # 2. Strict Directional Transitive Closure:
    # Seed TradeCore must pull in TradeCore + PersistenceStore only (NEVER the reverse dependent UI!)
    trade_closures = graph.compute_directional_closures(seed_nodes=["src/main/java/TradeCore.java"])
    assert len(trade_closures) == 1
    trade_comp = trade_closures[0]
    assert "src/main/java/TradeCore.java" in trade_comp
    assert "src/main/java/PersistenceStore.java" in trade_comp
    assert "src/main/java/UI.java" not in trade_comp  # Reverse dependent must NOT be pulled in!

    # Seed UI pulls in UI + TradeCore + PersistenceStore
    ui_closures = graph.compute_directional_closures(seed_nodes=["src/main/java/UI.java"])
    assert len(ui_closures) == 1
    ui_comp = ui_closures[0]
    assert set(ui_comp) == {"src/main/java/UI.java", "src/main/java/TradeCore.java", "src/main/java/PersistenceStore.java"}

    # Seed PersistenceStore pulls in only PersistenceStore
    persistence_closures = graph.compute_directional_closures(seed_nodes=["src/main/java/PersistenceStore.java"])
    assert len(persistence_closures) == 1
    assert persistence_closures[0] == ["src/main/java/PersistenceStore.java"]

    # 3. Independent graphs do not mix
    assert "src/main/java/StandaloneUtil.java" not in trade_comp
    assert "src/main/java/BossEntity.java" not in trade_comp


def test_dependency_resolver_cross_loader_matrix() -> None:
    from minecraft_mod_ai.dependency_resolver import resolve_dependency_for_target

    receipt_fabric = resolve_dependency_for_target("geckolib", target_loader="fabric", target_minecraft="1.21.1")
    assert receipt_fabric.is_resolved is True
    assert "geckolib-fabric" in receipt_fabric.resolved_coordinate
    assert receipt_fabric.selected_version == "4.6.0"

    receipt_neoforge = resolve_dependency_for_target("geckolib", target_loader="neoforge", target_minecraft="1.21.1")
    assert receipt_neoforge.is_resolved is True
    assert "geckolib-neoforge" in receipt_neoforge.resolved_coordinate


def test_host_acceptance_contracts_and_test_case_receipts() -> None:
    from minecraft_mod_ai.acceptance_contracts import (
        TestCaseReceipt,
        get_host_acceptance_contracts,
    )

    contracts = get_host_acceptance_contracts("boss.entity")
    assert len(contracts) == 4
    assert any(c.requirement_id == "REQ-BOSS-001" for c in contracts)
    assert any(c.host_test_class == "MMM_BossSpawnAcceptanceTest" for c in contracts)

    tc = TestCaseReceipt(
        test_id="MMM_BossSpawnAcceptanceTest.testSpawn",
        requirement_id="REQ-BOSS-001",
        executed=True,
        passed=True,
        duration_ms=12.5,
    )
    assert tc.passed is True
    assert tc.requirement_id == "REQ-BOSS-001"


def test_multi_donor_joint_composition_solver_and_sbom() -> None:
    from minecraft_mod_ai.composition_solver import (
        generate_reuse_manifest,
        solve_multi_donor_composition,
    )

    donor_a = source_transplant.DonorSlice(
        capability="boss.entity",
        repository="example/boss-a",
        commit_sha="1111111111111111111111111111111111111111",
        license_id="MIT",
        source_url="https://github.com/example/boss-a",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile("src/main/java/Boss.java", "b1", "sha:1", 100, ("Boss",)),
        ),
        seed_files=("src/main/java/Boss.java",),
        source_symbols=("Boss",),
        required_dependencies=("geckolib",),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    # 1. Compatible composition with unique files
    donor_b = source_transplant.DonorSlice(
        capability="item.equipment",
        repository="example/item-b",
        commit_sha="2222222222222222222222222222222222222222",
        license_id="MIT",
        source_url="https://github.com/example/item-b",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile("src/main/java/Item.java", "i1", "sha:2", 100, ("Item",)),
        ),
        seed_files=("src/main/java/Item.java",),
        source_symbols=("Item",),
        required_dependencies=("geckolib",),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    res_valid = solve_multi_donor_composition([donor_a, donor_b], target_loader="fabric", target_minecraft="1.21.1")
    assert res_valid.is_valid is True
    assert len(res_valid.conflicts) == 0

    manifest = generate_reuse_manifest([donor_a, donor_b], project_name="my_rpg_mod")
    assert manifest["total_reused_files"] == 2
    assert manifest["files"][0]["origin_repo"] == "example/boss-a"

    # 2. Conflicting composition with duplicate class definition
    donor_collision = source_transplant.DonorSlice(
        capability="boss.variant",
        repository="example/boss-collision",
        commit_sha="3333333333333333333333333333333333333333",
        license_id="MIT",
        source_url="https://github.com/example/boss-collision",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile("src/main/java/Boss.java", "b2", "sha:3", 100, ("Boss",)),
        ),
        seed_files=("src/main/java/Boss.java",),
        source_symbols=("Boss",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    res_conflict = solve_multi_donor_composition([donor_a, donor_collision])
    assert res_conflict.is_valid is False
    assert any(c.conflict_type == "class_collision" for c in res_conflict.conflicts)


def test_proof_transition_validator_rules() -> None:
    from minecraft_mod_ai.proof_level import ProofLevel, validate_proof_transition

    valid, _ = validate_proof_transition(
        ProofLevel.MATERIALIZED,
        ProofLevel.COMPILE_VERIFIED,
        receipt={"compile_passed": True, "authoritative_compile": True},
    )
    assert valid is True

    non_authoritative, non_authoritative_message = validate_proof_transition(
        ProofLevel.MATERIALIZED,
        ProofLevel.COMPILE_VERIFIED,
        receipt={"compile_passed": True},
    )
    assert non_authoritative is False
    assert "authoritative compile" in non_authoritative_message

    invalid_receipt, msg2 = validate_proof_transition(
        ProofLevel.MATERIALIZED,
        ProofLevel.COMPILE_VERIFIED,
        receipt=None,
    )
    assert invalid_receipt is False
    assert "MISSING_RECEIPT" in msg2

    illegal_jump, msg3 = validate_proof_transition(
        ProofLevel.DISCOVERED,
        ProofLevel.COMPILE_VERIFIED,
        receipt={"compile_passed": True, "authoritative_compile": True},
    )
    assert illegal_jump is False
    assert "ILLEGAL_TRANSITION" in msg3


def test_multi_donor_beam_search_composition_solver() -> None:
    from minecraft_mod_ai.composition_solver import search_best_donor_composition

    # Capability 1: Boss options (A1 vs A2)
    boss_a1 = source_transplant.DonorSlice(
        capability="boss.entity",
        repository="example/boss-a1",
        commit_sha="1111111111111111111111111111111111111111",
        license_id="MIT",
        source_url="https://github.com/example/boss-a1",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile("src/main/java/BossA1.java", "b1", "sha:1", 100, ("BossA1",)),
        ),
        seed_files=("src/main/java/BossA1.java",),
        source_symbols=("BossA1",),
        required_dependencies=("geckolib",),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    # Capability 2: Quest options (Q1 conflicts with BossA1, Q2 is clean)
    quest_q1_conflict = source_transplant.DonorSlice(
        capability="quest.system",
        repository="example/quest-q1",
        commit_sha="2222222222222222222222222222222222222222",
        license_id="MIT",
        source_url="https://github.com/example/quest-q1",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile("src/main/java/BossA1.java", "q1", "sha:q1", 100, ("BossA1",)),
        ),
        seed_files=("src/main/java/BossA1.java",),
        source_symbols=("BossA1",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    quest_q2_clean = source_transplant.DonorSlice(
        capability="quest.system",
        repository="example/quest-q2",
        commit_sha="3333333333333333333333333333333333333333",
        license_id="MIT",
        source_url="https://github.com/example/quest-q2",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile("src/main/java/QuestManager.java", "q2", "sha:q2", 100, ("QuestManager",)),
        ),
        seed_files=("src/main/java/QuestManager.java",),
        source_symbols=("QuestManager",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.9,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    candidates = {
        "boss.entity": (boss_a1,),
        "quest.system": (quest_q1_conflict, quest_q2_clean),
    }

    # Beam search must find the valid combination (boss_a1 + quest_q2_clean)
    result = search_best_donor_composition(candidates, target_loader="fabric", target_minecraft="1.21.1")
    assert result.is_valid is True
    assert len(result.selected_donors) == 2
    assert any(d.repository == "example/quest-q2" for d in result.selected_donors)


def test_final_project_assembler_orchestration_and_typed_merge(tmp_path) -> None:
    from dataclasses import replace

    from minecraft_mod_ai.final_project_assembler import FinalProjectAssembler
    from minecraft_mod_ai.reuse_artifacts import ReusableArtifactBundle
    from minecraft_mod_ai.residual_generation_contract import (
        DependencyRequirement,
        ResidualGenerationContract,
    )
    from minecraft_mod_ai.reuse_proof_executor import ResidualWorkOrder

    work_order = ResidualWorkOrder(
        capability="boss.entity",
        reused_classes=("BossEntity.java",),
        reused_symbols=("BossEntity",),
        missing_interfaces=("IBossPhase",),
        missing_resources=("assets/boss_mod/textures/entity/boss.png",),
        unbound_registries=("boss_mod:boss_entity",),
        glue_code_requirements=("Integrate BossEntity with ModEntities",),
    )

    assembler = FinalProjectAssembler(
        tmp_path,
        target_context={
            "loader": "fabric",
            "minecraft_version": "1.21.1",
            "target_modid": "my_rpg_mod",
            "target_package": "ai.minecraft.generated.rpg",
        },
    )
    bundle = ReusableArtifactBundle.from_same_project(
        "boss.entity",
        files={
            "src/main/java/BossEntity.java": (
                b"package ai.minecraft.generated.rpg;\n"
                b"public class BossEntity {}"
            ),
        },
        source_ref="test:boss-entity",
    )
    bundle = replace(
        bundle,
        proof_receipt={
            "schema_version": "mmm/same-project-proof-receipt-v1",
            "proof_level": "HOST_VERIFIED",
            "capability": bundle.capability,
            "bundle_id": bundle.bundle_id,
            "source_ref": bundle.source_ref,
            "file_hashes": dict(bundle.file_hashes),
        },
    )
    residual_contract = ResidualGenerationContract(
        capability="boss.entity",
        requirement_ids=("boss.entity",),
        required_new_artifacts=("src/main/java/IBossPhase.java",),
        required_dependency_changes=(
            DependencyRequirement("example:residual-api:1.0.0"),
        ),
    )

    res = assembler.assemble(
        reused_bundles=(bundle,),
        residual_files={
            "src/main/java/IBossPhase.java": "package ai.minecraft.generated.rpg;\npublic interface IBossPhase {}",
        },
        fresh_files={
            "src/main/java/QuestSystem.java": "package ai.minecraft.generated.rpg;\npublic class QuestSystem {}",
        },
        work_orders=(work_order,),
        residual_contracts=(residual_contract,),
    )

    assert res.is_valid is True
    assert res.reused_file_count == 1
    assert res.residual_file_count == 1
    assert res.fresh_file_count == 1
    assert (tmp_path / "build.gradle").exists()
    assert (tmp_path / "reuse-manifest.json").exists()
    assert (tmp_path / "src/main/java/BossEntity.java").exists()
    assert (tmp_path / "src/main/java/IBossPhase.java").exists()
    assert (tmp_path / "src/main/java/QuestSystem.java").exists()
    assert "example:residual-api:1.0.0" in (tmp_path / "build.gradle").read_text(encoding="utf-8")
    assert (tmp_path / ".minecraft_ai/residual-generation-contracts.json").exists()

    # The assembler-persisted policy becomes the default write lock for coder tools.
    import pytest
    from minecraft_mod_ai.source_patch import SourcePatchError, TransactionalSourcePatcher

    with pytest.raises(SourcePatchError, match="RESIDUAL_WRITE_CONTRACT"):
        TransactionalSourcePatcher(tmp_path).apply(
            [{
                "operation": "create",
                "path": "src/main/java/UndeclaredBossMutation.java",
                "content": "final class UndeclaredBossMutation {}\n",
            }]
        )

    # Test the production planner -> bundle -> contract -> assembler path.
    from minecraft_mod_ai.reuse_planner import (
        CompositionSelection,
        ReuseDecision,
        TargetImplementationPlan,
    )
    from minecraft_mod_ai.platform_catalog import adapter_for_target

    adapter = adapter_for_target("1.21.1", "fabric")
    plan_workspace = tmp_path / "plan-output"
    plan_assembler = FinalProjectAssembler(
        plan_workspace,
        target_context={
            "loader": "fabric",
            "minecraft_version": "1.21.1",
            "target_modid": "my_rpg_mod",
            "target_package": "ai.minecraft.generated.rpg",
        },
    )
    plan_contract = ResidualGenerationContract(
        capability="boss.entity",
        requirement_ids=("boss.entity",),
        required_new_artifacts=("src/main/java/PlanBossPhase.java",),
    )
    plan = TargetImplementationPlan(
        adapter=adapter,
        capabilities=(
            ReuseDecision(
                capability="boss.entity",
                mode="same_project",
                confidence=0.95,
                fresh_implementation_cost=10.0,
                fresh_verification_cost=5.0,
                artifact_bundle=bundle,
                proof_level="HOST_VERIFIED",
                proof_receipt=bundle.proof_receipt,
            ),
        ),
        platform_evidence=None,
        cross_component_integration_cost=0.0,
        platform_verification_cost=0.0,
        maintenance_risk=0.0,
        total_expected_cost=2.0,
        weighted_verified_reuse=0.95,
        fresh_work=0.0,
        adaptation_work=0.0,
        verification_work=0.0,
        uncertainty=0.0,
        reusable_registry_candidates=0,
        selected_composition=CompositionSelection(
            bundles=(bundle,),
            total_covered_requirements=("boss.entity",),
        ),
        residual_contracts=(plan_contract,),
    )

    plan_res = plan_assembler.assemble_plan(
        plan,
        residual_files={"src/main/java/PlanBossPhase.java": "public interface PlanBossPhase {}"},
        fresh_files={"src/main/java/PlanQuestSystem.java": "public class PlanQuestSystem {}"},
    )
    assert plan_res.is_valid is True
    assert plan_res.target_loader == "fabric"
    assert plan_res.target_minecraft == "1.21.1"
    assert plan_res.reused_file_count == 1
    assert plan_res.residual_file_count == 1
    assert (plan_workspace / "assembly-manifest.json").exists()
    assert (plan_workspace / "dependency-lock.json").exists()


def test_final_project_assembler_requires_bundle_bound_proof_receipt(tmp_path) -> None:
    from dataclasses import replace

    from minecraft_mod_ai.final_project_assembler import FinalProjectAssembler
    from minecraft_mod_ai.reuse_artifacts import ReusableArtifactBundle

    assembler = FinalProjectAssembler(
        tmp_path,
        target_context={"loader": "fabric", "minecraft_version": "1.21.1"},
    )
    bundle = ReusableArtifactBundle.from_same_project(
        "trade.transaction",
        files={"src/main/java/TradeTransaction.java": "final class TradeTransaction {}\n"},
        proof_receipt={"proof_level": "HOST_VERIFIED"},
    )

    rejected = assembler.assemble(reused_bundles=(bundle,))

    assert rejected.is_valid is False
    assert rejected.errors == ("BUNDLE_PROOF_REQUIRED: same_project:trade.transaction",)

    component = ReusableArtifactBundle.from_verified_component(
        "component-trade",
        "trade.transaction",
        files={"src/main/java/TradeTransaction.java": "final class TradeTransaction {}\n"},
    )
    component = replace(
        component,
        proof_receipt={
            "schema_version": "mmm/registry-component-proof-receipt-v1",
            "proof_level": "COMPILE_VERIFIED",
            "capability": component.capability,
            "bundle_id": component.bundle_id,
            "source_ref": component.source_ref,
            "file_hashes": dict(component.file_hashes),
        },
    )

    accepted = assembler.assemble(reused_bundles=(component,))

    assert accepted.is_valid is True
    assert accepted.reused_file_count == 1
    assert (tmp_path / "src/main/java/TradeTransaction.java").exists()


def test_residual_generation_contract_write_guards() -> None:
    import pytest
    from minecraft_mod_ai.residual_generation_contract import (
        ResidualGenerationContract,
        ProtectedReuseArtifactError,
        ResidualWritePreconditionError,
        ResidualScopeViolation,
        validate_residual_write,
    )

    contract = ResidualGenerationContract(
        capability="trade.custom_npc",
        protected_artifacts={"src/main/java/TradeNpc.java": "sha256:" + "c" * 64},
        protected_symbols=("TradeNpc",),
        allowed_write_paths=("src/main/java/TradeInterface.java",),
        expected_old_sha256={
            "src/main/java/TradeInterface.java": "a" * 64,
        },
        required_new_artifacts=("src/main/java/NewTradeService.java",),
    )

    # 1. Modifying protected artifact must raise ProtectedReuseArtifactError
    with pytest.raises(ProtectedReuseArtifactError):
        validate_residual_write("src/main/java/TradeNpc.java", None, contract)

    # 2. Writing outside allowed paths / prefixes must raise ResidualScopeViolation
    with pytest.raises(ResidualScopeViolation):
        validate_residual_write("build.gradle", None, contract)

    # 3. Existing files require the exact pre-write bytes from the contract.
    with pytest.raises(ResidualWritePreconditionError):
        validate_residual_write("src/main/java/TradeInterface.java", None, contract)
    with pytest.raises(ResidualWritePreconditionError):
        validate_residual_write("src/main/java/TradeInterface.java", "b" * 64, contract)
    validate_residual_write("src/main/java/TradeInterface.java", "a" * 64, contract)

    # 4. Creates are exact declared files, never a broad source-root prefix.
    validate_residual_write("src/main/java/NewTradeService.java", None, contract)
    with pytest.raises(ResidualScopeViolation):
        validate_residual_write("src/main/java/UndeclaredTradeService.java", None, contract)


def test_reusable_artifact_bundle_and_repository_locator() -> None:
    from minecraft_mod_ai.reuse_artifacts import ReusableArtifactBundle
    from minecraft_mod_ai.repository_artifact_index import RepositoryArtifactIndex
    from minecraft_mod_ai.capability_implementation_locator import CapabilityImplementationLocator

    bundle = ReusableArtifactBundle.from_same_project(
        capability="magic.spell",
        files={"src/main/java/SpellEntity.java": b"public class SpellEntity {}"},
        symbols=("SpellEntity",),
    )
    assert bundle.origin_kind == "same_project"
    assert "src/main/java/SpellEntity.java" in bundle.protected_paths
    assert len(bundle.file_hashes) == 1

    # Test RepositoryArtifactIndex and CapabilityImplementationLocator
    tree = [
        {"path": "src/main/java/com/mod/BossEntity.java", "sha": "b1"},
        {"path": "src/main/resources/assets/boss_mod/models/entity/boss.json", "sha": "b2"},
        {"path": "src/main/resources/assets/boss_mod/textures/entity/boss.png", "sha": "b3"},
    ]
    index = RepositoryArtifactIndex.build_from_tree("example/boss-mod", "sha:1", tree)
    index.populate_java_symbols("src/main/java/com/mod/BossEntity.java", "package com.mod;\npublic class BossEntity {}\nRegistry.register(item, \"boss_mod:boss_entity\");")

    seeds = CapabilityImplementationLocator.locate_seeds("boss.entity", index)
    assert len(seeds) > 0
    assert seeds[0].node_id == "src/main/java/com/mod/BossEntity.java"
    assert seeds[0].score >= 8.0


def test_build_model_and_resource_merge_registry() -> None:
    import pytest

    from minecraft_mod_ai.build_model import BuildModel
    from minecraft_mod_ai.resource_merge_registry import ResourceMergeRegistry

    model = BuildModel.for_target_context(
        {"loader": "fabric", "minecraft_version": "1.21.1"}
    )
    model.add_repository("https://maven.terraformersmc.com/releases/")
    model.add_dependency("com.terraformersmc:modmenu:11.0.0")

    rendered = model.render_gradle(modid="test_mod")
    assert "com.terraformersmc:modmenu:11.0.0" in rendered
    assert "https://maven.terraformersmc.com/releases/" in rendered
    assert "JavaVersion.VERSION_21" in rendered

    tags_a = '{"values": ["mod:item1"]}'
    tags_b = '{"values": ["mod:item2"]}'
    merged_tags, ok, _ = ResourceMergeRegistry.merge(
        "data/mod/tags/items/tools.json", tags_a, tags_b
    )
    assert ok is True
    assert "mod:item1" in merged_tags and "mod:item2" in merged_tags

    lang_a = '{"item.mod.sword": "Iron Sword"}'
    lang_b = '{"item.mod.shield": "Iron Shield"}'
    merged_lang, ok, _ = ResourceMergeRegistry.merge(
        "assets/mod/lang/en_us.json", lang_a, lang_b
    )
    assert ok is True
    assert "Iron Sword" in merged_lang and "Iron Shield" in merged_lang

    with pytest.raises(ValueError, match="TRAVERSAL"):
        ResourceMergeRegistry.canonical_path(
            "../outside.json", target_modid="test_mod"
        )


