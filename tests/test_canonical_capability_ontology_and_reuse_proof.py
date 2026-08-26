from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.canonical_capability_ontology import (
    canonical_domain_map,
    resolve_capabilities_from_phrase,
    resolve_capabilities_from_phrase_structured,
)
from minecraft_mod_ai.capability_semantic_inference import (
    enrich_resolution_with_semantic_inference,
)
from minecraft_mod_ai.evidence_first_planning import _DOMAIN_TERM_MAP
from minecraft_mod_ai.reuse_planner import decompose_capability_graph, _CAPABILITY_HINTS
from minecraft_mod_ai.reuse_adapters import apply_deterministic_adapters
from minecraft_mod_ai.reuse_proof_executor import (
    execute_candidate_fallback_loop,
)
from minecraft_mod_ai import source_transplant


def test_unified_ontology_zero_drift() -> None:
    ontology_map = canonical_domain_map()
    assert _DOMAIN_TERM_MAP == ontology_map
    assert _CAPABILITY_HINTS == ontology_map

    assert "trade" in ontology_map
    assert "boss" in ontology_map
    assert "nuclear" in ontology_map
    assert "medieval" in ontology_map


def test_medieval_theme_subsystem_archetype_expansion() -> None:
    caps = resolve_capabilities_from_phrase("중세 모드")
    assert len(caps) >= 5
    assert "trade.shop_registry" in caps
    assert "economy.currency" in caps
    assert "item.equipment" in caps
    assert "quest.state" in caps
    assert "worldgen.structure" in caps

    graph = decompose_capability_graph("중세 모드 만들어줘")
    assert len(graph.nodes) >= 5
    assert any("trade" in node or "shop" in node for node in graph.nodes)
    assert any("economy" in node or "currency" in node for node in graph.nodes)
    assert any("structure" in node or "worldgen" in node for node in graph.nodes)


def test_medieval_banking_unresolved_concept_preservation() -> None:
    # "중세 은행 대출 시스템": "중세" archetype expands while "은행 대출" is preserved as unresolved
    res = resolve_capabilities_from_phrase_structured("중세 은행 대출 시스템")
    origins = {n.capability_id: n.origin for n in res.nodes}

    assert any(origins.get(k) == "archetype_inferred" for k in origins)
    assert any(origins.get(k) == "unresolved_concept" for k in origins)
    assert len(res.unresolved_spans) >= 1
    assert "은행 대출" in res.unresolved_spans[0] or "eunhaeng" in res.nodes[-1].capability_id

    # Semantic enrichment converts unresolved concept to provisional capabilities
    enriched = enrich_resolution_with_semantic_inference(res)
    assert any(node.capability_id.startswith("provisional:") for node in enriched.nodes)
    assert any("persistence.state_store" in edge[1] for edge in enriched.edges)


def test_nuclear_fusion_generator_functional_decomposition() -> None:
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


def test_reuse_proof_executor_fallback_loop() -> None:
    # Candidate A: fails compile verification
    donor_a = source_transplant.DonorSlice(
        capability="combat.damage",
        repository="example/broken-mod",
        commit_sha="1111111111111111111111111111111111111111",
        license_id="MIT",
        source_url="https://github.com/example/broken-mod",
        target_compatibility="adapt",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Broken.java",
                blob_sha="b1",
                sha256="sha256:1111",
                size_bytes=100,
                symbols=("Broken",),
            ),
        ),
        seed_files=("src/main/java/Broken.java",),
        source_symbols=("Broken",),
        required_dependencies=("heavy_library",),
        donor_tests=(),
        confidence=0.95,
        adaptation_cost=45.0,
        closure_complete=True,
    )

    # Candidate B: passes compile verification
    donor_b = source_transplant.DonorSlice(
        capability="combat.damage",
        repository="example/clean-mod",
        commit_sha="2222222222222222222222222222222222222222",
        license_id="MIT",
        source_url="https://github.com/example/clean-mod",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Clean.java",
                blob_sha="b2",
                sha256="sha256:2222",
                size_bytes=80,
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

    def mock_checker(files, context):
        if any("Broken" in p for p in files):
            return {"compile_passed": False, "unresolved_symbols": ["MissingDep"]}
        return {"compile_passed": True}

    selected_donor, receipts = execute_candidate_fallback_loop(
        candidates=[donor_a, donor_b],
        capability="combat.damage",
        target_workspace="/tmp/fake_ws",
        target_context={"target_package": "ai.test"},
        compile_checker=mock_checker,
    )

    assert selected_donor is not None
    assert selected_donor.repository == "example/clean-mod"
    assert len(receipts) == 2
    assert receipts[0].compile_passed is False
    assert receipts[1].compile_passed is True
    assert receipts[1].proof_level == "COMPILE_VERIFIED"


def test_unpunctuated_natural_korean_maplestory_prompt_regression() -> None:
    # Verbatim exact user failure prompt without punctuation or explicit arrow separators
    prompt = "메이플스토리 모드 만들어줘 잡몹부터 보스까지 템들 레벨도 점점 성장 강화시스템등 모두 구현해야해"

    # 1. Request Catalog must contain all decomposed requirements, not collapsed to caps[0]
    from minecraft_mod_ai.evidence_first_planning import build_request_catalog
    catalog = build_request_catalog(prompt, {})
    req_caps = {req["capability"] for req in catalog["requirements"]}

    assert "mob.spawning" in req_caps
    assert "boss.entity" in req_caps
    assert "item.equipment" in req_caps or "item.weapon" in req_caps
    assert "progression.level" in req_caps
    assert "item.upgrade" in req_caps

    # 2. Decompose Capability Graph must retain all subsystems
    graph = decompose_capability_graph(prompt)
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
    fake_code = b"package com.donor.mod;\npublic class Item {}"
    import hashlib
    fake_sha = "sha256:" + hashlib.sha256(fake_code).hexdigest()

    def mock_fetch(client, repo, blob_sha):
        return fake_code

    monkeypatch.setattr(source_transplant, "_fetch_blob_bytes", mock_fetch)

    donor = source_transplant.DonorSlice(
        capability="item.equipment",
        repository="example/mod",
        commit_sha="3333333333333333333333333333333333333333",
        license_id="MIT",
        source_url="https://github.com/example/mod",
        target_compatibility="metadata_exact",
        files=(
            source_transplant.DonorFile(
                path="src/main/java/Item.java",
                blob_sha="3333333333333333333333333333333333333333",
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

    # When materialization succeeds but no compiler is executed, proof level MUST be MATERIALIZED (no fake compile verification)
    from minecraft_mod_ai.reuse_proof_executor import execute_reuse_proof
    receipt = execute_reuse_proof(donor, target_workspace="", target_context={})

    assert receipt.compile_passed is False
    assert receipt.tests_passed is False
    assert receipt.proof_level == "MATERIALIZED"


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
        donor_tests=(),
        confidence=0.95,
        adaptation_cost=0.0,
        closure_complete=True,
    )

    evaluated_files = {}

    def mock_compile(files, context):
        evaluated_files.update(files)
        return {"compile_passed": True, "tests_passed": True}

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
    from minecraft_mod_ai.reuse_planner import ReuseDecision, TargetImplementationPlan
    from minecraft_mod_ai.platform_catalog import adapter_for_target

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


def test_diagnostic_repair_adapter_generates_stubs() -> None:
    from minecraft_mod_ai.reuse_adapters import DiagnosticRepairAdapter

    files = {"src/main/java/ai/mod/Main.java": "package ai.mod;\npublic class Main {}"}
    repaired = DiagnosticRepairAdapter.repair_unresolved_symbols(
        files,
        unresolved_symbols=["CustomBossEntity", "SpecialWeaponItem"],
        target_context={"target_package": "ai.mod"},
    )

    assert len(repaired) == 2
    assert "src/main/java/ai/mod/CustomBossEntity.java" in files
    assert "src/main/java/ai/mod/SpecialWeaponItem.java" in files
    assert "public class CustomBossEntity" in files["src/main/java/ai/mod/CustomBossEntity.java"]


def test_scaffold_minimal_ephemeral_workspace(tmp_path) -> None:
    from minecraft_mod_ai.reuse_proof_executor import scaffold_minimal_ephemeral_workspace

    scaffold_minimal_ephemeral_workspace(
        tmp_path,
        target_context={
            "minecraft_version": "1.21.1",
            "target_modid": "maple_mod",
            "java_version": "21",
        },
    )

    assert (tmp_path / "build.gradle").exists()
    assert (tmp_path / "settings.gradle").exists()
    assert (tmp_path / "gradle.properties").exists()

    bg_text = (tmp_path / "build.gradle").read_text(encoding="utf-8")
    assert "archivesName = 'maple_mod'" in bg_text
    assert "com.mojang:minecraft:1.21.1" in bg_text



