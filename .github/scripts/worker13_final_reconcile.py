from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one exact replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_test(path: str, name: str, source: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^def {re.escape(name)}\([^\n]*\).*?(?=^def test_|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"{path}: expected one {name}, found {len(matches)}")
    replacement = dedent(source).strip() + "\n\n\n"
    text = text[: matches[0].start()] + replacement + text[matches[0].end() :]
    target.write_text(text, encoding="utf-8")


replace_once(
    "minecraft_mod_ai/reuse_build_verifier.py",
    '''    from .verified_scaffold_registry import (
        GRADLE_DISTRIBUTION_SHA256S,
        GRADLE_WRAPPER_SHA256S,
        SUPPORTED_TARGET_SPECS,
    )
''',
    '''    from .platform_catalog import adapter_for_target
    from .verified_scaffold_registry import (
        GRADLE_DISTRIBUTION_SHA256S,
        GRADLE_WRAPPER_SHA256S,
        validate_scaffold_buildability,
    )
''',
)

replace_once(
    "minecraft_mod_ai/reuse_build_verifier.py",
    '''    target_spec = SUPPORTED_TARGET_SPECS.get((loader, minecraft_version))
    target_matrix_verified = bool(
        target_spec
        and str(target_spec.get("gradle_version") or "") == gradle_version
        and str(target_spec.get("java_release") or "") == java_version
    )
''',
    '''    provider_adapter = None
    if loader and minecraft_version:
        try:
            provider_adapter = adapter_for_target(minecraft_version, loader)
            validate_scaffold_buildability(provider_adapter)
        except (ValueError, RuntimeError):
            provider_adapter = None
    target_matrix_verified = bool(
        provider_adapter
        and str(provider_adapter.gradle) == gradle_version
        and str(provider_adapter.java_version) == java_version
        and str(provider_adapter.gradle_sha256).casefold() == distribution_sha256
    )
''',
)

replace_test(
    "tests/test_agentic_research_game_design.py",
    "test_sectioned_game_design_generates_each_field_once_with_exact_schema",
    r'''
def test_sectioned_game_design_generates_each_section_once_with_exact_schema() -> None:
    router = _SectionRouter()
    research = {
        "research_brief": {"domains": []},
        "domain_notes": [],
        "deterministic": {},
        "errors": [],
    }

    result = agentic.generate_sectioned_game_design(
        game_design,
        router,
        "연구를 먼저 하고 모드를 설계해줘",
        research=research,
    )

    expected_sections = [
        tuple(fields)
        for _section_id, fields, _properties in agentic._SECTION_SPECS
    ]
    assert len(router.calls) == len(expected_sections)
    assert all(call["role"] == "planner" for call in router.calls)
    assert all(call["response_format"] == "json" for call in router.calls)
    assert all(call["enable_tools"] is False for call in router.calls)
    for fields, call in zip(expected_sections, router.calls, strict=True):
        schema = call["response_schema"]
        assert isinstance(schema, dict)
        section_schema = schema["properties"]["section"]
        assert list(section_schema["properties"]) == list(fields)
        assert section_schema["required"] == list(fields)
        assert section_schema["additionalProperties"] is False
    assert result["title"] == "연구 기반 모드"
    assert result["core_loop"]
    assert result["acceptance_tests"]
    assert "art_direction" not in result
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_reuse_proof_executor_fallback_loop",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_strict_materialized_proof_level_without_compile",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_closure_incomplete_capped_at_partial_reuse",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_loader_aware_scaffold_neoforge_and_wrapper",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_artifact_level_partial_reuse_slicing",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_two_stage_compile_and_test_separation",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_donor_without_tests_capped_at_compile_verified",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_proof_transition_validator_rules",
    r'''
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
''',
)

replace_test(
    "tests/test_canonical_capability_ontology_and_reuse_proof.py",
    "test_build_model_and_resource_merge_registry",
    r'''
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
''',
)

new_test = dedent(
    r'''
    from __future__ import annotations

    from minecraft_mod_ai import reuse_build_verifier as verifier


    def test_build_toolchain_target_matrix_uses_executable_provider(
        monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(verifier, "_java_major_version", lambda: "21")
        wrapper_dir = tmp_path / "gradle" / "wrapper"
        wrapper_dir.mkdir(parents=True)
        (wrapper_dir / "gradle-wrapper.properties").write_text(
            "distributionUrl=https\\://services.gradle.org/distributions/gradle-8.10.2-bin.zip\n"
            + "distributionSha256Sum="
            + ("0" * 64)
            + "\n",
            encoding="utf-8",
        )
        (wrapper_dir / "gradle-wrapper.jar").write_bytes(b"synthetic-test-wrapper")
        (tmp_path / "build.gradle").write_text(
            "plugins { id 'fabric-loom' version 'test-loom' }\n"
            "dependencies { minecraft 'com.mojang:minecraft:1.21.1' }\n",
            encoding="utf-8",
        )

        receipt = verifier._inspect_build_toolchain(tmp_path)

        assert receipt.loader == "fabric"
        assert receipt.minecraft_version == "1.21.1"
        assert receipt.gradle_version == "8.10.2"
        assert receipt.java_version == "21"
        assert receipt.target_matrix_verified is True
    '''
).lstrip()
test_path = ROOT / "tests/test_worker13_reuse_build_verifier_target_authority.py"
if test_path.exists():
    raise RuntimeError(f"{test_path}: unexpected pre-existing Worker 13 regression")
test_path.write_text(new_test, encoding="utf-8")
