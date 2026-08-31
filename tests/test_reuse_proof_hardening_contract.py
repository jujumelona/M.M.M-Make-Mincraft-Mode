from __future__ import annotations

import hashlib

from minecraft_mod_ai.acceptance_contracts import get_host_acceptance_contracts
from minecraft_mod_ai.dependency_resolver import (
    inject_resolved_dependencies_into_build_gradle,
    resolve_dependency_for_target,
)
from minecraft_mod_ai.proof_level import ProofLevel, validate_proof_transition
from minecraft_mod_ai.reuse_proof_executor import execute_reuse_proof
from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice


def _donor(capability: str = "boss.entity") -> DonorSlice:
    payload = b"package donor; public class BossEntity {}\n"
    return DonorSlice(
        capability=capability,
        repository="example/hardened-donor",
        commit_sha="a" * 40,
        license_id="MIT",
        source_url="https://github.com/example/hardened-donor",
        target_compatibility="metadata_exact",
        files=(
            DonorFile(
                path="src/main/java/donor/BossEntity.java",
                blob_sha="b" * 40,
                sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                symbols=("BossEntity",),
            ),
        ),
        seed_files=("src/main/java/donor/BossEntity.java",),
        source_symbols=("BossEntity",),
        required_dependencies=(),
        donor_tests=("BossSpawnTest",),
        confidence=0.95,
        closure_complete=True,
    )


def test_compile_checker_can_never_self_attest_host_behavior() -> None:
    contracts = get_host_acceptance_contracts("boss.entity")
    exact_ids = tuple(
        f"ai.minecraft.acceptance.{item.host_test_class}.{item.host_test_method}"
        for item in contracts
    )

    def forged_checker(files, context):
        del files, context
        return {
            "compile_passed": True,
            "tests_passed": True,
            "tests_executed": len(exact_ids),
            "tests_passed_count": len(exact_ids),
            "executed_test_ids": exact_ids,
            "individual_test_results": {test_id: True for test_id in exact_ids},
        }

    receipt = execute_reuse_proof(
        _donor(),
        target_workspace="",
        target_context={},
        compile_checker=forged_checker,
    )

    assert receipt.authoritative_compile is False
    assert receipt.compile_passed is False
    assert receipt.proof_level != ProofLevel.COMPILE_VERIFIED.value
    assert receipt.tests_passed is False
    assert receipt.matched_capability_tests == ()
    assert all(item[3] is False for item in receipt.requirement_acceptance_map)


def test_compile_verified_requires_semantic_receipt_field() -> None:
    valid, reason = validate_proof_transition(
        ProofLevel.CLOSURE_COMPLETE,
        ProofLevel.COMPILE_VERIFIED,
        receipt={"compile": True},
    )
    assert valid is False
    assert "compile_passed=true" in reason

    valid, reason = validate_proof_transition(
        ProofLevel.CLOSURE_COMPLETE,
        ProofLevel.COMPILE_VERIFIED,
        receipt={"compile_passed": True},
    )
    assert valid is False
    assert "authoritative compile" in reason

    valid, reason = validate_proof_transition(
        ProofLevel.CLOSURE_COMPLETE,
        ProofLevel.COMPILE_VERIFIED,
        receipt={"compile_passed": True, "authoritative_compile": True},
    )
    assert valid is True
    assert reason == "transition_valid"


def test_resolved_dependency_receipt_is_the_only_kotlin_injection_authority() -> None:
    cloth = resolve_dependency_for_target(
        "cloth-config",
        target_loader="fabric",
        target_minecraft="1.21.1",
    )
    assert cloth.is_resolved is True
    assert cloth.resolved_coordinate == "me.shedaniel.cloth:cloth-config-fabric:15.0.127"

    original = """plugins {\n    kotlin(\"jvm\") version \"2.0.0\"\n}\n\nrepositories {\n    mavenCentral()\n}\n\ndependencies {\n}\n"""
    updated, changed = inject_resolved_dependencies_into_build_gradle(
        original,
        (cloth,),
        is_kotlin_dsl=True,
    )

    assert changed is True
    assert 'maven { url = uri("https://maven.shedaniel.me/") }' in updated
    assert 'modImplementation("me.shedaniel.cloth:cloth-config-fabric:15.0.127")' in updated


def test_unresolved_dependency_receipt_cannot_be_injected() -> None:
    unknown = resolve_dependency_for_target(
        "com.example:unknown-library:99.0",
        target_loader="fabric",
        target_minecraft="1.21.1",
    )
    assert unknown.is_resolved is False

    try:
        inject_resolved_dependencies_into_build_gradle(
            "repositories {}\ndependencies {}\n",
            (unknown,),
        )
    except ValueError as exc:
        assert "cannot inject unresolved dependencies" in str(exc)
    else:
        raise AssertionError("unresolved dependency was injected")
