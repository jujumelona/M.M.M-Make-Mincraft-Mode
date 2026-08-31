from __future__ import annotations

import zipfile
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
from minecraft_mod_ai import atomic_requirement_contract, quality_evidence
from minecraft_mod_ai.atomic_requirement_contract import (
    AtomicRequirementError,
    compile_ir,
    semantic_review,
    validate_ir,
)
from minecraft_mod_ai.clean_room_verification_contract import (
    SCHEMA as CLEAN_ROOM_SCHEMA,
)
from minecraft_mod_ai.clean_room_verification_contract import (
    jar_content_sha256,
)
from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.validation_diagnostic_contract import (
    diagnostic_errors,
    flatten_diagnostics,
)
from minecraft_mod_ai.repair_engine import RepairEngine


def _proposal(prompt: str, *, related: bool = True) -> SimpleNamespace:
    module = SimpleNamespace(
        module_id="frost_sword",
        kind="weapon",
        config={
            "description": (
                "frost sword freezes enemies"
                if related
                else "unrelated copper storage machine"
            )
        },
        depends_on=(),
        required_gates=(),
    )
    acceptance = (
        "The frost sword freezes enemies in a runtime test."
        if related
        else "Copper storage machine opens."
    )
    return SimpleNamespace(
        requested_prompt=prompt,
        modules=(module,),
        assets=(),
        acceptance_tests=(acceptance,),
        game_design={},
    )


def test_atomic_ir_covers_directly_supported_requirement() -> None:
    proposal = _proposal("Add a frost sword that freezes enemies.")
    ir = compile_ir(proposal)
    assert ir["schema_version"] == "mmm/atomic-requirement-ir-v1"
    assert ir["atom_count"] == 1
    assert ir["unresolved_atom_ids"] == []
    assert ir["atoms"][0]["status"] == "COVERED"
    assert ir["atoms"][0]["implementation_refs"] == [
        "implementation:module:frost_sword"
    ]


def test_atomic_ir_never_fabricates_coverage() -> None:
    proposal = _proposal("Add a lunar portal.", related=False)
    ir = compile_ir(proposal)
    assert ir["unresolved_atom_ids"] == [ir["atoms"][0]["atom_id"]]
    assert ir["atoms"][0]["implementation_refs"] == []
    assert ir["atoms"][0]["status"] == "REVIEW_REQUIRED"


def test_enumerated_request_items_become_separate_atoms() -> None:
    proposal = _proposal(
        "Add a frost sword, add a lunar portal, add a frost shield.",
        related=True,
    )
    ir = compile_ir(proposal)
    assert ir["atom_count"] == 3
    assert [atom["text"] for atom in ir["atoms"]] == [
        "Add a frost sword",
        "add a lunar portal",
        "add a frost shield.",
    ]
    assert len(ir["unresolved_atom_ids"]) >= 1


def test_semantic_reviewer_rejects_out_of_range_candidate_indexes() -> None:
    proposal = _proposal("Add a lunar portal.", related=False)
    ir = compile_ir(proposal)

    class Router:
        def generate_tool_decision(self, *_args, **_kwargs):
            return {
                "supported": True,
                "implementation_indexes": [999],
                "acceptance_indexes": [0],
            }

    with pytest.raises(AtomicRequirementError, match="invalid index"):
        semantic_review(Router(), proposal, ir)


def test_atomic_ir_hash_and_catalog_bindings_are_fail_closed() -> None:
    proposal = _proposal("Add a frost sword that freezes enemies.")
    ir = compile_ir(proposal)
    proposal.game_design["_atomic_requirement_ir"] = ir
    assert validate_ir(proposal) is ir
    tampered = dict(ir)
    tampered["prompt_sha256"] = "sha256:" + "0" * 64
    proposal.game_design["_atomic_requirement_ir"] = tampered
    with pytest.raises(AtomicRequirementError, match="request binding"):
        validate_ir(proposal)


def test_jdt_mapping_is_flattened_and_only_errors_block() -> None:
    receipt = {
        "diagnostics": {
            "file:///A.java": [
                {"severity": 1, "message": "cannot resolve symbol"},
                {"severity": 2, "message": "warning"},
            ],
            "file:///B.java": [{"severity": 3, "message": "info"}],
        }
    }
    flattened = flatten_diagnostics(receipt)
    assert [item["message"] for item in flattened] == [
        "cannot resolve symbol",
        "warning",
        "info",
    ]
    assert [item["message"] for item in diagnostic_errors(receipt)] == [
        "cannot resolve symbol"
    ]


def test_repair_signature_uses_real_jdt_mapping() -> None:
    first = RepairEngine._signature(
        {
            "diagnostics": {
                "diagnostics": {
                    "file:///A.java": [
                        {"severity": 1, "message": "first", "code": "x"}
                    ]
                }
            },
            "build": {"status": "SKIPPED"},
        }
    )
    second = RepairEngine._signature(
        {
            "diagnostics": {
                "diagnostics": {
                    "file:///A.java": [
                        {"severity": 1, "message": "second", "code": "x"}
                    ]
                }
            },
            "build": {"status": "SKIPPED"},
        }
    )
    assert first != second


def test_clean_room_proof_is_required_for_build_quality(tmp_path) -> None:
    jar = tmp_path / "verified.jar"
    jar.write_bytes(b"jar")
    digest = quality_evidence._regular_file_sha256(jar)
    assert digest is not None
    live_only = {
        "status": "PASS",
        "commands": [{"name": "build", "exit_code": 0, "timed_out": False}],
    }
    assert quality_evidence._clean_build_evidence(live_only) is None
    proven = {
        **live_only,
        "clean_room_build": {
            "schema_version": CLEAN_ROOM_SCHEMA,
            "status": "PASS",
            "source_fingerprint": "abc123",
            "build": {
                "status": "PASS",
                "commands": [
                    {"name": "build", "exit_code": 0, "timed_out": False}
                ],
            },
            "jar_validation": {"status": "PASS", "checks_run": 7},
            "jar_path": str(jar),
            "jar_sha256": digest,
            "live_jar_content_sha256": "sha256:" + "1" * 64,
            "clean_jar_content_sha256": "sha256:" + "1" * 64,
        },
    }
    evidence = quality_evidence._clean_build_evidence(proven)
    assert evidence is not None
    assert any(ref.startswith("clean-room-build:") for ref in evidence[0])


def test_jar_content_hash_ignores_zip_metadata(tmp_path) -> None:
    first = tmp_path / "first.jar"
    second = tmp_path / "second.jar"
    info_a = zipfile.ZipInfo("example/Foo.class", date_time=(2020, 1, 1, 0, 0, 0))
    info_b = zipfile.ZipInfo("example/Foo.class", date_time=(2025, 1, 1, 0, 0, 0))
    with zipfile.ZipFile(first, "w") as archive:
        archive.writestr(info_a, b"same-bytecode")
    with zipfile.ZipFile(second, "w") as archive:
        archive.writestr(info_b, b"same-bytecode")
    assert jar_content_sha256(first) == jar_content_sha256(second)


def test_incremental_inner_build_alias_does_not_replace_clean_room_quality() -> None:
    build = {
        "commands": [{"name": "build", "exit_code": 0, "timed_out": False}]
    }
    assert CompleteProductionOrchestrator._command_receipt_passed(build, "clean_build")


def test_final_architecture_contracts_are_installed_without_atomic_plan_gate() -> None:
    assert not getattr(CompleteGameDesignPlanner.plan, "_mmm_atomic_requirement_ir", False)
    assert not getattr(
        CompleteProductionOrchestrator.execute,
        "_mmm_atomic_release_guard",
        False,
    )
    assert getattr(
        CompleteProductionOrchestrator._evaluate_quality,
        "_mmm_clean_room_quality",
        False,
    )
    assert getattr(
        quality_evidence._clean_build_evidence,
        "_mmm_clean_room_required",
        False,
    )
    assert getattr(RepairEngine._signature, "_mmm_flattened_jdt", False)
    assert getattr(RepairEngine._context, "_mmm_flattened_jdt", False)
    assert getattr(
        atomic_requirement_contract._implementations,
        "_mmm_compact_catalog",
        False,
    )
    assert getattr(
        atomic_requirement_contract._atom_ranges,
        "_mmm_enumeration_atomizer",
        False,
    )
    assert getattr(
        quality_evidence.compile_quality_evidence,
        "_mmm_atomic_correctness_evidence",
        False,
    )
    assert (
        orchestrator_module.compile_quality_evidence
        is quality_evidence.compile_quality_evidence
    )
