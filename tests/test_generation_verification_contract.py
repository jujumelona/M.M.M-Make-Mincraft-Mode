from __future__ import annotations

import hashlib
import inspect
import json

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerator,
    _host_finalize_missing_generation_verification,
)
from minecraft_mod_ai.generation_verification_contract import (
    candidate_rank_key,
    classify_generation_verification,
)


JAVA_PATH = "src/main/java/demo/Test.java"
RESOURCE_PATH = "src/main/resources/assets/demo/value.json"


def _receipt(
    status: str = "PASS",
    *,
    target_path: str = JAVA_PATH,
    authority: str = "generation_tool_loop",
    verifier_tool: str | None = "target_compile",
    compile_backed_java: bool = True,
    validation_status: str | None = None,
    termination_reason: str | None = None,
):
    return {
        "schema_version": "mmm/generation-verification-v1",
        "status": status,
        "authority": authority,
        "validation_status": (
            validation_status
            if validation_status is not None
            else ("PASS" if status == "PASS" else "DEFERRED")
        ),
        "termination_reason": (
            termination_reason
            if termination_reason is not None
            else (
                "VERIFICATION_PASSED"
                if status == "PASS"
                else "VERIFICATION_DEFERRED_TO_TARGET_COMPILE"
            )
        ),
        "verifier_tool": verifier_tool,
        "target_path": target_path,
        "compile_backed_java": compile_backed_java,
        "downstream_required_gate": (
            "target_compile"
            if status == "DEFERRED_TO_TARGET_COMPILE"
            else None
        ),
    }


def _classify(receipt, *, touched=(JAVA_PATH,), gates=("target_compile",)):
    return classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=receipt,
        touched_paths=touched,
        required_gates=gates,
    )


def test_compile_backed_pass_is_bound_to_exact_target_compile_receipt() -> None:
    result = _classify(_receipt())

    assert result["generation_status"] == "PASS"
    assert result["verifier_tier"] == 2
    assert result["receipt_target_matches"] is True
    assert result["receipt_semantics_valid"] is True


def test_deferred_compile_is_admissible_only_with_exact_downstream_gate() -> None:
    result = _classify(_receipt("DEFERRED_TO_TARGET_COMPILE"))

    assert result["generation_status"] == "DEFERRED_TO_TARGET_COMPILE"
    assert result["verifier_tier"] == 1
    assert result["downstream_required_gate"] == "target_compile"


def test_equivalent_relative_path_and_gate_spelling_are_canonicalized() -> None:
    result = classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=_receipt(target_path="./src/main/java/demo/Test.java"),
        touched_paths=(JAVA_PATH,),
        required_gates=("target compile",),
    )

    assert result["receipt_target_matches"] is True
    assert result["target_compile_required"] is True
    assert result["verifier_tier"] == 2


def test_java_target_compile_gate_cannot_be_downgraded_by_receipt_flag() -> None:
    result = classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=_receipt(
            compile_backed_java=False,
            verifier_tool="target_compile",
        ),
        touched_paths=(JAVA_PATH,),
        required_gates=("target_compile",),
    )

    assert result["java_target"] is True
    assert result["compile_required"] is True
    assert result["compile_backed_java"] is False
    assert result["receipt_semantics_valid"] is False
    assert result["verifier_tier"] == 0


def test_target_compile_gate_never_accepts_non_java_receipt_as_compile_pass() -> None:
    result = classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=_receipt(
            target_path=RESOURCE_PATH,
            verifier_tool="target_compile",
            compile_backed_java=True,
        ),
        touched_paths=(RESOURCE_PATH,),
        required_gates=("target_compile",),
    )

    assert result["java_target"] is False
    assert result["compile_required"] is True
    assert result["receipt_semantics_valid"] is False
    assert result["verifier_tier"] == 0


def test_verifier_receipt_cannot_be_replayed_for_a_different_file() -> None:
    result = _classify(_receipt(target_path="src/main/java/demo/Other.java"))

    assert result["receipt_target_matches"] is False
    assert result["receipt_semantics_valid"] is False
    assert result["verifier_tier"] == 0


def test_compile_backed_pass_rejects_non_compile_verifier() -> None:
    result = _classify(_receipt(verifier_tool="java_diagnostics"))

    assert result["receipt_target_matches"] is True
    assert result["receipt_semantics_valid"] is False
    assert result["verifier_tier"] == 0


def test_receipt_authority_is_fail_closed() -> None:
    result = _classify(_receipt(authority="model"))

    assert result["verification_authority"] == "unknown"
    assert result["verifier_tier"] == 0


def test_non_java_generation_pass_can_use_non_compile_host_verification() -> None:
    result = classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=_receipt(
            target_path=RESOURCE_PATH,
            verifier_tool=None,
            compile_backed_java=False,
        ),
        touched_paths=(RESOURCE_PATH,),
        required_gates=("resource",),
    )

    assert result["generation_status"] == "PASS"
    assert result["verifier_tier"] == 2


def test_rank_is_strictly_lexicographic_by_verifier_tier() -> None:
    passed = _classify(_receipt())
    deferred = _classify(_receipt("DEFERRED_TO_TARGET_COMPILE"))

    pass_key = candidate_rank_key(
        score=-1_000_000_000.0,
        candidate_index=999,
        verifier=passed,
        patch_size=99_999_999,
    )
    deferred_key = candidate_rank_key(
        score=1_000_000_000.0,
        candidate_index=0,
        verifier=deferred,
        patch_size=1,
    )

    assert pass_key < deferred_key


def test_single_candidate_generator_enforces_shared_verifier_contract() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)

    assert "classify_generation_verification" in source
    assert "GENERATION_VERIFICATION_RECEIPT_INVALID" in source


def _project_receipt(touched_paths):
    normalized = tuple(sorted(set(touched_paths)))
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "mmm/generation-verification-v1",
        "status": "DEFERRED_TO_PROJECT_BUILD",
        "authority": "generation_tool_loop",
        "validation_status": "DEFERRED",
        "termination_reason": "VERIFICATION_DEFERRED_TO_PROJECT_BUILD",
        "verifier_tool": "project_build",
        "target_path": None,
        "compile_backed_java": False,
        "downstream_required_gate": "project_build",
        "verification_scope": "project",
        "touched_paths_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "touched_path_count": len(normalized),
    }


def test_project_build_deferral_binds_complete_touched_path_set() -> None:
    touched = (JAVA_PATH, RESOURCE_PATH)
    result = classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=_project_receipt(touched),
        touched_paths=touched,
        required_gates=("project build",),
    )

    assert result["generation_status"] == "DEFERRED_TO_PROJECT_BUILD"
    assert result["verifier_tier"] == 1
    assert result["project_build_required"] is True
    assert result["project_scope_matches"] is True
    assert result["receipt_target_matches"] is False


def test_project_build_deferral_rejects_changed_touched_path_set() -> None:
    receipt = _project_receipt((JAVA_PATH, RESOURCE_PATH))
    result = classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=receipt,
        touched_paths=(JAVA_PATH,),
        required_gates=("project build",),
    )

    assert result["generation_status"] == "FAIL"
    assert result["project_scope_matches"] is False
    assert result["verifier_tier"] == 0


def test_authored_project_build_scope_overrides_fragment_local_pass(tmp_path) -> None:
    fragment_receipt = _receipt(
        target_path=JAVA_PATH,
        verifier_tool=None,
        compile_backed_java=False,
    )
    touched = (JAVA_PATH, RESOURCE_PATH)
    receipt = _host_finalize_missing_generation_verification(
        tmp_path,
        generation_verification=fragment_receipt,
        touched_paths=touched,
        required_gates=("project build",),
    )

    assert receipt is not None
    assert receipt["status"] == "DEFERRED_TO_PROJECT_BUILD"
    assert receipt["verification_scope"] == "project"
    assert receipt["touched_path_count"] == 2
    assert receipt["generation_time_receipt"] == fragment_receipt

    classified = classify_generation_verification(
        source_status="SOURCE_GENERATED",
        receipt=receipt,
        touched_paths=touched,
        required_gates=("project build",),
    )
    assert classified["generation_status"] == "DEFERRED_TO_PROJECT_BUILD"
    assert classified["verifier_tier"] == 1
