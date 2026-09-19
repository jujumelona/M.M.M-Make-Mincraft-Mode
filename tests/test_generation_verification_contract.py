from __future__ import annotations

import inspect

from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
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
