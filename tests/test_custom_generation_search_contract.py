from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_generation_search_contract as custom_search
from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator


@dataclass
class _Module:
    kind: str
    config: dict = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()


class _Router:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return "ok"


def test_custom_search_width_is_risk_adaptive_when_native_slots_exist(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    assert custom_search._width(_Module(kind="item")) == 1
    risky = _Module(
        kind="custom_java",
        config={"networking": "server authoritative", "persistence": True},
        depends_on=("state", "protocol"),
    )
    assert custom_search._width(risky) == 2


def test_custom_search_auto_never_serializes_candidates_on_one_slot(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    risky = _Module(
        kind="custom_java",
        config={"networking": "server authoritative", "persistence": True},
        depends_on=("state", "protocol"),
    )
    assert custom_search._width(risky) == 1


def test_custom_search_candidate_threads_inherit_isolated_contextvars() -> None:
    marker: ContextVar[str] = ContextVar("custom_search_marker", default="missing")
    token = marker.set("parent")

    def read_then_mutate(value: str) -> tuple[str, str]:
        inherited = marker.get()
        marker.set(value)
        return inherited, marker.get()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                custom_search._submit_with_copied_context(pool, read_then_mutate, "child-a"),
                custom_search._submit_with_copied_context(pool, read_then_mutate, "child-b"),
            ]
            results = [future.result() for future in futures]

        assert results == [("parent", "child-a"), ("parent", "child-b")]
        assert marker.get() == "parent"
    finally:
        marker.reset(token)


def test_strategy_router_only_augments_coder_role() -> None:
    base = _Router()
    router = custom_search._StrategyRouter(
        base,
        strategy="api_contract_first",
        candidate_index=1,
        count=2,
    )
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "task"},
    ]
    router.generate_text("coder", messages, response_format="json")
    assert len(base.calls[0][1]) == 3
    assert "api_contract_first" in base.calls[0][1][1]["content"]

    router.generate_text("planner", messages, response_format="json")
    assert base.calls[1][1] == messages


def test_custom_generation_public_target_overrides_are_not_exposed() -> None:
    signature = inspect.signature(CustomModuleGenerator.generate)
    assert "minecraft_version" not in signature.parameters
    assert "loader" not in signature.parameters
    assert "mappings" not in signature.parameters


def _generation_receipt(status: str) -> dict:
    return {
        "schema_version": "mmm/generation-verification-v1",
        "status": status,
        "authority": "generation_tool_loop",
        "validation_status": (
            "PASS" if status == "PASS" else "DEFERRED"
        ),
        "termination_reason": (
            "VERIFICATION_PASSED"
            if status == "PASS"
            else "VERIFICATION_DEFERRED_TO_TARGET_COMPILE"
        ),
        "verifier_tool": "target_compile",
        "target_path": "src/main/java/demo/Test.java",
        "compile_backed_java": True,
        "downstream_required_gate": (
            "target_compile"
            if status == "DEFERRED_TO_TARGET_COMPILE"
            else None
        ),
    }


def test_candidate_verifier_tiers_exact_host_terminal_state() -> None:
    assert custom_search._generation_verifier_tier(
        {
            "generation_status": "PASS",
            "target_compile_required": True,
        }
    ) == 2
    assert custom_search._generation_verifier_tier(
        {
            "generation_status": "DEFERRED_TO_TARGET_COMPILE",
            "target_compile_required": True,
            "downstream_required_gate": "target_compile",
        }
    ) == 1
    assert custom_search._generation_verifier_tier(
        {
            "generation_status": "DEFERRED_TO_TARGET_COMPILE",
            "target_compile_required": False,
            "downstream_required_gate": "target_compile",
        }
    ) == 0
    assert custom_search._generation_verifier_tier(
        {"generation_status": "FAIL"}
    ) == 0


def test_candidate_selection_preserves_nonselectable_verification_evidence() -> None:
    evaluations = [
        (
            2_000_001.0,
            0,
            None,
            {},
            {
                "generation_status": "PASS",
                "verification_authority": "generation_tool_loop",
                "verifier_tier": 2,
            },
        ),
        (
            0.0,
            1,
            None,
            {},
            {
                "generation_status": "FAIL",
                "verification_authority": "generation_tool_loop",
                "verifier_tier": 0,
            },
        ),
    ]

    selectable = custom_search._require_selectable_evaluations(
        evaluations,
        verifier_index=4,
    )

    assert len(evaluations) == 2
    assert selectable == [evaluations[0]]


def test_candidate_search_fails_closed_without_admissible_terminal_state() -> None:
    evaluations = [
        (
            0.0,
            0,
            None,
            {},
            {
                "generation_status": "FAIL",
                "verification_authority": "generation_tool_loop",
            },
        ),
        (
            0.0,
            1,
            None,
            {},
            {
                "generation_status": "MISSING",
                "verification_authority": "unknown",
            },
        ),
    ]

    with pytest.raises(RuntimeError, match="no candidate with trustworthy verification"):
        custom_search._require_selectable_evaluations(
            evaluations,
            verifier_index=4,
        )


def test_candidate_verifier_uses_exact_generation_pass_receipt(tmp_path) -> None:
    score, verifier = custom_search._verify_candidate(
        tmp_path,
        {
            "status": "SOURCE_GENERATED",
            "generation_verification": _generation_receipt("PASS"),
            "touched_paths": ["src/main/java/demo/Test.java"],
            "operation_count": 1,
            "runtime_tests": [],
            "required_gates": ["target_compile"],
        },
    )

    assert verifier["generation_status"] == "PASS"
    assert verifier["terminal_verification_status"] == "PASS"
    assert verifier["verification_authority"] == "generation_tool_loop"
    assert verifier["target_compile_required"] is True
    assert verifier["verifier_tier"] == 2
    assert score >= 2_000_000.0
    assert verifier["jdt_status"] == "NOT_RUN"


def test_candidate_verifier_preserves_deferred_target_compile_state(tmp_path) -> None:
    score, verifier = custom_search._verify_candidate(
        tmp_path,
        {
            "status": "SOURCE_GENERATED",
            "generation_verification": _generation_receipt(
                "DEFERRED_TO_TARGET_COMPILE"
            ),
            "touched_paths": ["src/main/java/demo/Test.java"],
            "operation_count": 1,
            "runtime_tests": [],
            "required_gates": ["target_compile"],
        },
    )

    assert verifier["generation_status"] == "DEFERRED_TO_TARGET_COMPILE"
    assert verifier["verifier_tier"] == 1
    assert verifier["downstream_required_gate"] == "target_compile"
    assert score >= 1_000_000.0
    assert score < 2_000_000.0
    assert custom_search._candidate_verifier_selectable(verifier)


def test_candidate_verifier_rejects_source_generated_without_terminal_receipt(
    tmp_path,
) -> None:
    _score, verifier = custom_search._verify_candidate(
        tmp_path,
        {
            "status": "SOURCE_GENERATED",
            "touched_paths": ["src/main/java/demo/Test.java"],
            "operation_count": 1,
            "runtime_tests": [],
            "required_gates": ["target_compile"],
        },
    )

    assert verifier["generation_status"] == "FAIL"
    assert verifier["terminal_verification_status"] == "MISSING"
    assert verifier["verifier_tier"] == 0
    assert not custom_search._candidate_verifier_selectable(verifier)


def test_candidate_verifier_rejects_malformed_generation_result(tmp_path) -> None:
    _score, verifier = custom_search._verify_candidate(
        tmp_path,
        {
            "status": "BROKEN",
            "generation_verification": _generation_receipt("PASS"),
            "touched_paths": ["src/main/java/demo/Test.java"],
            "operation_count": 1,
            "runtime_tests": [],
            "required_gates": ["target_compile"],
        },
    )

    assert verifier["generation_status"] == "FAIL"
    assert verifier["source_status"] == "BROKEN"
    assert verifier["verifier_tier"] == 0
    assert not custom_search._candidate_verifier_selectable(verifier)


def test_generation_terminal_receipt_is_context_local_and_structured() -> None:
    state = SimpleNamespace(
        mutation_context=SimpleNamespace(
            target_path="src/main/java/demo/Test.java"
        ),
        latest_verifier_tool="target_compile",
        validation_status="PASS",
        termination_reason="VERIFICATION_PASSED",
    )
    tool_loop.clear_generation_verification_receipt()
    tool_loop._record_terminal_generation_verification(
        state,
        terminal_status="PASS",
        compile_backed_java=True,
    )

    receipt = tool_loop.current_generation_verification_receipt()
    assert receipt == {
        "schema_version": "mmm/generation-verification-v1",
        "status": "PASS",
        "authority": "generation_tool_loop",
        "validation_status": "PASS",
        "termination_reason": "VERIFICATION_PASSED",
        "verifier_tool": "target_compile",
        "target_path": "src/main/java/demo/Test.java",
        "compile_backed_java": True,
        "downstream_required_gate": None,
    }
    receipt["status"] = "CORRUPTED"
    assert tool_loop.current_generation_verification_receipt()["status"] == "PASS"


def test_candidate_verifier_has_no_candidate_local_jdt_dependency() -> None:
    source = inspect.getsource(custom_search._verify_candidate)
    assert "JavaLanguageService" not in source
    assert "diagnostic_errors" not in source
    assert "timeout_seconds=60" not in source


def test_target_values_fail_closed_without_complete_host_target() -> None:
    with pytest.raises(ValueError, match="mappings"):
        custom_search._target_values(
            {
                "minecraft_version": "1.20.1",
                "loader": "fabric",
            }
        )
