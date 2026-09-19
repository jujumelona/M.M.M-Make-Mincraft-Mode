from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field

import pytest

from minecraft_mod_ai import custom_generation_search_contract as custom_search
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


def test_candidate_verifier_selects_host_generation_pass_only() -> None:
    assert custom_search._candidate_verifier_selectable(
        {
            "generation_status": "PASS",
            "verification_authority": "generation_tool_loop",
        }
    )
    assert not custom_search._candidate_verifier_selectable(
        {
            "generation_status": "FAIL",
            "verification_authority": "generation_tool_loop",
        }
    )
    assert not custom_search._candidate_verifier_selectable(
        {
            "jdt_status": "AVAILABLE",
            "jdt_error_count": 0,
        }
    )


def test_candidate_selection_preserves_nonselectable_verification_evidence() -> None:
    evaluations = [
        (
            1.0,
            0,
            None,
            {},
            {
                "generation_status": "PASS",
                "verification_authority": "generation_tool_loop",
                "jdt_status": "NOT_RUN",
            },
        ),
        (
            -2.85,
            1,
            None,
            {},
            {
                "generation_status": "FAIL",
                "verification_authority": "generation_tool_loop",
                "jdt_status": "NOT_RUN",
            },
        ),
    ]

    selectable = custom_search._require_selectable_evaluations(
        evaluations,
        verifier_index=4,
    )

    assert len(evaluations) == 2
    assert selectable == [evaluations[0]]


def test_candidate_search_fails_closed_without_host_generation_pass() -> None:
    evaluations = [
        (
            -2.85,
            0,
            None,
            {},
            {
                "generation_status": "FAIL",
                "verification_authority": "generation_tool_loop",
            },
        ),
        (
            -118.0,
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


def test_candidate_verifier_reuses_generation_gate_without_starting_jdt(tmp_path) -> None:
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

    assert verifier["generation_status"] == "PASS"
    assert verifier["verification_authority"] == "generation_tool_loop"
    assert verifier["target_compile_required"] is True
    assert verifier["java_path_count"] == 1
    assert verifier["jdt_status"] == "NOT_RUN"
    assert verifier["jdt_error_count"] is None


def test_candidate_verifier_rejects_malformed_generation_receipt(tmp_path) -> None:
    _score, verifier = custom_search._verify_candidate(
        tmp_path,
        {
            "status": "BROKEN",
            "touched_paths": ["src/main/java/demo/Test.java"],
            "operation_count": 1,
            "runtime_tests": [],
            "required_gates": ["target_compile"],
        },
    )

    assert verifier["generation_status"] == "FAIL"
    assert verifier["source_status"] == "BROKEN"
    assert not custom_search._candidate_verifier_selectable(verifier)


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
