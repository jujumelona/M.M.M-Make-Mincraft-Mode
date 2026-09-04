from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.generation_output_budget import (
    GenerationOutputBudgetError,
    apply_payload_generation_budget,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    _fixed_point_tool_results,
    _verification_outcome,
)


def test_verifier_transport_failure_is_unavailable_not_source_fail() -> None:
    payload = {"ok": False, "error": "JDT LS did not publish diagnostics"}
    assert _verification_outcome("java_diagnostics", payload) == "UNAVAILABLE"


def test_jdt_unavailable_receipt_stays_unavailable() -> None:
    payload = {
        "ok": True,
        "result": {
            "status": "UNAVAILABLE",
            "error": "language server unavailable",
            "diagnostics": {},
        },
    }
    assert _verification_outcome("java_diagnostics", payload) == "UNAVAILABLE"


def test_jdt_source_diagnostic_is_a_trustworthy_fail() -> None:
    payload = {
        "ok": True,
        "result": {
            "status": "PASS",
            "diagnostics": {
                "file:///Example.java": [
                    {"severity": 1, "message": "cannot resolve symbol"}
                ]
            },
        },
    }
    assert _verification_outcome("java_diagnostics", payload) == "FAIL"


def test_clean_jdt_receipt_is_pass() -> None:
    payload = {
        "ok": True,
        "result": {"status": "PASS", "diagnostics": {}},
    }
    assert _verification_outcome("java_diagnostics", payload) == "PASS"


def test_nonzero_build_exit_is_source_validation_fail() -> None:
    payload = {"ok": True, "result": {"exit_code": 1}}
    assert _verification_outcome("run_gradle_build", payload) == "FAIL"


def test_fixed_point_ignores_volatile_verifier_exception_text() -> None:
    call = SimpleNamespace(name="java_diagnostics")
    first = _fixed_point_tool_results(
        [(call, {"ok": False, "error": "timeout after 89.9s"})]
    )
    second = _fixed_point_tool_results(
        [(call, {"ok": False, "error": "timeout after 90.1s"})]
    )
    assert first == second == [
        {"name": "java_diagnostics", "verification_outcome": "UNAVAILABLE"}
    ]


def test_starved_structural_decode_fails_before_payload_is_sent() -> None:
    config = SimpleNamespace(
        adapter="",
        extra={},
        max_new_tokens=32,
        max_context=0,
    )
    payload = {
        "messages": [{"role": "user", "content": "edit the source"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "max_tokens": 1,
    }
    with pytest.raises(GenerationOutputBudgetError, match="STRUCTURAL_OUTPUT_BUDGET_UNVIABLE"):
        apply_payload_generation_budget(payload, config=config)
