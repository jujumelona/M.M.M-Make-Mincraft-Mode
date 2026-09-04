from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.generation_output_budget import (
    GenerationOutputBudgetError,
    apply_payload_generation_budget,
)
from minecraft_mod_ai.prefill_calibration_strictness_contract import install as install_prefill
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


class _FakeResponse:
    status_code = 200

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt

    def json(self):
        return {"prompt": self._prompt}


class _FakeHttpx:
    class TimeoutException(Exception):
        pass

    class Timeout:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []
        self.post = self._post

    def _post(self, url: str, *, json: dict, timeout):
        self.calls.append((url, json))
        return self.response


def test_prefill_calibration_uses_apply_template_without_generation() -> None:
    sentinel = "MMM_ASSISTANT_PREFILL_CALIBRATION_V1"
    httpx = _FakeHttpx(_FakeResponse("prefix" + sentinel + "<assistant-suffix>"))

    def forbidden_completion(*args, **kwargs):
        raise AssertionError("completion inference must not run during template calibration")

    module = SimpleNamespace(
        _PREFILL_CALIBRATION_SENTINEL=sentinel,
        _MAX_PREFILL_TEMPLATE_BYTES=512,
        _DEFAULT_COMPLETION_TIMEOUT_SECONDS=120.0,
        _DEFAULT_HTTPX_POST=object(),
        httpx=httpx,
        _positive_env_float=lambda name, default: default,
        _bounded_response_body=lambda response: "",
        _post_completion=forbidden_completion,
        _assistant_prefill_calibration_payload=lambda original: {
            "model": original.get("model", "model"),
            "messages": [
                {"role": "user", "content": "calibrate"},
                {"role": "assistant", "content": sentinel},
            ],
            "max_tokens": 0,
            "temperature": 0.0,
        },
        _calibrate_assistant_prefill_generation_prompt=lambda server_url, original: "legacy",
    )

    install_prefill(module)
    suffix = module._calibrate_assistant_prefill_generation_prompt(
        "http://127.0.0.1:8080", {"model": "qwen"}
    )

    assert suffix == "<assistant-suffix>"
    assert len(httpx.calls) == 1
    url, request = httpx.calls[0]
    assert url.endswith("/apply-template")
    assert "max_tokens" not in request
    assert "temperature" not in request


def test_prefill_calibration_rejects_ambiguous_sentinel() -> None:
    sentinel = "MMM_ASSISTANT_PREFILL_CALIBRATION_V1"
    httpx = _FakeHttpx(_FakeResponse(sentinel + "x" + sentinel))
    module = SimpleNamespace(
        _PREFILL_CALIBRATION_SENTINEL=sentinel,
        _MAX_PREFILL_TEMPLATE_BYTES=512,
        _DEFAULT_COMPLETION_TIMEOUT_SECONDS=120.0,
        _DEFAULT_HTTPX_POST=object(),
        httpx=httpx,
        _positive_env_float=lambda name, default: default,
        _bounded_response_body=lambda response: "",
        _assistant_prefill_calibration_payload=lambda original: {
            "messages": [{"role": "assistant", "content": sentinel}],
            "max_tokens": 0,
        },
        _calibrate_assistant_prefill_generation_prompt=lambda server_url, original: "legacy",
    )
    install_prefill(module)
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        module._calibrate_assistant_prefill_generation_prompt("http://localhost:8080", {})
