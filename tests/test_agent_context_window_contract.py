from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    LlamaCompletionBoundaryError,
    completion_boundary_error,
    completion_boundary_kind,
    context_recovery_exhausted,
    mark_context_recovery_exhausted,
)
from minecraft_mod_ai.llama_generation_budget import (
    apply_generation_budget,
    apply_structured_output_constraint,
)
from minecraft_mod_ai.llama_stream_efficiency_contract import (
    _bounded_timeout,
    _stream_idle_timeout_seconds,
    _tool_idle_timeout_seconds,
)
from minecraft_mod_ai.model_adapters import AdapterConfig
from minecraft_mod_ai.model_context_budget import (
    effective_context_tokens,
    emergency_fit_messages,
    fit_messages_to_context,
)
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.source_mutation_contract import mutation_history_applied


def _config() -> AdapterConfig:
    return AdapterConfig(
        role="coder",
        adapter="llama_cpp",
        model_id="test/qwen",
        max_context=262_144,
        max_new_tokens=8_192,
        extra={"runtime_context_default": 32_768},
    )


def _tool_schema(name: str = "search_code_rag") -> tuple[dict[str, object], ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    )


def _assistant_call(call_id: str, name: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool_result(call_id: str, name: str, payload: object) -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, sort_keys=True),
    }


def test_live_llama_context_wins_over_registry_advertised_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    assert effective_context_tokens(_config()) == 32_768


def test_default_transport_liveness_is_finite_and_below_legacy_stall(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_STREAM_IDLE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_IDLE_TIMEOUT_SECONDS", raising=False)
    assert 0 < _stream_idle_timeout_seconds() <= 120
    assert 0 < _tool_idle_timeout_seconds() <= 120

    unbounded = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    bounded = _bounded_timeout(unbounded, read_seconds=_tool_idle_timeout_seconds())
    assert bounded.read is not None
    assert 0 < float(bounded.read) <= 120

    stricter = httpx.Timeout(connect=30.0, read=17.0, write=30.0, pool=30.0)
    assert _bounded_timeout(stricter, read_seconds=120.0).read == 17.0


def test_structured_action_page_uses_json_schema_and_finite_budget() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    request = SimpleNamespace(response_format="json", response_schema=schema)
    constrained = apply_structured_output_constraint(
        {"model": "local", "max_tokens": -1},
        request=request,
    )
    response_format = constrained["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema

    bounded = apply_generation_budget(constrained, config=_config())
    assert 0 < bounded["max_tokens"] <= 8_192


def test_exhausted_context_recovery_preserves_original_boundary_identity() -> None:
    boundary = LlamaCompletionBoundaryError(
        "context full",
        kind=CONTEXT_PRESSURE,
        prompt_tokens=17_000,
        completion_tokens=15_000,
        max_tokens=8_192,
    )
    assert completion_boundary_kind(boundary) == CONTEXT_PRESSURE
    mark_context_recovery_exhausted(boundary)
    assert context_recovery_exhausted(boundary)
    assert completion_boundary_error(boundary) is boundary
    assert completion_boundary_kind(boundary) == ""


def test_compaction_preserves_tool_pairs_latest_mutation_and_archives_old_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", str(12 * 1024))
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_ARCHIVE", str(tmp_path / "context"))

    messages: list[dict[str, object]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "implement the requested source change"},
        _assistant_call("old-0", "search_code_rag"),
        _tool_result(
            "old-0",
            "search_code_rag",
            {"ok": True, "tool": "search_code_rag", "result": {"blob": "x" * 24_000}},
        ),
        _assistant_call("mutation", "apply_source_patch"),
        _tool_result(
            "mutation",
            "apply_source_patch",
            {
                "ok": True,
                "tool": "apply_source_patch",
                "_mmm_source_mutation": {
                    "tool": "apply_source_patch",
                    "status": "APPLIED_BY_HOST_RUNTIME",
                },
                "result": {"status": "applied"},
            },
        ),
    ]
    for index in range(3):
        call_id = f"recent-{index}"
        messages.extend(
            [
                _assistant_call(call_id, "search_code_rag"),
                _tool_result(
                    call_id,
                    "search_code_rag",
                    {
                        "ok": True,
                        "tool": "search_code_rag",
                        "result": {"value": f"recent-value-{index}"},
                    },
                ),
            ]
        )

    fitted = fit_messages_to_context(
        messages,
        config=_config(),
        tools=_tool_schema(),
    )
    emergency = emergency_fit_messages(fitted, budget_bytes=12 * 1024)

    assert mutation_history_applied(emergency)
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "recent-2"
        and "recent-value-2" in str(message.get("content", ""))
        for message in emergency
    )

    calls = {
        str(call.get("id"))
        for message in emergency
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", ())
        if isinstance(call, dict)
    }
    results = {
        str(message.get("tool_call_id"))
        for message in emergency
        if message.get("role") == "tool"
    }
    assert calls == results
    assert any(
        message.get("role") == "system"
        and "HOST COMPACTED VERIFIED CONTEXT" in str(message.get("content", ""))
        for message in emergency
    )
    assert list((tmp_path / "context").glob("*.json"))


def test_model_router_has_one_direct_tool_loop_owner() -> None:
    source = inspect.getsource(ModelRouter._generate_with_tools)
    assert "progress_aware_tool_loop" in source
    assert "generate_with_tools(" in source
    assert "while True" not in source


def test_runtime_sources_have_no_legacy_unbounded_llama_transport() -> None:
    root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    adapter = (root / "model_adapters" / "llama_cpp_adapter.py").read_text()
    hardware = (root / "llama_server_hardware_policy.py").read_text()
    bootstrap = (root / "runtime_bootstrap.py").read_text()

    assert "_DEFAULT_COMPLETION_TIMEOUT_SECONDS = 600.0" not in adapter
    assert "read=None" not in hardware
    assert "install_forced_tool_execution" in bootstrap


def test_superseded_context_and_coder_route_modules_are_deleted() -> None:
    root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    assert not (root / "small_model_context_compaction.py").exists()
    assert not (root / "coder_tool_route_integrity_contract.py").exists()
    assert not (root / "causal_stale_tool_recovery_contract.py").exists()
