from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    LlamaCompletionBoundaryError,
)
from minecraft_mod_ai.llama_server_efficiency_contract import install as install_llama_efficiency
from minecraft_mod_ai.model_adapters import GenerationRequest
from minecraft_mod_ai.model_context_budget import (
    bounded_tool_message,
    effective_context_tokens,
    request_message_budget,
    tool_action_token_budget,
)
from minecraft_mod_ai.progress_aware_tool_loop import generate_with_tools


def _qwen35_config() -> SimpleNamespace:
    return SimpleNamespace(
        role="coder",
        adapter="llama_cpp",
        provider="local",
        exclusive_gpu=False,
        max_context=262144,
        max_input_tokens=0,
        max_new_tokens=-1,
        model_id="test/qwen",
        extra={
            "runtime_contract": "qwen",
            "decode_hotpath": "t4_mtp",
            "runtime_context_default": 32768,
        },
    )


def _tool_schema(name: str = "work_status") -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object"},
        },
    }


def test_live_context_not_registry_max_owns_qwen_tool_budget(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    config = _qwen35_config()
    tool = _tool_schema()

    assert effective_context_tokens(config) == 32768
    assert tool_action_token_budget(config) == 8192
    budget = request_message_budget(config, (tool,))
    assert 12 * 1024 <= budget < 48 * 1024


def test_oversized_tool_observation_is_archived_with_mutation_proof(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_ARCHIVE", str(tmp_path / "context"))
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    config = _qwen35_config()
    payload = {
        "ok": True,
        "tool": "apply_source_patch",
        "result": {"blob": "x" * 80_000},
        "_mmm_source_mutation": {
            "tool": "apply_source_patch",
            "status": "APPLIED_BY_HOST_RUNTIME",
        },
    }
    original = {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "apply_source_patch",
        "content": json.dumps(payload),
    }

    bounded = bounded_tool_message(
        original,
        config=config,
        tools=(_tool_schema("apply_source_patch"),),
    )
    decoded = json.loads(str(bounded["content"]))

    assert len(str(bounded["content"])) < len(str(original["content"]))
    assert decoded["_mmm_source_mutation"]["status"] == "APPLIED_BY_HOST_RUNTIME"
    archive = decoded["_mmm_context_compaction"]["raw_observation"]
    assert archive["available"] is True
    assert archive["sha256"].startswith("sha256:")


def test_context_pressure_recovery_keeps_tool_transport_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_ARCHIVE", str(tmp_path / "context"))
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)
    config = _qwen35_config()
    tool = _tool_schema()

    class Router:
        _agent_require_fresh_evidence = False

        @contextmanager
        def _generation_scope(self, _config):
            yield

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise LlamaCompletionBoundaryError(
                    "synthetic context pressure",
                    kind=CONTEXT_PRESSURE,
                    prompt_tokens=22000,
                    completion_tokens=0,
                    max_tokens=8192,
                )
            return SimpleNamespace(tool_calls=(), content="done")

    class Runtime:
        def call(self, stage: str, name: str, arguments: dict[str, object]):
            del stage, name, arguments
            raise AssertionError("no tool should execute in this synthetic turn")

    large_a = "a" * 21_000
    large_b = "b" * 21_000
    messages = (
        {"role": "user", "content": "continue the existing implementation"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old-a",
                    "type": "function",
                    "function": {"name": "work_status", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old-a", "name": "work_status", "content": large_a},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old-b",
                    "type": "function",
                    "function": {"name": "work_status", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old-b", "name": "work_status", "content": large_b},
    )
    request = GenerationRequest(
        messages=messages,
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    adapter = Adapter()

    result = generate_with_tools(
        Router(),
        config=config,
        adapter=adapter,
        request=request,
        runtime=Runtime(),
        stage="planning",
        role="planner",
    )

    assert result == "done"
    assert len(adapter.requests) == 2
    assert adapter.requests[0].tools == (tool,)
    assert adapter.requests[1].tools == (tool,)
    assert adapter.requests[1].tool_validation_schemas == (tool,)
    first_bytes = len(json.dumps(adapter.requests[0].messages).encode())
    second_bytes = len(json.dumps(adapter.requests[1].messages).encode())
    assert second_bytes < first_bytes


def test_installed_llama_efficiency_bounds_negative_tool_decode(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)

    def probe(*args, **kwargs):
        del args, kwargs
        return None

    autotune = SimpleNamespace(
        _mmm_server_efficiency_installed=False,
        _probe_server=probe,
        _resolve_model_path=lambda config: config.model_id,
        _cache_path=lambda: None,
        _server_version=lambda binary: binary,
        _hardware_identity=lambda: "test",
        _load_cached_decision=lambda fingerprint: None,
        _save_decision=lambda decision: None,
        _fingerprint=lambda config, binary, model_path: "old",
        ensure_tuned_server=lambda config, request: "",
        _AUTOTUNE_LOCK=threading.RLock(),
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
    )
    hardware = SimpleNamespace(
        _server_payload=lambda adapter, request: {
            "max_tokens": -1,
            "tools": list(request.tools),
        }
    )
    install_llama_efficiency(autotune, hardware)
    config = _qwen35_config()
    request = GenerationRequest(tools=(_tool_schema(),))

    payload = hardware._server_payload(SimpleNamespace(config=config), request)

    assert payload["max_tokens"] == 8192
    assert payload["cache_prompt"] is True
