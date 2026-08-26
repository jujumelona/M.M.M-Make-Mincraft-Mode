from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import qwen_agent_family_contract as qwen
from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter

_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_source_edit",
        "description": "Apply one semantic source edit.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["operation", "path"],
        },
    },
}


class _AliveProcess:
    def poll(self):
        return None


def _config():
    return SimpleNamespace(
        model_id="vendor/qwen-runtime",
        role="coder",
        max_context=32768,
        max_new_tokens=-1,
        extra={
            "runtime_contract": "qwen",
            "qwen_family": "qwen3.5",
            "qwen_tool_markup": "qwen3_coder_xml",
            "qwen_action_thinking_control": "enable_thinking_false",
            "qwen_preserve_thinking": False,
            "qwen_reasoning_effort": False,
            "qwen_assistant_prefill": True,
            "agent_thinking": True,
        },
    )


def _request(*, tools=(_TOOL,)):
    return GenerationRequest(
        messages=({"role": "user", "content": "Implement the requested edit."},),
        tools=tuple(tools),
        tool_choice="required" if tools else None,
        parallel_tool_calls=False,
    )


def _reset_qwen_phase_state():
    qwen._TOOL_SAFE_RUNTIME_ACTIVE = False
    qwen._TOOL_SAFE_RUNTIME_KEY = None


def test_installed_qwen_wrapper_owns_tool_safe_speculation_phase() -> None:
    assert getattr(LlamaCppAdapter.generate_turn, "_mmm_qwen_tool_safe_speculation", False)


def test_tool_phase_switches_mtp_to_non_spec_once(monkeypatch) -> None:
    _reset_qwen_phase_state()
    process = _AliveProcess()
    launched = []
    stopped = []

    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", process)
    monkeypatch.setattr(autotune, "_MANAGED_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(autotune, "_MANAGED_KEY", "fingerprint")
    monkeypatch.setattr(autotune, "_ATTEMPTED_KEYS", {"fingerprint"})
    monkeypatch.setattr(
        autotune,
        "_MMM_LLAMA_RUNTIME_RECEIPT",
        {
            "spec_type": "draft-mtp",
            "draft_n_max": 3,
            "ubatch": 512,
            "slots": 1,
            "cache_reuse": 64,
        },
        raising=False,
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_SPEC_TYPE", "draft-mtp")
    monkeypatch.setattr(autotune, "_server_binary", lambda: "/tmp/llama-server")
    monkeypatch.setattr(autotune, "_resolve_model_path", lambda config: "/tmp/model.gguf")
    monkeypatch.setattr(autotune, "_stop_server", lambda value: stopped.append(value))

    def launch(binary, model_path, config, selected):
        launched.append(selected)
        autotune._MANAGED_PROCESS = _AliveProcess()
        autotune._MANAGED_URL = "http://127.0.0.1:8911/v1"
        autotune._MMM_LLAMA_RUNTIME_RECEIPT = {
            "spec_type": selected.spec_type,
            "draft_n_max": selected.draft_n_max,
            "ubatch": selected.ubatch,
            "slots": selected.parallel,
            "cache_reuse": selected.cache_reuse,
        }
        os.environ["LLAMA_SERVER_URL"] = autotune._MANAGED_URL
        os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = selected.spec_type
        return autotune._MANAGED_URL

    monkeypatch.setattr(autotune, "_launch_selected", launch)

    qwen._ensure_tool_safe_runtime(_config(), _request())

    assert len(stopped) == 1
    assert len(launched) == 1
    selected = launched[0]
    assert selected.spec_type == "none"
    assert selected.draft_n_max == 0
    assert selected.ubatch == 512
    assert selected.parallel == 1
    assert selected.cache_reuse == 64
    assert autotune._MANAGED_KEY == "fingerprint"
    assert qwen._TOOL_SAFE_RUNTIME_ACTIVE is True

    # A consecutive tool/action round reuses the already-running non-spec server.
    qwen._ensure_tool_safe_runtime(_config(), _request())
    assert len(stopped) == 1
    assert len(launched) == 1

    _reset_qwen_phase_state()


def test_no_tool_turn_restores_cached_autotuned_runtime(monkeypatch) -> None:
    _reset_qwen_phase_state()
    qwen._TOOL_SAFE_RUNTIME_ACTIVE = True
    qwen._TOOL_SAFE_RUNTIME_KEY = "fingerprint"
    tool_process = _AliveProcess()
    stopped = []
    restored = []

    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", tool_process)
    monkeypatch.setattr(autotune, "_MANAGED_URL", "http://127.0.0.1:8911/v1")
    monkeypatch.setattr(autotune, "_MANAGED_KEY", "fingerprint")
    monkeypatch.setattr(autotune, "_ATTEMPTED_KEYS", {"fingerprint"})
    monkeypatch.setattr(
        autotune,
        "_MMM_LLAMA_RUNTIME_RECEIPT",
        {"spec_type": "none", "ubatch": 512, "slots": 1, "cache_reuse": 64},
        raising=False,
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8911/v1")
    monkeypatch.setattr(autotune, "_stop_server", lambda value: stopped.append(value))

    def ensure(config, request):
        restored.append(request)
        assert "fingerprint" not in autotune._ATTEMPTED_KEYS
        autotune._MANAGED_PROCESS = _AliveProcess()
        autotune._MANAGED_URL = "http://127.0.0.1:8910/v1"
        autotune._MANAGED_KEY = "fingerprint"
        autotune._ATTEMPTED_KEYS.add("fingerprint")
        autotune._MMM_LLAMA_RUNTIME_RECEIPT = {
            "spec_type": "draft-mtp",
            "draft_n_max": 3,
        }
        os.environ["LLAMA_SERVER_URL"] = autotune._MANAGED_URL
        return autotune._MANAGED_URL

    monkeypatch.setattr(autotune, "ensure_tuned_server", ensure)

    final_request = _request(tools=())
    qwen._restore_tool_runtime(_config(), final_request)

    assert stopped == [tool_process]
    assert restored == [final_request]
    assert qwen._TOOL_SAFE_RUNTIME_ACTIVE is False
    assert qwen._TOOL_SAFE_RUNTIME_KEY is None
    assert autotune._MMM_LLAMA_RUNTIME_RECEIPT["spec_type"] == "draft-mtp"

    _reset_qwen_phase_state()
