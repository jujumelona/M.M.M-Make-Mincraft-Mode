from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_hardware_policy as hardware
from minecraft_mod_ai.model_adapters.base import GenerationRequest

_TOOL = {
    "type": "function",
    "function": {
        "name": "read_project_file",
        "description": "Read one project file.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _payload(model_id: str) -> dict[str, object]:
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            model_id=model_id,
            role="coder_safe",
            max_new_tokens=8192,
        )
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "Implement the feature."},),
        tools=(_TOOL,),
        tool_choice="auto",
    )
    return hardware._server_payload(adapter, request)


def test_qwen36_27b_and_35b_a3b_share_precise_agent_contract() -> None:
    expected = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }

    for model_id in (
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    ):
        payload = _payload(model_id)
        assert payload["chat_template_kwargs"] == {
            "enable_thinking": True,
            "preserve_thinking": True,
        }
        assert "reasoning_effort" not in payload
        assert "repetition_penalty" not in payload
        for key, value in expected.items():
            assert payload[key] == value
