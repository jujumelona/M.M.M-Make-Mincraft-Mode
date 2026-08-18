from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_hardware_policy as hardware
from minecraft_mod_ai.model_adapters.base import GenerationRequest


_TOOL = {
    "type": "function",
    "function": {
        "name": "read_project_file",
        "description": "Read one project file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def _adapter(model_id: str, filename: str, *, role: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            model_id=model_id,
            role=role,
            max_new_tokens=8192,
            extra={"gguf_filename": filename},
        )
    )


def _request(**kwargs) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "Solve the task."},),
        **kwargs,
    )


def test_qwen36_general_thinking_sampling_is_model_specific() -> None:
    q27 = hardware._server_payload(
        _adapter(
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            role="researcher",
        ),
        _request(),
    )
    q35 = hardware._server_payload(
        _adapter(
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            role="planner",
        ),
        _request(),
    )

    assert q27["temperature"] == 1.0
    assert q27["top_p"] == 0.95
    assert q27["top_k"] == 20
    assert q27["presence_penalty"] == 0.0
    assert q27["chat_template_kwargs"] == {"enable_thinking": True}

    assert q35["temperature"] == 1.0
    assert q35["top_p"] == 0.95
    assert q35["top_k"] == 20
    assert q35["presence_penalty"] == 1.5
    assert q35["chat_template_kwargs"] == {"enable_thinking": True}


def test_qwen36_coder_uses_precise_coding_sampling() -> None:
    payload = hardware._server_payload(
        _adapter(
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            role="coder_safe",
        ),
        _request(),
    )

    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 0.0
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_qwen36_auto_tool_agent_preserves_thinking_with_role_sampling() -> None:
    payload = hardware._server_payload(
        _adapter(
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            role="researcher",
        ),
        _request(tools=(_TOOL,), tool_choice="auto"),
    )

    assert payload["temperature"] == 1.0
    assert payload["presence_penalty"] == 0.0
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert "reasoning_effort" not in payload


def test_qwen36_json_fill_uses_non_thinking_profile() -> None:
    payload = hardware._server_payload(
        _adapter(
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            role="planner",
        ),
        _request(response_format="json"),
    )

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 1.5
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_qwen36_forced_tool_remains_deterministic_non_thinking() -> None:
    payload = hardware._server_payload(
        _adapter(
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            role="researcher",
        ),
        _request(
            tools=(_TOOL,),
            tool_choice={
                "type": "function",
                "function": {"name": "read_project_file"},
            },
        ),
    )

    assert payload["temperature"] == 0.0
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
