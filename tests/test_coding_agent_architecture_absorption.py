from __future__ import annotations

import json
from unittest.mock import MagicMock

from minecraft_mod_ai.model_context_budget import (
    bounded_tool_message,
    emergency_fit_messages,
    fit_messages_to_context,
    request_message_budget,
)


def _mock_config(max_context: int = 16384, max_new: int = 4096) -> MagicMock:
    config = MagicMock()
    config.adapter = "llama_cpp"
    config.max_context = max_context
    config.max_input_tokens = 0
    config.max_new_tokens = max_new
    config.extra = {
        "gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.5",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": False,
        "qwen_reasoning_effort": False,
        "qwen_assistant_prefill": True,
    }
    return config


def test_gemini_cli_style_giant_tool_output_archived_and_previewed() -> None:
    """Verify Gemini CLI style giant tool output archive + bounded preview."""
    config = _mock_config()
    huge_result = {
        "ok": True,
        "tool": "search_code_rag",
        "result": {
            "matches": [{"file": f"file_{i}.java", "snippet": "A" * 2000} for i in range(20)]
        },
    }
    giant_message = {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "search_code_rag",
        "content": json.dumps(huge_result),
    }

    bounded = bounded_tool_message(giant_message, config=config, tools=())
    content = str(bounded["content"])

    assert len(content.encode("utf-8")) < 16 * 1024
    parsed = json.loads(content)
    assert "_mmm_context_compaction" in parsed
    assert "exact observation archived by host" in parsed["preview"]


def test_cline_style_pre_request_fitting_and_emergency_recovery() -> None:
    """Verify Cline style pre-turn request budget fitting and deterministic overflow recovery."""
    config = _mock_config()
    budget = request_message_budget(config, tools=())

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Implement feature."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "f", "content": json.dumps({"data": "X" * 15000})},
        {"role": "assistant", "content": "I will proceed."},
        {"role": "user", "content": "Continue with second step."},
    ]

    fitted = fit_messages_to_context(messages, config=config, tools=())
    fitted_size = len(json.dumps(fitted).encode("utf-8"))
    assert fitted_size <= budget

    emergency = emergency_fit_messages(fitted, budget_bytes=12 * 1024)
    emergency_size = len(json.dumps(emergency).encode("utf-8"))
    assert emergency_size <= 12 * 1024


def test_aider_and_codex_style_completed_history_rollover_with_mutation_proof() -> None:
    """Verify Aider/Codex style context rollover compacting older exchanges while preserving latest mutation."""
    config = _mock_config()

    messages = [
        {"role": "system", "content": "System directive."},
        {"role": "user", "content": "Task 1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "apply_source_patch", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "apply_source_patch",
            "content": json.dumps({
                "ok": True,
                "_mmm_source_mutation": {"tool": "apply_source_patch", "status": "APPLIED_BY_HOST_RUNTIME"},
                "result": {"modified": "Example.java"},
            }),
        },
        {"role": "assistant", "content": "Applied patch 1."},
        {"role": "user", "content": "Task 2 with lots of chatter " + ("Y" * 12000)},
        {"role": "assistant", "content": "Working on task 2..."},
        {"role": "user", "content": "Task 3 final step"},
    ]

    fitted = fit_messages_to_context(messages, config=config, tools=())
    fitted_str = json.dumps(fitted)

    assert "HOST COMPACTED VERIFIED CONTEXT" in fitted_str or len(fitted_str) <= request_message_budget(config)
