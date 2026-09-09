from __future__ import annotations

from dataclasses import dataclass, field

from minecraft_mod_ai.generation_output_budget import (
    apply_payload_generation_budget,
    generation_output_token_budget,
    tools_require_expansive_output,
)


def _source_edit_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one bounded structural source edit",
            "parameters": {"type": "object", "properties": {}},
        },
    }


@dataclass
class _DynamicConfig:
    adapter: str = "test"
    max_context: int = 32768
    max_input_tokens: int = 0
    max_new_tokens: int = 8192
    extra: dict = field(default_factory=lambda: {"dynamic_output_budget": True})


@dataclass
class _StaticConfig:
    adapter: str = "test"
    max_context: int = 32768
    max_input_tokens: int = 0
    max_new_tokens: int = 1024
    extra: dict = field(default_factory=dict)


def test_apply_source_edit_keeps_expansive_effect_classification() -> None:
    assert tools_require_expansive_output((_source_edit_schema(),)) is True


def test_source_mutation_is_not_capped_by_compact_tool_action_budget() -> None:
    budget = generation_output_token_budget(
        _DynamicConfig(),
        input_tokens=1_000,
        tools=(_source_edit_schema(),),
    )

    assert budget > 8192


def test_large_prompt_does_not_starve_forced_source_edit_output() -> None:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "x" * 100_000}],
        "tools": [_source_edit_schema()],
        "tool_choice": {
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
        "parallel_tool_calls": False,
        "max_tokens": 151,
    }

    bounded = apply_payload_generation_budget(payload, config=_DynamicConfig())

    assert bounded["max_tokens"] >= 4096


def test_dynamic_normal_source_edit_uses_remaining_context_not_compact_ceiling() -> None:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "x" * 55_000}],
        "tools": [_source_edit_schema()],
    }

    bounded = apply_payload_generation_budget(payload, config=_DynamicConfig())

    assert bounded["max_tokens"] > 8192


def test_explicit_static_output_ceiling_is_never_raised_by_floor() -> None:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "x" * 100_000}],
        "tools": [_source_edit_schema()],
    }

    bounded = apply_payload_generation_budget(payload, config=_StaticConfig())

    assert bounded["max_tokens"] == 1024


def test_host_validated_json_page_retains_budget_without_wire_grammar() -> None:
    # Qwen3.5 removes wire JSON constraints and native tools before this boundary.
    payload = {"messages": [{"role": "user", "content": "x" * 112_616}], "max_tokens": 1}
    bounded = apply_payload_generation_budget(
        payload, config=_DynamicConfig(), structured_output=True,
    )
    assert bounded["max_tokens"] >= 4096
    assert "tools" not in bounded
    assert "response_format" not in bounded


def test_static_json_page_limit_is_respected() -> None:
    bounded = apply_payload_generation_budget(
        {"messages": [{"role": "user", "content": "x" * 112_616}]},
        config=_StaticConfig(), structured_output=True,
    )
    assert bounded["max_tokens"] == 1024


def test_llama_budget_uses_request_contract_when_qwen_removes_grammar() -> None:
    from types import SimpleNamespace
    from minecraft_mod_ai.llama_generation_budget import install
    hardware = SimpleNamespace(_server_payload=lambda adapter, request: {
        "messages": [{"role": "user", "content": "x" * 112_616}],
        "max_tokens": 1,
    })
    install(hardware)
    request = SimpleNamespace(tools=(), response_format="json", response_schema={
        "type": "object", "properties": {"operation": {"type": "string"}},
    })
    payload = hardware._server_payload(SimpleNamespace(config=_DynamicConfig()), request)
    assert payload["max_tokens"] >= 4096
    assert "response_format" not in payload
