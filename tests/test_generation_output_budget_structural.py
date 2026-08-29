from __future__ import annotations

from dataclasses import dataclass, field

from minecraft_mod_ai.generation_output_budget import (
    apply_payload_generation_budget,
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


def test_apply_source_edit_is_reviewed_as_compact_structural_action() -> None:
    assert tools_require_expansive_output((_source_edit_schema(),)) is False


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
    assert bounded["max_tokens"] <= 8192


def test_dynamic_normal_source_edit_gets_full_structural_page() -> None:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "x" * 55_000}],
        "tools": [_source_edit_schema()],
    }

    bounded = apply_payload_generation_budget(payload, config=_DynamicConfig())

    assert bounded["max_tokens"] == 8192


def test_explicit_static_output_ceiling_is_never_raised_by_floor() -> None:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "x" * 100_000}],
        "tools": [_source_edit_schema()],
    }

    bounded = apply_payload_generation_budget(payload, config=_StaticConfig())

    assert bounded["max_tokens"] == 1024
