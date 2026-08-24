from __future__ import annotations

import json

from minecraft_mod_ai.coder_tool_route_integrity_contract import _require_mutation_surface


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _messages() -> tuple[dict, ...]:
    return (
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "task": "Implement the approved module source.",
                }
            ),
        },
    )


def test_retired_route_preflight_does_not_reject_selector_owned_surface() -> None:
    _require_mutation_surface(
        (_schema("apply_source_patch"),),
        messages=_messages(),
        stage="generation",
        role="coder",
    )


def test_complete_source_mutation_route_is_accepted() -> None:
    _require_mutation_surface(
        (
            _schema("inspect_existing_mod"),
            _schema("search_code_rag"),
            _schema("apply_source_patch"),
        ),
        messages=_messages(),
        stage="generation",
        role="coder",
    )
