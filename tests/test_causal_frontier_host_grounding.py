from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.causal_frontier_adapter import (
    CausalFrontierAdapter,
    remember_authorized_tools,
)
from minecraft_mod_ai.model_adapters import GenerationRequest


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _messages(*, valid_grounding: bool) -> tuple[dict, ...]:
    receipt = {
        "project_sha256": "sha256:" + "1" * 64,
        "observations_sha256": "sha256:" + "2" * 64,
    }
    if not valid_grounding:
        receipt.pop("observations_sha256")
    return (
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "task": "Implement the approved source change.",
                    "host_grounding": {
                        "schema_version": "mmm/host-owned-coder-grounding-v1",
                        "policy": {
                            "resolved_before_first_coder_decode": True,
                            "baseline_grounding_owned_by_host": True,
                            "baseline_grounding_optional_for_model": False,
                            "model_tool_choice_required_for_baseline": False,
                        },
                        "evidence_bindings": {
                            "project_exact_rag": {"receipt": receipt}
                        },
                    },
                }
            ),
        },
    )


def _frontier_tools(*, valid_grounding: bool) -> tuple[str, ...]:
    tools = (
        _schema("search_code_rag"),
        _schema("java_workspace_symbols"),
        _schema("apply_source_patch"),
    )
    captured: dict[str, tuple[str, ...]] = {}

    class Adapter:
        def generate_turn(self, request):
            captured["tools"] = tuple(
                item["function"]["name"] for item in request.tools
            )
            return SimpleNamespace(tool_calls=(), content="captured")

    remember_authorized_tools(tools)
    try:
        CausalFrontierAdapter(
            Adapter(),
            stage="generation",
            role="coder",
            require_fresh_evidence=True,
        ).generate_turn(
            GenerationRequest(
                messages=_messages(valid_grounding=valid_grounding),
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True,
            )
        )
    finally:
        remember_authorized_tools(())
    return captured["tools"]


def test_valid_host_grounding_unlocks_mutation_frontier() -> None:
    assert _frontier_tools(valid_grounding=True) == ("apply_source_patch",)


def test_incomplete_host_grounding_cannot_unlock_mutation() -> None:
    frontier = _frontier_tools(valid_grounding=False)
    assert frontier
    assert "apply_source_patch" not in frontier
    assert set(frontier).issubset({"search_code_rag", "java_workspace_symbols"})
