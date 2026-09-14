from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.planner import LocalTransformersPlanner
from minecraft_mod_ai import planner as planner_module
from minecraft_mod_ai import routed_planner
from minecraft_mod_ai.progress_aware_tool_loop import _atomic_output_recovery_instruction


def _mutation_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "apply one bounded source mutation",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string"},
                },
                "required": ["path", "operation"],
                "additionalProperties": False,
            },
        },
    }


def test_local_planner_uses_canonical_routed_pipeline_not_legacy_whole_json(monkeypatch):
    sentinel = object()
    seen: dict[str, object] = {}

    class FakeRoutedPlanner:
        def __init__(self, *, profile: str):
            seen["profile"] = profile

        def plan(self, prompt: str):
            seen["prompt"] = prompt
            return sentinel

    def legacy_json_must_not_run(text: str):
        raise AssertionError("production local planning entered legacy whole-object JSON parsing")

    monkeypatch.setattr(routed_planner, "RoutedPlanner", FakeRoutedPlanner)
    monkeypatch.setattr(planner_module, "_extract_json_object", legacy_json_must_not_run)

    result = LocalTransformersPlanner(profile="t4_local").plan("make a small item mod")

    assert result is sentinel
    assert seen == {"profile": "t4_local", "prompt": "make a small item mod"}


def test_output_recovery_is_current_slice_only_and_host_preserves_state():
    request = GenerationRequest(
        messages=({"role": "user", "content": "edit the localized source"},),
        tools=(_mutation_tool(),),
        tool_choice="required",
        parallel_tool_calls=False,
    )

    instruction = _atomic_output_recovery_instruction(request)
    lowered = instruction.casefold()

    assert "exactly one visible source-mutation tool" in lowered
    assert "one small semantic edit" in lowered
    assert "do not continue, reproduce, or complete that oversized payload" in lowered
    assert "the host will preserve the same mutation target and workspace state" in lowered
    assert "complete java file" in lowered


def test_output_recovery_never_requests_full_state_reconstruction():
    request = GenerationRequest(
        messages=({"role": "user", "content": "continue"},),
        tools=(),
        tool_choice=None,
    )

    instruction = _atomic_output_recovery_instruction(request).casefold()

    assert "already-grounded state" in instruction
    assert "do not emit a long reconstruction" in instruction
