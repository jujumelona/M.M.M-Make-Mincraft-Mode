from __future__ import annotations

import json

from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse, ToolCall
from minecraft_mod_ai.source_mutation_contract import mutation_observation_applied


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _implement_message() -> dict[str, object]:
    return {
        "role": "user",
        "content": json.dumps(
            {
                "phase": "implement_module",
                "task": "Implement the approved feature in the current project.",
            }
        ),
    }


def _patch_observation(*, applied: bool, call_id: str) -> dict[str, object]:
    result: dict[str, object]
    if applied:
        result = {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "operations": [
                {
                    "path": "src/main/java/example/Test.java",
                    "operation": "edit",
                    "before_sha256": "sha256:" + "1" * 64,
                    "after_sha256": "sha256:" + "2" * 64,
                }
            ],
        }
    else:
        result = {"status": "PASS"}
    return {
        "role": "tool",
        "name": "apply_source_patch",
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "ok": True,
                "tool": "apply_source_patch",
                "result": result,
            }
        ),
    }


def test_compat_writable_adapter_does_not_own_mutation_retry_policy() -> None:
    requests: list[GenerationRequest] = []

    class RecordingAdapter:
        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            requests.append(request)
            return GenerationResponse(
                tool_calls=(ToolCall(id="model-edit", name="apply_source_patch"),)
            )

    patch = _schema("apply_source_patch")
    request = GenerationRequest(
        messages=(_implement_message(), _patch_observation(applied=False, call_id="old")),
        tools=(patch,),
        tool_validation_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    turn = _WritableProgressAdapter(RecordingAdapter()).generate_turn(request)

    assert requests == [request]
    assert requests[0].tool_choice == "auto"
    assert requests[0].parallel_tool_calls is True
    assert [call.name for call in turn.tool_calls] == ["apply_source_patch"]


def test_transport_success_without_applied_receipt_is_not_mutation_progress() -> None:
    assert mutation_observation_applied(
        _patch_observation(applied=False, call_id="patch-unproved")
    ) is False
    assert mutation_observation_applied(
        _patch_observation(applied=True, call_id="patch-proved")
    ) is True
