from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.causal_frontier_adapter import CausalFrontierAdapter
from minecraft_mod_ai.causal_state_ledger import CausalStateLedger
from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
from minecraft_mod_ai.model_adapters import GenerationRequest


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


def _rag_observation(call_id: str = "rag-1") -> dict[str, object]:
    return {
        "role": "tool",
        "name": "search_code_rag",
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "ok": True,
                "result": {
                    "hits": [{"path": "src/main/java/example/Test.java"}],
                    "receipt": {
                        "result_count": 1,
                        "coverage_score": 1.0,
                        "relevance_score": 1.0,
                    },
                },
            }
        ),
    }


def _failed_edit(index: int) -> dict[str, object]:
    return {
        "role": "tool",
        "name": "apply_source_edit",
        "tool_call_id": f"edit-{index}",
        "content": json.dumps(
            {
                "ok": False,
                "tool": "apply_source_edit",
                "error": (
                    "AgentToolRuntimeError: exact edit did not match "
                    "[workspace_impact=unchanged]"
                ),
            }
        ),
    }


def _patch_observation(*, applied: bool, call_id: str) -> dict[str, object]:
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


class _RecordingAdapter:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate_turn(self, request: GenerationRequest):
        self.requests.append(request)
        if request.tool_choice == "auto":
            function = request.tools[0]["function"]
            assert isinstance(function, dict)
            name = str(function["name"])
        else:
            assert isinstance(request.tool_choice, dict)
            function = request.tool_choice["function"]
            assert isinstance(function, dict)
            name = str(function["name"])
        return SimpleNamespace(
            tool_calls=(SimpleNamespace(name=name),),
            content="",
        )


def test_repeated_safe_edit_failures_open_corrective_evidence_epoch() -> None:
    """Live composition must recover through evidence instead of leaking to round 12."""

    rag = _schema("search_code_rag")
    edit = _schema("apply_source_edit")
    surface = (rag, edit)
    inner = _RecordingAdapter()
    adapter = CausalFrontierAdapter(
        _WritableProgressAdapter(inner),
        stage="generation",
        role="coder",
        require_fresh_evidence=True,
        authorized_surface=surface,
        preference={},
    )
    failed_messages = (
        _implement_message(),
        _rag_observation(),
        _failed_edit(1),
        _failed_edit(2),
    )
    recovery_request = GenerationRequest(
        messages=failed_messages,
        tools=surface,
        tool_validation_schemas=surface,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    recovery_turn = adapter.generate_turn(recovery_request)

    assert recovery_turn.tool_calls[0].name == "search_code_rag"
    assert len(inner.requests) == 1
    assert inner.requests[0].tool_choice == "auto"
    assert [item["function"]["name"] for item in inner.requests[0].tools] == [
        "search_code_rag"
    ]

    refreshed_request = GenerationRequest(
        messages=(*failed_messages, _rag_observation("rag-recovery")),
        tools=surface,
        tool_validation_schemas=surface,
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    edit_turn = adapter.generate_turn(refreshed_request)

    assert edit_turn.tool_calls[0].name == "apply_source_edit"
    assert len(inner.requests) == 2
    assert inner.requests[1].tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }


def test_patch_transport_failures_do_not_create_adapter_local_retry_budget() -> None:
    """Mutation retry ownership stays in CausalStateLedger, not the transport wrapper."""

    patch = _schema("apply_source_patch")
    observations = (
        _patch_observation(applied=False, call_id="patch-1"),
        _patch_observation(applied=False, call_id="patch-2"),
    )
    inner = _RecordingAdapter()
    adapter = _WritableProgressAdapter(inner)
    request = GenerationRequest(
        messages=(_implement_message(), *observations),
        tools=(patch,),
        tool_validation_schemas=(patch,),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_patch"},
        },
        parallel_tool_calls=False,
    )

    turn = adapter.generate_turn(request)

    assert turn.tool_calls[0].name == "apply_source_patch"
    assert len(inner.requests) == 1
    assert inner.requests[0].tool_choice == request.tool_choice


def test_causal_state_requires_the_same_applied_mutation_proof() -> None:
    """Transport-only PASS must never synthesize the repaired/project_changed facts."""

    rag = _schema("search_code_rag")
    patch = _schema("apply_source_patch")
    schemas = (rag, patch)
    ledger = CausalStateLedger()
    transport_only = (
        _implement_message(),
        _rag_observation(),
        _patch_observation(applied=False, call_id="patch-unproved"),
    )

    unproved = ledger.resolve(
        transport_only,
        schemas,
        require_fresh_evidence=True,
        query_fn=lambda _messages: "repair",
    )

    assert "repaired" not in unproved.state
    assert "project_changed" not in unproved.state
    assert "evidence_ready" not in unproved.state

    proved = ledger.resolve(
        (*transport_only, _rag_observation("rag-after-unproved"), _patch_observation(applied=True, call_id="patch-proved")),
        schemas,
        require_fresh_evidence=True,
        query_fn=lambda _messages: "repair",
    )

    assert "repaired" in proved.state
    assert "project_changed" in proved.state
