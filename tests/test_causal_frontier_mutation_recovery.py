from __future__ import annotations

import json

from minecraft_mod_ai.causal_frontier_adapter import (
    _failed_mutation_needs_evidence_refresh,
    _latest_failed_source_mutation,
)


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_message(name: str, payload: dict[str, object]) -> dict[str, str]:
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps(payload),
    }


def _candidates() -> tuple[dict[str, object], ...]:
    return (
        _schema("java_workspace_symbols"),
        _schema("apply_source_edit"),
    )


def test_transport_failed_source_mutation_requires_fresh_evidence() -> None:
    messages = (
        _tool_message("java_workspace_symbols", {"ok": True}),
        _tool_message("apply_source_edit", {"ok": False, "error": "edit rejected"}),
    )

    assert _latest_failed_source_mutation(messages) == 1
    assert _failed_mutation_needs_evidence_refresh(messages, _candidates()) is True


def test_semantic_failed_source_mutation_requires_fresh_evidence() -> None:
    messages = (
        _tool_message("java_workspace_symbols", {"ok": True}),
        _tool_message(
            "apply_source_edit",
            {"ok": True, "result": {"status": "FAILED", "reason": "stale edit"}},
        ),
    )

    assert _latest_failed_source_mutation(messages) == 1
    assert _failed_mutation_needs_evidence_refresh(messages, _candidates()) is True


def test_fresh_evidence_after_failure_starts_new_repair_epoch() -> None:
    messages = (
        _tool_message("java_workspace_symbols", {"ok": True}),
        _tool_message("apply_source_edit", {"ok": False}),
        _tool_message("java_workspace_symbols", {"ok": True}),
    )

    assert _latest_failed_source_mutation(messages) == 1
    assert _failed_mutation_needs_evidence_refresh(messages, _candidates()) is False
