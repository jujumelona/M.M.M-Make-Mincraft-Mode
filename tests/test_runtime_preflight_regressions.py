from __future__ import annotations

import json
from typing import Any

from minecraft_mod_ai import causal_frontier_adapter as frontier_module
from minecraft_mod_ai import small_model_compacting_adapter as compaction_module
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse
from minecraft_mod_ai.retrieval_progress import _stable_value, evidence_fingerprint
from minecraft_mod_ai.runtime_preflight import run_runtime_preflight


class _CaptureAdapter:
    def __init__(self) -> None:
        self.request: GenerationRequest | None = None

    def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
        self.request = request
        return GenerationResponse(content="ok")


def test_large_structured_user_request_keeps_terminal_intent() -> None:
    content = json.dumps(
        {
            "phase": "implement_module",
            "task": "Implement the approved Minecraft/Fabric feature.",
            "research_context": "x" * 40_000,
        }
    )

    query = frontier_module._query(({"role": "user", "content": content},))

    assert "implement_module" in query
    assert "Implement the approved Minecraft/Fabric feature." in query
    assert len(query) < 12_000


def test_compaction_request_clone_preserves_upstream_fields(monkeypatch: Any) -> None:
    capture = _CaptureAdapter()
    adapter = compaction_module.CompactingAdapter(capture)
    request = GenerationRequest(
        messages=({"role": "user", "content": "hello"},),
        task="task-sentinel",
        prompt="prompt-sentinel",
        metadata={"trace": "metadata-sentinel"},
    )
    monkeypatch.setattr(
        compaction_module,
        "compact_messages",
        lambda messages: (*tuple(messages), {"role": "system", "content": "compacted"}),
    )

    response = adapter.generate_turn(request)

    assert response.content == "ok"
    assert capture.request is not None
    assert capture.request.task == request.task
    assert capture.request.prompt == request.prompt
    assert capture.request.metadata == request.metadata


def test_unordered_retrieval_values_have_one_canonical_fingerprint() -> None:
    assert _stable_value({"facts": {"b", "a"}}, drop_volatile=False) == {
        "facts": ["a", "b"]
    }
    assert evidence_fingerprint({"facts": {"a", "b"}}) == evidence_fingerprint(
        {"facts": frozenset(("b", "a"))}
    )


def test_model_free_runtime_preflight_is_idempotent() -> None:
    run_runtime_preflight()
    run_runtime_preflight()
