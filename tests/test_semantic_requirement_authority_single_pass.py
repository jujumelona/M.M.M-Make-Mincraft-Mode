from __future__ import annotations

import pytest

from minecraft_mod_ai import semantic_requirement_authority as authority


class _InvalidSemanticRouter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_tool_decision(self, *args, **kwargs):
        self.calls += 1
        return {"requirements": []}


def test_invalid_semantic_batch_fails_closed_without_repair_turn() -> None:
    prompt = "잡몹부터 보스까지 템들 레벨도 점점 성장 강화시스템등 모두 구현해야해"
    clauses = [
        {
            "clause_index": 0,
            "char_start": 0,
            "char_end": len(prompt),
            "text": prompt,
            "text_sha256": authority._sha256(prompt),
        }
    ]
    router = _InvalidSemanticRouter()

    with pytest.raises(
        authority._evidence.EvidencePlanError,
        match="rejected invalid model output",
    ):
        authority._generate_approved_nodes(prompt, router, clauses)

    assert router.calls == 1


def test_semantic_messages_do_not_carry_repair_protocol() -> None:
    prompt = "강화시스템 구현"
    clauses = [
        {
            "clause_index": 0,
            "char_start": 0,
            "char_end": len(prompt),
            "text": prompt,
            "text_sha256": authority._sha256(prompt),
        }
    ]

    messages = authority._model_messages(clauses)

    assert "repair_diagnostics" not in messages[1]["content"]
