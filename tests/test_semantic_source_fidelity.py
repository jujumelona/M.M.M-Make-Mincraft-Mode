from __future__ import annotations

import inspect
import json

from minecraft_mod_ai import semantic_batching_contract as batching
from minecraft_mod_ai import semantic_requirement_authority as semantic
from minecraft_mod_ai import request_requirements


def _clause(text: str, index: int = 0) -> dict[str, object]:
    return {
        "clause_index": index,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "text_sha256": "sha256:test",
    }


def test_host_grounding_accepts_a_behavior_locator_without_requiring_full_partition() -> None:
    text = "mine ore and trade goods"
    clause = _clause(text)

    mining = semantic._ground_source_anchor(clause, "mine ore")
    trading = semantic._ground_source_anchor(clause, "trade goods")

    assert mining is not None
    assert trading is not None
    assert mining["source_quote"] == "mine ore"
    assert trading["source_quote"] == "trade goods"
    assert mining["source_end"] <= trading["source_start"]


class _SinglePassRouter:
    def __init__(self, *, malformed: bool = False) -> None:
        self.calls = 0
        self.tool_names: list[str] = []
        self.malformed = malformed

    def generate_tool_decision(self, role, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        self.tool_names.append(str(kwargs.get("tool_name") or ""))
        if self.malformed:
            return {"bad": "payload"}
        payload = json.loads(messages[-1]["content"])
        clause = payload["host_owned_clauses"][0]
        return {
            "requirements": [
                {
                    "source_clause_index": clause["source_clause_index"],
                    "capability_id": "resource.mining",
                    "source_anchor": "mine ore",
                    "semantic_statement": "Players mine ore.",
                    "given": "Mineable ore exists.",
                    "when": "A player mines ore.",
                    "then": "Ore enters inventory.",
                    "semantic_type": "gameplay_mechanic",
                },
                {
                    "source_clause_index": clause["source_clause_index"],
                    "capability_id": "economy.trade",
                    "source_anchor": "trade goods",
                    "semantic_statement": "Players trade goods.",
                    "given": "A valid trade exists.",
                    "when": "A player confirms the trade.",
                    "then": "The validated exchange completes.",
                    "semantic_type": "gameplay_mechanic",
                },
            ]
        }


def test_semantic_batch_uses_one_structured_call_without_partition_repair() -> None:
    router = _SinglePassRouter()
    text = "mine ore and trade goods"

    nodes, receipts = batching._generate_bounded_nodes(
        router,
        [_clause(text)],
        batch_size=1,
    )

    assert router.calls == 1
    assert router.tool_names == ["compile_semantic_requirements"]
    assert [node["capability_id"] for node in nodes] == [
        "resource.mining",
        "economy.trade",
    ]
    assert receipts[0]["semantic_model_calls_total"] == 1
    assert receipts[0]["semantic_repair_turns_used"] == 0
    assert receipts[0]["host_fallback_count"] == 0


def test_malformed_semantic_output_keeps_host_clause_default() -> None:
    router = _SinglePassRouter(malformed=True)
    text = "mine ore and trade goods"

    nodes, receipts = batching._generate_bounded_nodes(
        router,
        [_clause(text)],
        batch_size=1,
    )

    assert router.calls == 1
    assert len(nodes) == 1
    assert nodes[0]["model_capability_choice"] == "custom.semantic"
    assert nodes[0]["source_quote"] == text
    assert receipts[0]["host_fallback_count"] == 1


def test_production_semantic_modules_contain_no_example_specific_mapping() -> None:
    source = (
        inspect.getsource(batching)
        + inspect.getsource(semantic)
        + inspect.getsource(request_requirements)
    ).casefold()
    assert "alien.entity" not in source
    assert "alien.combat" not in source
    assert "외게인" not in source
