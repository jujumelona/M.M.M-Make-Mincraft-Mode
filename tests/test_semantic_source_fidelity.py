from __future__ import annotations

import inspect

from minecraft_mod_ai import semantic_batching_contract as batching
from minecraft_mod_ai import semantic_source_fidelity as fidelity


def _clause(text: str, index: int = 0) -> dict[str, object]:
    return {
        "clause_index": index,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "text_sha256": "sha256:test",
    }


def _node(text: str, anchor: str, capability: str, clause_index: int = 0) -> dict[str, object]:
    start = text.index(anchor)
    return {
        "source_clause_index": clause_index,
        "source_start": start,
        "source_end": start + len(anchor),
        "capability_id": capability,
    }


def test_language_neutral_gate_rejects_dropped_authored_action_span() -> None:
    text = "외게인과 싸움 식민지화"
    diagnostics = fidelity.validate_semantic_source_partition(
        [
            _node(text, "외게인", "alien.entity"),
            _node(text, "식민지화", "colony.colonization"),
        ],
        [_clause(text)],
    )

    gaps = [item for item in diagnostics if item["error_code"] == "REQ_SOURCE_PARTITION_GAP"]
    assert gaps
    assert any("싸움" in span["text"] for span in gaps[0]["spans"])


def test_same_gate_rejects_unrelated_domain_source_loss() -> None:
    text = "trade goods for credits then launch ship"
    diagnostics = fidelity.validate_semantic_source_partition(
        [
            _node(text, "trade goods", "economy.trade"),
            _node(text, "launch ship", "space.launch"),
        ],
        [_clause(text)],
    )

    gaps = [item for item in diagnostics if item["error_code"] == "REQ_SOURCE_PARTITION_GAP"]
    assert gaps
    assert any("credits" in span["text"] for span in gaps[0]["spans"])


def test_gate_accepts_non_overlapping_complete_partition() -> None:
    text = "mine ore and trade goods"
    diagnostics = fidelity.validate_semantic_source_partition(
        [
            _node(text, "mine ore and", "resource.mining"),
            _node(text, "trade goods", "economy.trade"),
        ],
        [_clause(text)],
    )

    assert diagnostics == ()


def test_gate_rejects_double_owned_authored_characters() -> None:
    text = "build and upgrade ship"
    first = _node(text, "build and upgrade", "spaceship.component_crafting")
    second = _node(text, "upgrade ship", "spacecraft.performance_upgrade")

    diagnostics = fidelity.validate_semantic_source_partition(
        [first, second],
        [_clause(text)],
    )

    assert any(item["error_code"] == "REQ_SOURCE_PARTITION_OVERLAP" for item in diagnostics)


class _RepairingRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict[str, object]]] = []

    def generate_tool_decision(self, role, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        self.messages.append([dict(message) for message in messages])
        if self.calls == 1:
            return {
                "requirements": [
                    {
                        "source_clause_index": 0,
                        "capability_id": "resource.mining",
                        "source_anchor": "mine ore",
                        "semantic_statement": "Players mine ore.",
                        "given": "Mineable ore exists.",
                        "when": "A player mines ore.",
                        "then": "Ore enters inventory.",
                    }
                ]
            }
        return {
            "requirements": [
                {
                    "source_clause_index": 0,
                    "capability_id": "resource.mining",
                    "source_anchor": "mine ore and",
                    "semantic_statement": "Players mine ore.",
                    "given": "Mineable ore exists.",
                    "when": "A player mines ore.",
                    "then": "Ore enters inventory.",
                },
                {
                    "source_clause_index": 0,
                    "capability_id": "economy.trade",
                    "source_anchor": "trade goods",
                    "semantic_statement": "Players trade goods.",
                    "given": "A valid trade exists.",
                    "when": "A player confirms the trade.",
                    "then": "The validated exchange completes.",
                },
            ]
        }


def test_bounded_semantic_batch_repairs_source_loss_once() -> None:
    router = _RepairingRouter()
    text = "mine ore and trade goods"
    nodes, receipts = batching._generate_bounded_nodes(
        router,
        [_clause(text)],
        batch_size=1,
    )

    assert router.calls == 2
    assert [node["capability_id"] for node in nodes] == ["resource.mining", "economy.trade"]
    assert receipts[0]["semantic_attempts"] == 2
    assert receipts[0]["semantic_repaired"] is True
    repair_system_message = str(router.messages[1][0]["content"])
    assert "REQ_SOURCE_PARTITION_GAP" in repair_system_message
    assert "source_anchor is not a keyword label" in repair_system_message


def test_production_fidelity_gate_contains_no_example_specific_mapping() -> None:
    source = inspect.getsource(fidelity).casefold()
    assert "alien.entity" not in source
    assert "alien.combat" not in source
    assert "외게인" not in source
    assert "싸움" not in source
