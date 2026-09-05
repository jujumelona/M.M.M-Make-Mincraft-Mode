from __future__ import annotations

import inspect
import json

from minecraft_mod_ai import semantic_batching_contract as batching
from minecraft_mod_ai import semantic_leaf_pipeline as leaf_pipeline
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


def test_segmentation_schema_has_no_capability_authority() -> None:
    schema = leaf_pipeline._segmentation_schema(0)
    properties = schema["properties"]["leaves"]["items"]["properties"]

    assert "capability_id" not in properties
    assert set(properties) == {
        "source_clause_index",
        "source_anchor",
        "semantic_statement",
        "given",
        "when",
        "then",
        "semantic_type",
    }


def test_classification_schema_can_only_choose_leaf_and_capability() -> None:
    schema = leaf_pipeline._classification_schema(1)
    properties = schema["properties"]["classifications"]["items"]["properties"]

    assert set(properties) == {"leaf_index", "capability_id"}
    assert schema["properties"]["classifications"]["items"]["additionalProperties"] is False


def test_classification_rejects_attempt_to_rewrite_immutable_semantics() -> None:
    result, diagnostics = leaf_pipeline._classification_diagnostics(
        {
            "classifications": [
                {
                    "leaf_index": 0,
                    "capability_id": "resource.mining",
                    "semantic_statement": "rewritten",
                }
            ]
        },
        1,
    )

    assert result is None
    assert any(item["error_code"] == "REQ_MODEL_AUTHORITY_OVERREACH" for item in diagnostics)


class _TwoStageRepairingRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict[str, object]]] = []
        self.tool_names: list[str] = []

    def generate_tool_decision(self, role, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        self.messages.append([dict(message) for message in messages])
        self.tool_names.append(str(kwargs.get("tool_name") or ""))

        if self.calls == 1:
            return {
                "leaves": [
                    {
                        "source_clause_index": 0,
                        "source_anchor": "mine ore",
                        "semantic_statement": "Players mine ore.",
                        "given": "Mineable ore exists.",
                        "when": "A player mines ore.",
                        "then": "Ore enters inventory.",
                    }
                ]
            }
        if self.calls == 2:
            return {
                "leaves": [
                    {
                        "source_clause_index": 0,
                        "source_anchor": "mine ore and",
                        "semantic_statement": "Players mine ore.",
                        "given": "Mineable ore exists.",
                        "when": "A player mines ore.",
                        "then": "Ore enters inventory.",
                    },
                    {
                        "source_clause_index": 0,
                        "source_anchor": "trade goods",
                        "semantic_statement": "Players trade goods.",
                        "given": "A valid trade exists.",
                        "when": "A player confirms the trade.",
                        "then": "The validated exchange completes.",
                    },
                ]
            }
        if self.calls == 3:
            return {
                "classifications": [
                    {"leaf_index": 0, "capability_id": "resource.mining"},
                    {"leaf_index": 1, "capability_id": "economy.trade"},
                ]
            }
        raise AssertionError("unexpected extra model call")


def test_bounded_semantic_batch_repairs_segmentation_then_classifies_immutable_leaves() -> None:
    router = _TwoStageRepairingRouter()
    text = "mine ore and trade goods"
    nodes, receipts = batching._generate_bounded_nodes(
        router,
        [_clause(text)],
        batch_size=1,
    )

    assert router.calls == 3
    assert router.tool_names == [
        "segment_semantic_requirements",
        "segment_semantic_requirements",
        "classify_semantic_requirements",
    ]
    assert [node["capability_id"] for node in nodes] == ["resource.mining", "economy.trade"]
    assert receipts[0]["segmentation_attempts"] == 2
    assert receipts[0]["classification_attempts"] == 1
    assert receipts[0]["semantic_model_calls_total"] == 3
    assert receipts[0]["semantic_repair_turns_used"] == 1
    assert receipts[0]["segmentation_repaired"] is True
    assert receipts[0]["classification_repaired"] is False

    repair_system_message = str(router.messages[1][0]["content"])
    assert "REQ_SOURCE_PARTITION_GAP" in repair_system_message
    assert "source_anchor is not a keyword label" in repair_system_message

    classification_payload = json.loads(str(router.messages[2][1]["content"]))
    grounded = classification_payload["host_grounded_leaves"]
    assert [item["source_text"] for item in grounded] == ["mine ore and", "trade goods"]
    assert [item["semantic_statement"] for item in grounded] == [
        "Players mine ore.",
        "Players trade goods.",
    ]


def test_production_fidelity_modules_contain_no_example_specific_mapping() -> None:
    source = (
        inspect.getsource(fidelity) + inspect.getsource(leaf_pipeline)
    ).casefold()
    assert "alien.entity" not in source
    assert "alien.combat" not in source
    assert "외게인" not in source
    assert "싸움" not in source
