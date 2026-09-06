from __future__ import annotations

import inspect
import json

from minecraft_mod_ai import request_requirements
from minecraft_mod_ai import semantic_batching_contract as batching
from minecraft_mod_ai import semantic_requirement_authority as semantic


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


def test_compound_space_prompt_survives_old_resegment_shaped_output() -> None:
    prompt = (
        "우주모드 인데 자원파밍 돈모으기 거래 등으로 우주선을 부위마다 만들어서 만들수있고 "
        "무기 선원 우주선 성능을 거래 구매 등으로 업그레이드 확장 할 수 있고 그렇게해서 "
        "우주로 나갈수있고 우주로 나가면 다른행성의 특수 광물 외게인과 싸움 식민지화등 "
        "여러가지가 가능한 모드"
    )

    class OldShapeRouter:
        def __init__(self) -> None:
            self.calls = 0
            self.tool_names: list[str] = []

        def generate_tool_decision(self, role, messages, **kwargs):  # noqa: ANN001, ANN003
            del role, messages
            self.calls += 1
            self.tool_names.append(str(kwargs.get("tool_name") or ""))
            return {
                "groups": [
                    {
                        "group_id": "compound_0",
                        "parent_semantic_statement": "resource, money and trade bundle",
                    }
                ]
            }

    router = OldShapeRouter()
    catalog = batching.build_bounded_requirement_catalog(prompt, router=router)
    requirements = catalog["requirements"]
    receipt = catalog["semantic_audit"]["semantic_batches"][0]

    assert router.calls == 1
    assert router.tool_names == ["compile_semantic_requirements"]
    assert requirements
    assert requirements[0]["template_profile"]["model_capability_choice"] == "custom.semantic"
    assert receipt["semantic_model_calls_total"] == 1
    assert receipt["semantic_repair_turns_used"] == 0
    assert receipt["host_fallback_count"] == 1


def test_production_semantic_modules_have_no_resegmentation_path_or_example_mapping() -> None:
    source = (
        inspect.getsource(batching)
        + inspect.getsource(semantic)
        + inspect.getsource(request_requirements)
    ).casefold()
    assert "semantic_leaf_pipeline" not in source
    assert "resegment_compound_requirements" not in source
    assert "alien.entity" not in source
    assert "alien.combat" not in source
    assert "외게인" not in source
