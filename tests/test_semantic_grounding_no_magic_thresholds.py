from __future__ import annotations

from minecraft_mod_ai.semantic_requirement_authority import (
    _evaluate_batch,
    build_approved_requirement_catalog,
)


class _Router:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def generate_tool_decision(self, *_args, **_kwargs):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("unexpected semantic repair call")
        return self.payloads.pop(0)


def _item(capability: str, anchor: str, statement: str) -> dict:
    return {
        "source_clause_index": 0,
        "capability_id": capability,
        "source_anchor": anchor,
        "semantic_statement": statement,
        "given": "the authored gameplay state exists",
        "when": "the player performs the authored action",
        "then": "the authored outcome is observable",
    }


def test_short_korean_anchor_is_grounded_without_minimum_length_rule() -> None:
    prompt = "자원을 모아서 거래하고 돈을 벌 수 있다."
    router = _Router(
        {
            "requirements": [
                _item("economy.trade", "거래", "수집한 자원을 거래한다"),
            ]
        }
    )

    catalog = build_approved_requirement_catalog(prompt, router)

    assert router.calls == 1
    span = catalog["requirements"][0]["source_span"]
    assert span["text"] == "거래"
    assert prompt[span["char_start"] : span["char_end"]] == "거래"


def test_typo_locator_uses_authored_semantic_evidence_without_ratio_cutoff() -> None:
    prompt = "우주선을 부위마다 만들어서 조립하고 우주선 성능을 업그레이드한다."
    router = _Router(
        {
            "requirements": [
                _item(
                    "vehicle.spacecraft.assembly",
                    "우무선을 부위마다 만들어서",
                    "우주선을 부위마다 만들어서 조립한다",
                ),
                _item(
                    "vehicle.spacecraft.upgrade",
                    "우주선 성능을 업그레이드",
                    "우주선 성능을 업그레이드한다",
                ),
            ]
        }
    )

    catalog = build_approved_requirement_catalog(prompt, router)

    assert router.calls == 1
    assert {item["capability"] for item in catalog["requirements"]} == {
        "vehicle.spacecraft.assembly",
        "vehicle.spacecraft.upgrade",
    }
    assembly = next(
        item for item in catalog["requirements"]
        if item["capability"] == "vehicle.spacecraft.assembly"
    )
    assert "만들어서" in assembly["source_span"]["text"]
    assert assembly["source_span"]["grounding_method"] == "fuzzy_host_alignment"


def test_invalid_leaf_does_not_erase_valid_siblings_from_same_clause() -> None:
    prompt = "Players gather crystals and trade crystals."
    clause = {
        "clause_index": 0,
        "char_start": 0,
        "char_end": len(prompt),
        "text": prompt,
        "text_sha256": "unused-by-batch-evaluator",
    }
    payload = {
        "requirements": [
            _item("resource.gathering", "gather crystals", "gather crystals"),
            _item("semantic_deadbeef", "trade crystals", "trade crystals"),
        ]
    }

    nodes, invalid_clauses, diagnostics = _evaluate_batch(payload, [clause])

    assert [node["capability_id"] for node in nodes] == ["resource.gathering"]
    assert invalid_clauses == {0}
    assert not any(item["error_code"] == "REQ_SOURCE_COVERAGE" for item in diagnostics)
