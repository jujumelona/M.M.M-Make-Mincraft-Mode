from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.evidence_first_planning import EvidencePlanError
from minecraft_mod_ai.semantic_requirement_authority import (
    build_approved_requirement_catalog,
    validate_approved_requirement_catalog,
)


def _item(capability: str, quote: str) -> dict:
    return {
        "capability_id": capability,
        "source_quote": quote,
        "semantic_statement": capability.replace(".", " "),
        "given": "the feature precondition exists",
        "when": "the player performs the requested behavior",
        "then": "the requested outcome is observable",
    }


class TextRouter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra semantic call")
        value = self.responses.pop(0)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class NativeRouter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        return self.responses.pop(0)

    def generate_text(self, *args, **kwargs):
        raise AssertionError("native semantic authority must not use free-form JSON")


def test_semantic_authority_requests_each_authored_clause_separately():
    prompt = "Players gather crystals. Players travel to a new region."
    router = TextRouter([
        {"requirements": [_item("resource.gathering", "gather crystals")]},
        {"requirements": [_item("world.travel", "travel to a new region")]},
    ])
    catalog = build_approved_requirement_catalog(prompt, router)
    validate_approved_requirement_catalog(catalog, prompt=prompt)
    assert len(router.calls) == 2
    first = json.loads(router.calls[0]["messages"][1]["content"])
    second = json.loads(router.calls[1]["messages"][1]["content"])
    assert first["current_clause"] != second["current_clause"]
    assert router.calls[0]["kwargs"]["response_format"] == "text"
    assert router.calls[0]["kwargs"]["response_schema"] is None
    assert {r["capability"] for r in catalog["requirements"]} == {"resource.gathering", "world.travel"}
    assert all(r["provenance_role"] == "explicit" for r in catalog["requirements"])
    assert all(r["depends_on"] == [] for r in catalog["requirements"])


def test_one_bad_json_retries_only_the_failed_clause():
    prompt = "Players gather crystals. Players open a portal."
    router = TextRouter([
        {"requirements": [_item("resource.gathering", "gather crystals")]},
        "not json",
        {"requirements": [_item("progression.portal", "open a portal")]},
    ])
    catalog = build_approved_requirement_catalog(prompt, router)
    assert len(catalog["requirements"]) == 2
    assert len(router.calls) == 3
    retry = json.loads(router.calls[2]["messages"][1]["content"])
    assert retry["current_clause_index"] == 1
    assert retry["repair_diagnostic"]["error_code"] == "REQ_CLAUSE_MODEL_RESPONSE"
    assert retry["repair_diagnostic"]["repair_scope"] == "clause:1"


def test_native_tool_decision_is_preferred_for_clause_semantics():
    prompt = "Players collect fragments."
    router = NativeRouter([{"requirements": [_item("resource.collection", "collect fragments")]}])
    catalog = build_approved_requirement_catalog(prompt, router)
    assert len(catalog["requirements"]) == 1
    assert router.calls[0]["kwargs"]["tool_name"] == "approve_semantic_clause"
    props = router.calls[0]["kwargs"]["parameters"]["properties"]["requirements"]["items"]["properties"]
    assert "provenance_role" not in props
    assert "depends_on" not in props


def test_multiple_meanings_in_one_clause_need_distinct_grounding_quotes():
    prompt = "Players gather crystals, then trade crystals."
    router = TextRouter([
        {"requirements": [_item("resource.gathering", "gather crystals")]},
        {"requirements": [_item("economy.trade", "trade crystals")]},
    ])
    catalog = build_approved_requirement_catalog(prompt, router)
    assert len(catalog["requirements"]) == 2
    assert {r["source_span"]["text"] for r in catalog["requirements"]} == {"gather crystals", "trade crystals"}


def test_opaque_capability_retries_clause_and_fails_closed_on_repeat():
    prompt = "Players discover a hidden mechanic."
    invalid = {"requirements": [_item("semantic_13ee7693e9ed", "discover a hidden mechanic")]}
    router = TextRouter([invalid, invalid])
    with pytest.raises(EvidencePlanError, match="semantic clause approval reached a no-progress fixed point"):
        build_approved_requirement_catalog(prompt, router)
    second = json.loads(router.calls[1]["messages"][1]["content"])
    assert second["repair_diagnostic"]["error_code"] == "REQ_CAPABILITY_ID"
    assert second["repair_diagnostic"]["repair_scope"] == "clause:0"


def test_model_cannot_promote_design_provenance_or_cross_clause_dependencies():
    prompt = "Players exchange collected items."
    supplied = _item("economy.exchange", "exchange collected items")
    supplied["provenance_role"] = "selected_design_alternative"
    supplied["depends_on"] = ["anything"]
    router = TextRouter([{"requirements": [supplied]}])
    catalog = build_approved_requirement_catalog(prompt, router)
    requirement = catalog["requirements"][0]
    assert requirement["provenance_role"] == "explicit"
    assert requirement["depends_on"] == []
