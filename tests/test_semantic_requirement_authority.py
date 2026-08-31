from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.evidence_first_planning import EvidencePlanError
from minecraft_mod_ai.semantic_requirement_authority import (
    build_approved_requirement_catalog,
    validate_approved_requirement_catalog,
)


def _item(
    capability: str,
    anchor: str,
    *,
    clause_index: int = 0,
) -> dict:
    return {
        "source_clause_index": clause_index,
        "capability_id": capability,
        "source_anchor": anchor,
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
        if not self.responses:
            raise AssertionError("unexpected extra semantic call")
        return self.responses.pop(0)

    def generate_text(self, *args, **kwargs):
        raise AssertionError("native semantic authority must not use free-form JSON")


def test_semantic_authority_batches_all_authored_clauses_in_one_turn():
    prompt = "Players gather crystals. Players travel to a new region."
    router = TextRouter(
        [
            {
                "requirements": [
                    _item("resource.gathering", "gather crystals", clause_index=0),
                    _item("world.travel", "travel to a new region", clause_index=1),
                ]
            }
        ]
    )

    catalog = build_approved_requirement_catalog(prompt, router)

    validate_approved_requirement_catalog(catalog, prompt=prompt)
    assert len(router.calls) == 1
    request = json.loads(router.calls[0]["messages"][1]["content"])
    assert len(request["host_owned_clauses"]) == 2
    assert "repair_diagnostics" not in request
    assert router.calls[0]["kwargs"]["response_format"] == "text"
    assert router.calls[0]["kwargs"]["response_schema"] is None
    assert {item["capability"] for item in catalog["requirements"]} == {
        "resource.gathering",
        "world.travel",
    }
    assert catalog["semantic_audit"]["normal_model_turns"] == 1
    assert catalog["semantic_audit"]["max_repair_turns"] == 0
    assert catalog["semantic_audit"]["generation_policy"] == "single_pass_constrained"
    assert catalog["semantic_audit"]["source_grounding_owner"] == "host"


def test_minor_model_copy_error_is_host_aligned_without_retry():
    prompt = "우주선을 부위마다 만들어서 만들 수 있고."
    router = TextRouter(
        [
            {
                "requirements": [
                    _item(
                        "vehicle.spacecraft.assembly",
                        "우무선을 부위마다 만들어서 만들수있고",
                    )
                ]
            }
        ]
    )

    catalog = build_approved_requirement_catalog(prompt, router)

    assert len(router.calls) == 1
    span = catalog["requirements"][0]["source_span"]
    assert span["text"] == "우주선을 부위마다 만들어서 만들 수 있고"
    assert prompt[span["char_start"] : span["char_end"]] == span["text"]
    assert span["grounding_method"] == "fuzzy_host_alignment"
    assert span["grounding_similarity"] > 0.9
    assert span["model_anchor"] == "우무선을 부위마다 만들어서 만들수있고"


def test_native_tool_schema_exposes_semantics_not_provenance():
    prompt = "Players collect fragments."
    router = NativeRouter(
        [
            {
                "requirements": [
                    _item("resource.collection", "collect fragments")
                ]
            }
        ]
    )

    catalog = build_approved_requirement_catalog(prompt, router)

    assert len(catalog["requirements"]) == 1
    assert len(router.calls) == 1
    assert router.calls[0]["kwargs"]["tool_name"] == "compile_semantic_requirements"
    props = router.calls[0]["kwargs"]["parameters"]["properties"]["requirements"]["items"]["properties"]
    assert "source_anchor" in props
    for forbidden in (
        "source_quote",
        "provenance_role",
        "depends_on",
        "derived_from",
        "local_id",
        "char_start",
        "char_end",
    ):
        assert forbidden not in props


def test_two_requirements_in_one_clause_receive_distinct_exact_host_spans():
    prompt = "Players can gather and trade crystals."
    router = TextRouter(
        [
            {
                "requirements": [
                    _item("resource.gathering", "gather"),
                    _item("economy.trade", "trade crystals"),
                ]
            }
        ]
    )

    catalog = build_approved_requirement_catalog(prompt, router)

    spans = [item["source_span"] for item in catalog["requirements"]]
    assert len(spans) == 2
    assert {span["text"] for span in spans} == {"gather", "trade crystals"}
    assert spans[0]["char_start"] != spans[1]["char_start"]
    assert all(
        prompt[span["char_start"] : span["char_end"]] == span["text"]
        for span in spans
    )


def test_invalid_clause_rejects_atomic_batch_without_second_model_call():
    prompt = "Players gather crystals. Players open a portal."
    router = TextRouter(
        [
            {
                "requirements": [
                    _item("resource.gathering", "gather crystals", clause_index=0),
                    _item("semantic_13ee7693e9ed", "open a portal", clause_index=1),
                ]
            },
            {
                "requirements": [
                    _item("progression.portal", "open a portal", clause_index=1),
                ]
            },
        ]
    )

    with pytest.raises(
        EvidencePlanError,
        match="semantic requirement authority rejected invalid model output",
    ):
        build_approved_requirement_catalog(prompt, router)

    assert len(router.calls) == 1


def test_unrelated_anchor_is_rejected_without_repair_call():
    prompt = "Players gather crystals."
    router = TextRouter(
        [
            {"requirements": [_item("resource.gathering", "build a submarine")]},
            {"requirements": [_item("resource.gathering", "gather crystals")]},
        ]
    )

    with pytest.raises(
        EvidencePlanError,
        match="semantic requirement authority rejected invalid model output",
    ):
        build_approved_requirement_catalog(prompt, router)

    assert len(router.calls) == 1


def test_invalid_semantics_make_exactly_one_model_call():
    prompt = "Players discover a hidden mechanic."
    invalid = {
        "requirements": [
            _item("semantic_13ee7693e9ed", "discover a hidden mechanic")
        ]
    }
    router = TextRouter([invalid, invalid])

    with pytest.raises(
        EvidencePlanError,
        match="semantic requirement authority rejected invalid model output",
    ):
        build_approved_requirement_catalog(prompt, router)

    assert len(router.calls) == 1


def test_model_cannot_promote_design_provenance_or_dependencies():
    prompt = "Players exchange collected items."
    supplied = _item("economy.exchange", "exchange collected items")
    supplied["provenance_role"] = "selected_design_alternative"
    supplied["depends_on"] = ["anything"]
    router = TextRouter([{"requirements": [supplied]}])

    catalog = build_approved_requirement_catalog(prompt, router)

    requirement = catalog["requirements"][0]
    assert requirement["provenance_role"] == "explicit"
    assert requirement["depends_on"] == []
    assert requirement["derived_from"] == []
