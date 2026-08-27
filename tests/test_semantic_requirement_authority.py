from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.evidence_first_planning import EvidencePlanError
from minecraft_mod_ai.semantic_requirement_authority import (
    build_approved_requirement_catalog,
    validate_approved_requirement_catalog,
)


class _Router:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("semantic authority requested an unexpected extra response")
        value = self.responses.pop(0)
        return json.dumps(value, ensure_ascii=False)


def _req(
    local_id: str,
    capability_id: str,
    clause: int,
    quote: str,
    *,
    role: str = "explicit",
    derived_from=(),
    depends_on=(),
    reason: str = "",
):
    return {
        "local_id": local_id,
        "capability_id": capability_id,
        "provenance_role": role,
        "source_clause_index": clause,
        "source_quote": quote,
        "semantic_statement": capability_id.replace(".", " "),
        "derived_from": list(derived_from),
        "depends_on": list(depends_on),
        "derivation_reason": reason,
        "observable_behavior": {
            "given": f"{local_id} precondition",
            "when": f"{local_id} action occurs",
            "then": f"{local_id} observable outcome occurs",
        },
    }


def test_authority_creates_precise_stable_graph_and_concrete_acceptance():
    prompt = "Players gather crystals and trade them. Players travel to a new region."
    router = _Router(
        [
            {
                "requirements": [
                    _req("gather", "resource.gathering", 0, "gather crystals"),
                    _req(
                        "trade",
                        "economy.trade",
                        0,
                        "trade them",
                        depends_on=("gather",),
                    ),
                    _req(
                        "travel",
                        "world.travel",
                        1,
                        "travel to a new region",
                        depends_on=("trade",),
                    ),
                ]
            }
        ]
    )

    catalog = build_approved_requirement_catalog(prompt, router)
    validate_approved_requirement_catalog(catalog, prompt=prompt)

    assert catalog["semantic_audit"]["status"] == "APPROVED"
    assert catalog["semantic_audit"]["covered_clause_count"] == 2
    assert len(catalog["requirements"]) == 3
    assert len({item["requirement_id"] for item in catalog["requirements"]}) == 3

    by_cap = {item["capability"]: item for item in catalog["requirements"]}
    gather = by_cap["resource.gathering"]
    trade = by_cap["economy.trade"]
    travel = by_cap["world.travel"]
    assert prompt[gather["source_span"]["char_start"]:gather["source_span"]["char_end"]] == "gather crystals"
    assert trade["depends_on"] == [gather["requirement_id"]]
    assert travel["depends_on"] == [trade["requirement_id"]]
    assert trade["acceptance"][0].startswith("Given ")
    assert "; when " in trade["acceptance"][0]
    assert "; then " in trade["acceptance"][0]


def test_authority_uses_typed_feedback_to_repair_missing_clause_coverage():
    prompt = "Players gather crystals. Players unlock a portal."
    incomplete = {
        "requirements": [
            _req("gather", "resource.gathering", 0, "gather crystals"),
        ]
    }
    repaired = {
        "requirements": [
            _req("gather", "resource.gathering", 0, "gather crystals"),
            _req("portal", "progression.portal", 1, "unlock a portal", depends_on=("gather",)),
        ]
    }
    router = _Router([incomplete, repaired])

    catalog = build_approved_requirement_catalog(prompt, router)

    assert len(router.calls) == 2
    second_payload = json.loads(router.calls[1]["messages"][1]["content"])
    diagnostic = second_payload["repair_diagnostic"]
    assert diagnostic["error_code"] == "REQ_SOURCE_COVERAGE"
    assert diagnostic["json_path"] == "$.requirements"
    assert diagnostic["repair_scope"] == "$.requirements"
    assert catalog["semantic_audit"]["unresolved_clause_count"] == 0


def test_authority_blocks_design_alternative_from_authored_requirement_phase():
    prompt = "Players can exchange gathered items."
    invalid = {
        "requirements": [
            _req(
                "shop",
                "ui.shop",
                0,
                "exchange gathered items",
                role="selected_design_alternative",
            )
        ]
    }
    router = _Router([invalid, invalid])

    with pytest.raises(EvidencePlanError, match="no-progress fixed point"):
        build_approved_requirement_catalog(prompt, router)

    second_payload = json.loads(router.calls[1]["messages"][1]["content"])
    assert (
        second_payload["repair_diagnostic"]["error_code"]
        == "REQ_PROVENANCE_OVERREACH"
    )


def test_authority_blocks_opaque_semantic_hash_from_becoming_a_capability():
    prompt = "Players can discover a hidden mechanic."
    invalid = {
        "requirements": [
            _req(
                "hidden",
                "semantic_13ee7693e9ed",
                0,
                "discover a hidden mechanic",
            )
        ]
    }
    router = _Router([invalid, invalid])

    with pytest.raises(EvidencePlanError, match="no-progress fixed point"):
        build_approved_requirement_catalog(prompt, router)

    second_payload = json.loads(router.calls[1]["messages"][1]["content"])
    assert second_payload["repair_diagnostic"]["error_code"] == "REQ_CAPABILITY_ID"


def test_authority_requires_derived_proof_and_known_graph_references():
    prompt = "Players can unlock advanced crafting."
    invalid = {
        "requirements": [
            _req(
                "craft",
                "crafting.advanced",
                0,
                "unlock advanced crafting",
                role="logically_derived",
                derived_from=("missing",),
                reason="Needed for the explicit goal.",
            )
        ]
    }
    router = _Router([invalid, invalid])

    with pytest.raises(EvidencePlanError):
        build_approved_requirement_catalog(prompt, router)
