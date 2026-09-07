from __future__ import annotations

import pytest

from minecraft_mod_ai.implementation_template_contract import build_implementation_template
from minecraft_mod_ai.planning_detail_template import (
    DETAIL_FIELDS,
    DETAIL_SLOT_GUIDANCE,
    WORKSHEET_SCHEMA,
    validate_worksheet,
    worksheet_prompt,
)
from minecraft_mod_ai.planning_state_implementation import _compile_requirement_plan
from minecraft_mod_ai.research_requirement_template import build_facet_slot


def _worksheet() -> dict[str, object]:
    return {
        key: {
            "specification": (
                f"{key} section defines concrete evidence-backed implementation behavior "
                "with an explicit owner, condition, boundary, and observable result."
            ),
            "constraint_evidence_refs": ["ev:1"],
        }
        for key in DETAIL_FIELDS
    }


def test_engineering_worksheet_exposes_deep_small_model_checklists() -> None:
    assert len(DETAIL_FIELDS) == 10
    assert set(DETAIL_SLOT_GUIDANCE) == set(DETAIL_FIELDS)
    assert all(len(items) >= 7 for items in DETAIL_SLOT_GUIDANCE.values())
    assert set(WORKSHEET_SCHEMA["required"]) == set(DETAIL_FIELDS)
    for key in DETAIL_FIELDS:
        section = WORKSHEET_SCHEMA["properties"][key]
        assert section["properties"]["specification"]["minLength"] >= 24
        assert section["properties"]["constraint_evidence_refs"]["uniqueItems"] is True
        for concern in DETAIL_SLOT_GUIDANCE[key]:
            assert concern in section["description"]
    prompt = worksheet_prompt()
    assert "mandatory completion protocol" in prompt
    assert "Never use a bare N/A" in prompt
    assert "distinct section-specific specification" in prompt
    assert "cross-check" in prompt


def test_engineering_worksheet_fails_closed_on_shallow_duplicate_or_forged_fills() -> None:
    valid = _worksheet()
    assert validate_worksheet(valid, {"ev:1"}) == valid

    placeholder = _worksheet()
    placeholder["persistence"] = {
        "specification": "N/A",
        "constraint_evidence_refs": ["ev:1"],
    }
    with pytest.raises(ValueError, match="persistence has no concrete specification"):
        validate_worksheet(placeholder, {"ev:1"})

    shallow = _worksheet()
    shallow["algorithm"] = {
        "specification": "Handle it correctly.",
        "constraint_evidence_refs": ["ev:1"],
    }
    with pytest.raises(ValueError, match="algorithm has no concrete specification"):
        validate_worksheet(shallow, {"ev:1"})

    copied = _worksheet()
    copied["state_model"] = dict(copied["behavior_contract"])
    with pytest.raises(ValueError, match="state_model duplicates behavior_contract"):
        validate_worksheet(copied, {"ev:1"})

    forged = _worksheet()
    forged["algorithm"] = {
        "specification": "Concrete ordered algorithm with branches and observable result.",
        "constraint_evidence_refs": ["ev:forged"],
    }
    with pytest.raises(ValueError, match="algorithm has invalid constraint evidence"):
        validate_worksheet(forged, {"ev:1"})

    duplicate_ref = _worksheet()
    duplicate_ref["verification"] = {
        "specification": "Given a valid state, when behavior runs, then the expected result is observable.",
        "constraint_evidence_refs": ["ev:1", "ev:1"],
    }
    with pytest.raises(ValueError, match="verification has invalid constraint evidence"):
        validate_worksheet(duplicate_ref, {"ev:1"})


def test_detailed_planner_receives_one_bounded_plain_text_slot_per_section() -> None:
    class Router:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def generate_text(
            self,
            role: str,
            messages: list[dict[str, str]],
            **kwargs: object,
        ) -> str:
            assert role == "planner"
            assert kwargs["response_format"] == "text"
            assert kwargs["enable_tools"] is False
            self.calls.append([dict(message) for message in messages])
            section = next(
                line.removeprefix("Section: ")
                for line in messages[1]["content"].splitlines()
                if line.startswith("Section: ")
            )
            return (
                f"{section} uses one authoritative owner with explicit state transitions, "
                "bounded failure behavior, deterministic limits, and observable verification outcomes."
            )

    router = Router()
    state = {
        "research_queue": [
            {
                "research_id": "research_001",
                "requirement_ref": "req_001",
                "status": "complete",
            }
        ],
        "evidence": [
            {
                "research_ref": "research_001",
                "sufficient": True,
                "claims": ["Persistent state requires an explicit authoritative lifecycle."],
                "evidence_refs": ["ev:1"],
            }
        ],
    }
    plan = _compile_requirement_plan(
        router,
        state,
        {
            "requirement_id": "req_001",
            "statement": "Persist the gameplay state across reload.",
        },
    )

    assert len(router.calls) == len(DETAIL_FIELDS)
    seen_sections: list[str] = []
    all_prompt_text: list[str] = []
    for messages in router.calls:
        system_text = messages[0]["content"]
        user_text = messages[1]["content"]
        all_prompt_text.extend((system_text, user_text))
        assert "plain prose only" in system_text
        assert "no JSON" in system_text
        assert "Persist the gameplay state across reload." in user_text
        assert user_text.count("Section: ") == 1
        section = next(
            line.removeprefix("Section: ")
            for line in user_text.splitlines()
            if line.startswith("Section: ")
        )
        seen_sections.append(section)
        assert section in DETAIL_FIELDS
        assert all(concern in user_text for concern in DETAIL_SLOT_GUIDANCE[section])

    joined = "\n".join(all_prompt_text)
    assert tuple(seen_sections) == tuple(DETAIL_FIELDS)
    assert "req_001" not in joined
    assert "ev:1" not in joined
    assert tuple(plan["engineering_worksheet"]) == tuple(DETAIL_FIELDS)
    assert all(
        row["constraint_evidence_refs"] == []
        for row in plan["engineering_worksheet"].values()
    )


def test_research_facet_slot_carries_explicit_review_method() -> None:
    slot = build_facet_slot(
        planning_context={"authority": "host_only"},
        requirement={
            "requirement_id": "req_001",
            "capability": "network.transaction",
            "statement": "Synchronize the transaction safely.",
            "implementation_capabilities": ["network.server_authority"],
            "artifact_obligations": [],
            "acceptance": ["The server rejects invalid transactions."],
        },
        facet="server_network_authority",
        baseline={},
        task_slice=[],
        evidence_catalog=[{"evidence_ref": "ev:1"}],
        allowed_evidence_refs=["ev:1"],
    )
    model_slot = slot["model_slot"]
    assert len(model_slot["rules"]) >= 12
    assert "authoritative side" in model_slot["question"]
    assert "join/reconnect/resync" in model_slot["question"]
    assert any("Never invent" in rule for rule in model_slot["rules"])
    assert any("observable and falsifiable" in rule for rule in model_slot["rules"])


def test_coder_execution_steps_include_small_model_edit_and_cleanup_protocol() -> None:
    task = {
        "task_id": "core_runtime_api",
        "task_sha256": "sha256:" + "a" * 64,
        "semantic_outcome": "The runtime behavior is observable in Minecraft.",
        "requirement_refs": ["req_runtime"],
        "depends_on": [],
        "consumes": [],
        "provides": ["runtime_done"],
        "target_cell": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "yarn",
            "java_version": "21",
        },
        "implementation_obligations": [
            "Bind the researched runtime behavior to the owned CoreRuntime symbol."
        ],
        "required_gates": ["source_static_validation", "target_compile"],
        "public_acceptance": ["The runtime behavior is observable."],
        "owned_anchors": [
            {
                "kind": "symbol",
                "locator": "src/main/java/demo/CoreRuntime.java#CoreRuntime",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "main",
            }
        ],
    }
    contract = build_implementation_template(task)
    assert contract["implementation_steps"]
    for step in contract["implementation_steps"]:
        assert len(step["execution_checklist"]) >= 8
        assert "no obsolete/placeholder" in step["done_when"]
        assert any(
            "Inspect the existing owned target" in item
            for item in step["execution_checklist"]
        )
        assert any(
            "Do not leave TODO" in item for item in step["execution_checklist"]
        )
    assert "cleanup_rule" in contract["protected_boundaries"]
    assert any(
        "no TODO/FIXME/stub" in item
        for item in contract["completion_predicate"]["conditions"]
    )
