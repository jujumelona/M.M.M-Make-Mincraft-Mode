from __future__ import annotations

import json
from typing import Any

from minecraft_mod_ai.evidence_first_planning import compile_evidence_first_plan
from minecraft_mod_ai.minecraft_template_catalog import profile_for_capability
from minecraft_mod_ai.reuse_planner import decompose_capability_graph
from minecraft_mod_ai.semantic_batching_contract import build_bounded_requirement_catalog


class _SemanticRouter:
    def __init__(self, capabilities_by_anchor: dict[str, str]) -> None:
        self.capabilities_by_anchor = dict(capabilities_by_anchor)
        self.calls = 0

    def generate_tool_decision(self, role, messages, **kwargs):  # noqa: ANN001, ANN003
        assert role == "planner"
        assert kwargs["tool_name"] == "compile_semantic_requirements"
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        requirements = []
        for clause in payload["host_owned_clauses"]:
            anchor = str(clause["text"]).strip().rstrip(".")
            capability = self.capabilities_by_anchor[anchor]
            requirements.append(
                {
                    "source_clause_index": int(clause["source_clause_index"]),
                    "capability_id": capability,
                    "source_anchor": anchor,
                    "semantic_statement": anchor,
                    "given": "the authored precondition holds",
                    "when": "the authored action occurs",
                    "then": "the authored outcome is observed",
                    "semantic_type": "gameplay_mechanic",
                }
            )
        return {"requirements": requirements}


def _router(capabilities: tuple[str, ...], anchors: tuple[str, ...]) -> _SemanticRouter:
    return _SemanticRouter(dict(zip(anchors, capabilities, strict=True)))


def test_unknown_capability_keeps_host_custom_template_identity() -> None:
    profile = profile_for_capability("custom.semantic_deadbeefcafebabe")
    assert profile.template_id == "custom_gameplay"
    assert profile.implementation_capabilities


def test_bounded_semantic_catalog_preserves_every_authored_leaf() -> None:
    prompt = (
        "Spawn hostile mobs.\n"
        "Add a boss entity.\n"
        "Add equipment.\n"
        "Add progression levels.\n"
        "Add item upgrades."
    )
    capabilities = (
        "mob.spawning",
        "boss.entity",
        "item.equipment",
        "progression.level",
        "item.upgrade",
    )
    anchors = (
        "Spawn hostile mobs",
        "Add a boss entity",
        "Add equipment",
        "Add progression levels",
        "Add item upgrades",
    )
    router = _router(capabilities, anchors)
    catalog = build_bounded_requirement_catalog(prompt, router=router)

    assert {item["capability"] for item in catalog["requirements"]} == set(capabilities)
    assert all(len(item["provides"]) == 1 for item in catalog["requirements"])
    assert catalog["semantic_audit"]["semantic_model_calls_total_observed"] == 5
    assert catalog["semantic_audit"]["semantic_repair_turns_used"] == 0
    assert router.calls == 5


def test_frozen_catalog_compiles_a_task_chain_for_every_root() -> None:
    prompt = "Spawn hostile mobs.\nAdd a boss entity.\nAdd equipment."
    capabilities = ("mob.spawning", "boss.entity", "item.equipment")
    anchors = ("Spawn hostile mobs", "Add a boss entity", "Add equipment")
    catalog = build_bounded_requirement_catalog(
        prompt,
        router=_router(capabilities, anchors),
    )
    plan = compile_evidence_first_plan(
        prompt,
        {"_evidence_request_catalog": catalog},
        target_decision={
            "target": {
                "minecraft_version": "1.21.1",
                "loader": "neoforge",
                "source_api_family": "neoforge",
            }
        },
    )
    assert set(capabilities) <= {gap["capability"] for gap in plan["gap_catalog"]}
    task_refs = {ref for task in plan["tasks"] for ref in task["requirement_refs"]}
    assert task_refs == {
        requirement["requirement_id"]
        for requirement in plan["request_catalog"]["requirements"]
    }
    assert all(task["template_id"] != "semantic_implementation" for task in plan["tasks"])


def test_prompt_unknown_is_one_opaque_but_design_scope_does_not_inflate() -> None:
    graph = decompose_capability_graph("Add seasonal rune banking.")
    opaque = [node for node in graph.nodes if node.startswith("provisional:")]
    assert len(opaque) == 1
    design = {"capabilities": [f"system.feature_{index}" for index in range(96)]}
    scoped = decompose_capability_graph("Implement the declared systems.", design=design)
    assert len(scoped.nodes) == 96
    assert not any(node.startswith("provisional:") for node in scoped.nodes)
