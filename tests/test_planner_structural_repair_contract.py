from __future__ import annotations

from typing import Any

from minecraft_mod_ai.evidence_first_planning import compile_evidence_first_plan
from minecraft_mod_ai.minecraft_template_catalog import profile_for_capability
from minecraft_mod_ai.reuse_planner import decompose_capability_graph
from minecraft_mod_ai.semantic_batching_contract import build_bounded_requirement_catalog


class _SemanticRouter:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)

    def generate_tool_decision(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if not self.outputs:
            raise AssertionError("unexpected semantic model call")
        return self.outputs.pop(0)


def _segmentation(index: int, anchor: str) -> dict[str, Any]:
    return {
        "leaves": [
            {
                "source_clause_index": index,
                "source_anchor": anchor,
                "semantic_statement": anchor,
                "given": "the authored precondition holds",
                "when": "the authored action occurs",
                "then": "the authored outcome is observed",
                "semantic_type": "gameplay_mechanic",
            }
        ]
    }


def _classification(capability: str) -> dict[str, Any]:
    return {
        "classifications": [
            {
                "leaf_index": 0,
                "capability_id": capability,
            }
        ]
    }


def _two_stage_outputs(
    capabilities: tuple[str, ...],
    anchors: tuple[str, ...],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index, (capability, anchor) in enumerate(zip(capabilities, anchors, strict=True)):
        outputs.append(_segmentation(index, anchor))
        outputs.append(_classification(capability))
    return outputs


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
    router = _SemanticRouter(_two_stage_outputs(capabilities, anchors))
    catalog = build_bounded_requirement_catalog(prompt, router=router)
    assert {item["capability"] for item in catalog["requirements"]} == set(capabilities)
    assert all(len(item["provides"]) == 1 for item in catalog["requirements"])
    assert catalog["semantic_audit"]["semantic_model_calls_total_observed"] == 10


def test_frozen_catalog_compiles_a_task_chain_for_every_root() -> None:
    prompt = "Spawn hostile mobs.\nAdd a boss entity.\nAdd equipment."
    capabilities = ("mob.spawning", "boss.entity", "item.equipment")
    anchors = ("Spawn hostile mobs", "Add a boss entity", "Add equipment")
    catalog = build_bounded_requirement_catalog(
        prompt,
        router=_SemanticRouter(_two_stage_outputs(capabilities, anchors)),
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
