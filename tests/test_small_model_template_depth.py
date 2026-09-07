from __future__ import annotations

from minecraft_mod_ai.implementation_template_contract import build_implementation_template
from minecraft_mod_ai.research_requirement_template import build_facet_slot


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
