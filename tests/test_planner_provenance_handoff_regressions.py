from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.research_requirement_evidence import facet_relevant_refs
from minecraft_mod_ai.research_requirement_schema import FACETS, FACET_HINTS
from minecraft_mod_ai.small_model_task_capsule_contract import (
    compact_task_local_module_contract,
)


def test_facet_evidence_requires_requirement_provenance() -> None:
    facet = FACETS[0]
    hint = FACET_HINTS[facet][0]
    requirement = {
        "requirement_id": "req_alpha_123",
        "capability": "gameplay.alpha",
        "statement": "Provide the alpha gameplay mechanic.",
        "provides": ["capability:gameplay.alpha"],
        "gameplay_capabilities": ["gameplay.alpha"],
        "implementation_capabilities": [],
    }
    evidence = [
        {
            "evidence_ref": "evidence:unrelated",
            "summary": {"claim": f"{hint} implementation guidance"},
        },
        {
            "evidence_ref": "evidence:bound",
            "summary": {
                "requirement_ref": "req_alpha_123",
                "claim": f"{hint} implementation guidance",
            },
        },
    ]

    refs = facet_relevant_refs(evidence, requirement, {})

    assert "evidence:bound" in refs[facet]
    assert "evidence:unrelated" not in {
        ref for facet_refs in refs.values() for ref in facet_refs
    }


def test_coder_capsule_uses_only_host_compiled_execution_authority() -> None:
    anchor = {
        "kind": "symbol",
        "locator": "src/main/java/generated/mod/AlphaService.java#AlphaService",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    task = {
        "task_id": "task_alpha",
        "semantic_outcome": "Implement alpha gameplay behavior",
        "requirement_refs": ["req_alpha_123"],
        "implementation_capabilities": ["gameplay.alpha.service"],
        "implementation_obligations": [
            "Implement the host-owned alpha service behavior at the reserved symbol."
        ],
        "owned_anchors": [anchor],
        "production_bindings": [
            {
                "task_ref": "task_alpha",
                "reuse_action": "fresh",
                "owned_anchors": [anchor],
            }
        ],
        "target_cell": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "mojang",
            "java_version": "21",
        },
    }
    module = SimpleNamespace(
        module_id="task_alpha",
        kind="custom_java",
        config={
            "evidence_task": task,
            "model_fill": {
                "hole_fills": [
                    {
                        "hole_id": "retired_model_owned_hole",
                        "implementation_decision": "unauthorized",
                        "required_gates": ["model_must_not_author_this"],
                    }
                ]
            },
        },
        depends_on=(),
        required_gates=(),
    )

    contract = compact_task_local_module_contract(module)
    compact = contract["evidence_task"]
    execution = compact["coder_execution_contract"]

    assert execution["schema_version"] == "mmm/coder-execution-contract-v2"
    assert execution["task_ref"] == "task_alpha"
    assert execution["target_constraints"]["minecraft_version"] == "1.21.1"
    assert execution["target_constraints"]["mappings"] == "mojang"
    assert execution["targets"][0]["locator"] == anchor["locator"]
    assert execution["implementation_steps"][0]["obligation"] == task["implementation_obligations"][0]
    assert "planner_fill" not in compact
    assert "implementation_template" not in compact
    assert "model_fill" not in contract
    assert "model_fill" not in compact
