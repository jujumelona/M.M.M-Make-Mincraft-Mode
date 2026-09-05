from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import evidence_first_planning
from minecraft_mod_ai.implementation_template_contract import build_implementation_template
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


def test_coder_capsule_preserves_only_sanitized_planner_hole_fills() -> None:
    task = {
        "task_id": "task_alpha",
        "semantic_outcome": "Implement alpha gameplay behavior",
        "requirement_refs": ["req_alpha_123"],
        "implementation_capabilities": ["gameplay.alpha.service"],
        "target_cell": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
        },
    }
    template = build_implementation_template(task)
    hole_id = template["holes"][0]["hole_id"]
    module = SimpleNamespace(
        module_id="task_alpha",
        kind="custom_java",
        config={
            "evidence_task": task,
            "model_fill": {
                "hole_fills": [
                    {
                        "hole_id": hole_id,
                        "implementation_decision": "Use the host-owned alpha service contract.",
                        "local_steps": ["Implement the bounded task-local behavior."],
                        "required_gates": ["model_must_not_author_this"],
                    },
                    {
                        "hole_id": "hole_fake_not_host_owned",
                        "implementation_decision": "unauthorized",
                    },
                ]
            },
        },
        depends_on=(),
        required_gates=(),
    )

    contract = compact_task_local_module_contract(module)
    compact = contract["evidence_task"]

    assert compact["implementation_template"]["template_sha256"] == template["template_sha256"]
    assert compact["planner_fill"]["hole_fills"] == [
        {
            "hole_id": hole_id,
            "implementation_decision": "Use the host-owned alpha service contract.",
            "local_steps": ["Implement the bounded task-local behavior."],
        }
    ]


def test_semantic_fallback_preserves_every_explicit_capability(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_first_planning,
        "resolve_capabilities_from_phrase_structured",
        lambda _clause: SimpleNamespace(
            nodes=[
                SimpleNamespace(capability_id="economy.trade", origin="explicit"),
                SimpleNamespace(capability_id="resource.mining", origin="explicit"),
            ]
        ),
    )

    ir = evidence_first_planning._stub_semantic_model("x", 0, 1, "x")

    assert ir.gameplay_capability_candidates == (
        "economy.trade",
        "resource.mining",
    )
    variants = evidence_first_planning._semantic_ir_variants(ir)
    assert [variant.gameplay_capability_candidates for variant in variants] == [
        ("economy.trade",),
        ("resource.mining",),
    ]
