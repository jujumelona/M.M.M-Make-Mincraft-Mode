from types import SimpleNamespace

from minecraft_mod_ai import authored_scope_research_contract as scope
from minecraft_mod_ai import evidence_obligation_contract as obligations


PROMPT = "메이플 스토리 모드 만들어줘 잡몹부터 보스까지 템들 레벨도 점점 성장 강화시스템등 모두 구현해야해"


def _approved_catalog():
    rows = [
        ("req_mob", "mob.spawning", "잡몹", "Regular mobs spawn and participate in progression."),
        ("req_boss", "boss.entity", "보스", "Boss encounters culminate each progression tier."),
        ("req_item", "item.equipment", "템들", "Equipment items are obtainable across tiers."),
        ("req_level", "progression.level", "레벨도 점점 성장", "Player levels grow through gameplay progression."),
        ("req_upgrade", "item.upgrade", "강화시스템", "Equipment can be enhanced through an upgrade system."),
    ]
    requirements = []
    for requirement_id, capability, source_text, semantic in rows:
        start = PROMPT.index(source_text)
        requirements.append(
            {
                "requirement_id": requirement_id,
                "capability": capability,
                "semantic_statement": semantic,
                "statement": PROMPT,
                "source_span": {
                    "char_start": start,
                    "char_end": start + len(source_text),
                    "text": source_text,
                },
                "mandatory": True,
                "gameplay_capabilities": [capability],
                "provides": [f"capability:{capability}"],
            }
        )
    return {
        "schema_version": "mmm/approved-requirement-graph-v1",
        "requirements": requirements,
        "catalog_sha256": "fixture",
    }


def test_candidate_cannot_bypass_approved_atomic_obligation_dag():
    catalog = _approved_catalog()
    fake_obligations = SimpleNamespace(
        _catalog_for=lambda prompt: catalog if prompt == PROMPT else None,
        build_evidence_obligation_brief=lambda prompt, value, design: {
            "origin": "approved_requirement_graph",
            "catalog": value,
        },
    )
    legacy_called = False

    def legacy(prompt, design, candidate):
        nonlocal legacy_called
        legacy_called = True
        return {"origin": "legacy_whole_request"}

    result = scope._approved_research_normalize(
        fake_obligations,
        legacy,
        PROMPT,
        {},
        {"domains": [{"domain_id": "request", "requirements": [PROMPT]}]},
    )

    assert result["origin"] == "approved_requirement_graph"
    assert result["catalog"] is catalog
    assert not legacy_called


def test_maple_prompt_atomic_requirements_become_independent_research_obligations():
    brief = obligations.build_evidence_obligation_brief(
        PROMPT,
        _approved_catalog(),
        {},
    )

    assert brief["origin"] == "approved_requirement_graph"
    nodes = brief["evidence_obligation_dag"]["nodes"]
    requirement_ids = {node["requirement_id"] for node in nodes}
    assert requirement_ids == {
        "req_mob",
        "req_boss",
        "req_item",
        "req_level",
        "req_upgrade",
    }
    capabilities = {node["capability"] for node in nodes}
    assert {
        "mob.spawning",
        "boss.entity",
        "item.equipment",
        "progression.level",
        "item.upgrade",
    } <= capabilities
    assert len(brief["domains"]) > len(requirement_ids)
    assert all(
        domain["requirements"] != [PROMPT]
        for domain in brief["domains"]
    )


def test_knowledge_plan_reuses_active_catalog_instead_of_router_none_rebuild(monkeypatch):
    catalog = _approved_catalog()
    monkeypatch.setattr(scope, "_active_catalog", lambda prompt: catalog)
    validated = []

    class Nodes:
        @staticmethod
        def _sha(value):
            return "sha256:test"

    knowledge = SimpleNamespace(
        _base_compile_minecraft_knowledge_plan=lambda prompt, design: {
            "policy": {},
            "plan_sha256": "old",
        },
        _authored_requirement_lifecycle=lambda value: [
            {"requirement_id": item["requirement_id"], "state": "PRESERVED_FOR_RESEARCH"}
            for item in value["requirements"]
        ],
        _nodes=Nodes,
        validate_plan=lambda plan: validated.append(plan),
    )
    previous_called = False

    def previous(prompt, design=None):
        nonlocal previous_called
        previous_called = True
        return {"authored_request_catalog": {"requirements": [{"capability": "semantic_deadbeef"}]}}

    plan = scope._compile_knowledge_plan_with_active_catalog(
        knowledge,
        previous,
        PROMPT,
        {},
    )

    assert not previous_called
    assert plan["authored_request_catalog"] == catalog
    assert plan["authored_request_catalog"] is not catalog
    assert len(plan["authored_requirements"]) == 5
    assert plan["policy"]["catalog_rebuild_after_freeze"] is False
    assert plan["policy"]["authored_requirement_routing_owner"] == "approved_requirement_graph"
    assert {route["capability"] for route in plan["authored_capability_routes"]} == {
        "mob.spawning",
        "boss.entity",
        "item.equipment",
        "progression.level",
        "item.upgrade",
    }
    assert validated and validated[-1] is plan
