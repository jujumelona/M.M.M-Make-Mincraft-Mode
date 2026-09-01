from __future__ import annotations

from minecraft_mod_ai import agentic_research_game_design as design_agent
from minecraft_mod_ai import minecraft_knowledge_nodes as knowledge
from minecraft_mod_ai import pre_design_domain_research
from minecraft_mod_ai import small_model_predesign_research as small


def test_canonical_predesign_path_bypasses_corrective_state_machine():
    assert pre_design_domain_research.research_document_domain.__module__ == "minecraft_mod_ai.pre_design_domain_research"
    assert callable(small.research_document_domain)


def test_irrelevant_page_never_becomes_blocking_gap():
    class Router:
        def generate_text(self, *args, **kwargs):
            return "NONE"

    class Project:
        @staticmethod
        def _read_evidence_pages(document):
            return [
                {
                    "page_ref": "host#1",
                    "content": "Microsoft Build Student Zone learning path unrelated material",
                }
            ]

        @staticmethod
        def _prompt_document_receipt(document):
            return {"page_count": 1}

    note = small.research_document_domain(
        object(),
        Project(),
        Router(),
        prompt="식민지화 우주 모드",
        domain={"domain_id": "request", "objective": "space colonization", "queries": []},
        document={"page_count": 1},
        trace_metadata=None,
    )
    assert note["sufficient"] is True
    assert note["fixed_point"] is False
    assert note["gaps"] == []
    assert note["research_evidence_status"] == "no_relevant_external_evidence"


def test_predesign_model_uses_plain_text_and_host_exact_quote():
    calls = []

    class Router:
        def generate_text(self, *args, **kwargs):
            calls.append(kwargs)
            return (
                "EVIDENCE\thost#1\tSpace stations can orbit planets."
                "\tUse an orbiting station abstraction."
            )

    claims, diagnostics, model_calls = small._extract_batch(
        Router(),
        domain={"objective": "space station", "queries": ["minecraft space station"]},
        pages=[
            {
                "page_ref": "host#1",
                "content": "Space stations can orbit planets. Other text.",
            }
        ],
    )
    assert diagnostics == []
    assert model_calls == 1
    assert claims and claims[0]["evidence_refs"] == ["host#1"]
    assert claims[0]["support_quote"] == "Space stations can orbit planets."
    assert calls[0]["response_format"] == "text"
    assert calls[0]["response_schema"] is None
    assert calls[0]["enable_tools"] is False


def test_advisory_empty_evidence_is_valid_host_state():
    design_agent._validate_sufficient_research(
        {
            "sufficient": True,
            "claims": [],
            "research_mode": "advisory_predesign",
            "research_evidence_status": "no_relevant_external_evidence",
        },
        allowed_refs=frozenset(),
    )


def test_stateful_space_request_activates_persistence_network_worldgen():
    plan = knowledge.compile_minecraft_knowledge_plan(
        "우주로 가서 다른 행성을 식민지화하고 특수 광물을 캐며 돈과 거래로 "
        "우주선과 선원을 업그레이드한다"
    )
    predicates = {item["predicate_id"]: item for item in plan["branch_predicates"]}
    assert predicates["needs_persistence"]["value"] is True
    assert predicates["needs_network"]["value"] is True
    assert predicates["needs_worldgen"]["value"] is True
