from __future__ import annotations

import json

from minecraft_mod_ai.planning_contract_ssot import MODEL_UNRESOLVED_REASONS
from minecraft_mod_ai.planning_state_research import (
    _providers_for,
    _reference_names_for_research,
)
from minecraft_mod_ai.pre_design_domain_research import _grounded_evidence_cards


class _FakeProjectRag:
    def __init__(self, *, title: str, url: str, source_id: str, content: str) -> None:
        self._page = {
            "page_ref": "page://fixture",
            "content": json.dumps(
                {
                    "source_id": source_id,
                    "source_type": "wikipedia",
                    "url": url,
                    "title": title,
                    "content_sha256": "sha256:fixture",
                    "content": content,
                }
            ),
        }

    def _read_evidence_pages(self, _document):
        return [self._page]


def _maplestory_domain() -> dict[str, object]:
    return {
        "domain_id": "r_001",
        "objective": "What documented systems, rules, and behavior define MapleStory?",
        "requirements": ["MapleStory gameplay rules and progression"],
        "queries": ["MapleStory documented systems behavior rules"],
        "required_anchor_terms": ["MapleStory"],
    }


def test_prompt_model_cannot_invent_external_fact_unknowns() -> None:
    assert "external_fact" not in MODEL_UNRESOLVED_REASONS


def test_generic_web_route_never_falls_back_to_reference_or_minecraft_providers() -> None:
    assert _providers_for(["web_sources"]) == []
    assert _providers_for(["reference_sources", "web_sources"]) == ["wikipedia"]


def test_reference_query_uses_only_reference_named_by_that_research_item() -> None:
    state = {
        "references": [
            {"name": "MapleStory"},
            {"name": "Terraria"},
        ]
    }
    research = {
        "objective": "What documented systems define MapleStory?",
        "information_needed": "MapleStory gameplay rules and progression",
    }

    assert _reference_names_for_research(state, research) == ["MapleStory"]


def test_unrelated_davinci_page_is_rejected_even_with_generic_semantic_overlap() -> None:
    rag = _FakeProjectRag(
        title="DaVinci Resolve",
        url="https://en.wikipedia.org/wiki/DaVinci_Resolve",
        source_id="wikipedia:DaVinci_Resolve",
        content=(
            "This platform has a system with progression, player workflows, rules, "
            "and game-related terminology that overlaps the research query."
        ),
    )

    cards = _grounded_evidence_cards(rag, {"pages": []}, _maplestory_domain())

    assert cards == []


def test_named_reference_source_passes_identity_gate_and_keeps_exact_evidence() -> None:
    source_text = (
        "MapleStory progression is organized around character levels, quests, and class "
        "advancement systems documented for the game."
    )
    rag = _FakeProjectRag(
        title="MapleStory",
        url="https://en.wikipedia.org/wiki/MapleStory",
        source_id="wikipedia:MapleStory",
        content=source_text,
    )

    cards = _grounded_evidence_cards(rag, {"pages": []}, _maplestory_domain())

    assert len(cards) == 1
    assert cards[0]["required_identity_anchor"] == "MapleStory"
    assert cards[0]["exact_excerpt"] == source_text
