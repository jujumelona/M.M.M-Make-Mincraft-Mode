from __future__ import annotations

import json
from typing import Any

import pytest

from minecraft_mod_ai.planning_state_contract import _sha
from minecraft_mod_ai.planning_state_resolution import (
    _fit_context_to_budget,
    _rehash,
    _resolved_context,
    compile_researched_requirements,
)
from minecraft_mod_ai.pre_design_domain_research import (
    _MAX_DOMAIN_EVIDENCE_CARDS,
    _MAX_EXCERPT_CHARS,
    _exact_excerpt,
    _grounded_evidence_cards,
)


def test_exact_excerpt_bounds_length_and_guarantees_exact_substring() -> None:
    large_paragraph = (
        "Introduction to energy systems. "
        + ("Padding background information that does not match. " * 30)
        + "Persistent energy storage is owned by the server and survives reload. "
        + ("Trailing documentation prose and irrelevant details. " * 30)
    )
    wanted = {"energy", "storage"}
    excerpt, score = _exact_excerpt(large_paragraph, wanted)

    assert score > 0
    assert len(excerpt) <= _MAX_EXCERPT_CHARS
    assert excerpt in large_paragraph
    assert "energy" in excerpt.casefold() or "storage" in excerpt.casefold()


def test_exact_excerpt_prefers_concise_term_dense_chunk_over_huge_blob() -> None:
    concise_chunk = "Energy storage is managed on the server."
    huge_blob = "Energy storage " + ("filler " * 1000)

    content = f"{huge_blob}\n\n{concise_chunk}"
    wanted = {"energy", "storage"}
    excerpt, score = _exact_excerpt(content, wanted)

    assert score == 2
    assert excerpt == concise_chunk


def test_grounded_evidence_cards_caps_top_k_distinct_salient_cards() -> None:
    pages = []
    for i in range(12):
        source = {
            "source_id": f"source:{i}",
            "source_type": "test_page",
            "url": f"https://example.invalid/{i}",
            "title": f"Page {i}",
            "content_sha256": f"sha256:{i}",
            "content": f"Energy storage module variation {i} provides power transfer capabilities.",
        }
        pages.append(
            {
                "page_ref": f"sha256:doc#page={i}/12",
                "content": json.dumps(source),
            }
        )

    class _MockRag:
        def _read_evidence_pages(self, _doc):
            return list(pages)

    domain = {
        "domain_id": "r_001",
        "objective": "Energy storage research",
        "requirements": ["energy storage"],
    }
    document = {"page_count": len(pages)}

    cards = _grounded_evidence_cards(_MockRag(), document, domain)

    assert len(cards) <= _MAX_DOMAIN_EVIDENCE_CARDS
    assert len(cards) == 4
    for card in cards:
        assert card["domain_term_overlap"] > 0
        assert len(card["exact_excerpt"]) <= _MAX_EXCERPT_CHARS
        assert card["verification"] == "host_exact_substring_from_materialized_source_page"


def test_resolved_context_formats_clean_qa_without_python_repr() -> None:
    state = {
        "goal": {"statement": "Build copper pipes"},
        "known": [{"statement": "Connects to steam boilers"}],
        "unresolved": [
            {
                "unresolved_id": "u_001",
                "question": "How to handle fluid transfer?",
            }
        ],
        "resolved": [
            {
                "unresolved_id": "u_001",
                "resolution": [{"claim": "Fluid transfer is tick-rate limited"}],
            }
        ],
        "evidence": [
            {
                "research_ref": "r_001",
                "claims": [
                    {"claim": "Fluid transfer is tick-rate limited"},
                    {"claim": "Copper pipes oxidize over time"},
                ],
            }
        ],
    }

    context = _resolved_context(state)

    assert context["goal"] == "Build copper pipes"
    assert context["known"] == [{"statement": "Connects to steam boilers"}]
    assert len(context["resolved"]) == 1
    assert context["resolved"][0]["question"] == "How to handle fluid transfer?"
    assert context["resolved"][0]["resolution"] == "Fluid transfer is tick-rate limited"
    assert "[{'claim'" not in json.dumps(context)
    assert context["research_claims"] == ["Copper pipes oxidize over time"]


class _RecordingRouter:
    def __init__(self):
        self.recorded_messages = None

    def generate_tool_decision(self, role, messages, **kwargs):
        self.recorded_messages = messages
        return {
            "requirements": [
                {
                    "statement": "Player can craft and place copper pipes",
                    "semantic_capability": "block_placement",
                    "acceptance": ["Pipes place and connect correctly"],
                }
            ]
        }


def test_compile_researched_requirements_fits_context_and_emits_valid_json() -> None:
    prompt = "Build a comprehensive magic wand and spell system"
    state = {
        "schema_version": "mmm/planning-state-v1",
        "original_prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "state_sha256": "",
        "plan_ready": False,
        "goal": {"statement": "Magic wand system"},
        "known": [{"known_id": "k_001", "statement": "Wand fires elemental bolts"}],
        "references": [],
        "scope_status": "unspecified",
        "unresolved": [
            {
                "unresolved_id": f"u_{i:03d}",
                "question": f"Question {i}?",
                "status": "resolved",
                "reason": "external_fact",
                "resolution_route": "external_research",
                "source_kinds": ["web_sources"],
                "information_needed": f"Details about question {i}",
                "research_ref": f"r_{i:03d}",
                "blocks": [],
            }
            for i in range(1, 20)
        ],
        "research_queue": [
            {
                "research_id": f"r_{i:03d}",
                "resolves": [f"u_{i:03d}"],
                "status": "complete",
                "information_needed": f"Details for {i}",
                "objective": f"Objective for {i}",
                "queries": [],
                "source_kinds": ["web_sources"],
            }
            for i in range(1, 20)
        ],
        "evidence": [
            {
                "research_ref": f"r_{i:03d}",
                "source": "grounded_materialized_pages",
                "claims": [{"claim": f"Evidence fact {i} " + ("detail " * 20)}],
                "evidence_refs": [f"source:ref_{i}"],
                "sufficient": True,
            }
            for i in range(1, 20)
        ],
        "resolved": [
            {
                "unresolved_id": f"u_{i:03d}",
                "resolution": f"Resolution fact {i} " + ("detail " * 20),
                "basis": "grounded_research",
                "evidence_refs": [f"source:ref_{i}"],
            }
            for i in range(1, 20)
        ],
        "decisions": [],
        "implementation_candidates": [],
        "coverage": [],
        "blockers": [],
    }
    state = _rehash(state)

    router = _RecordingRouter()
    updated = compile_researched_requirements(router, prompt, state)

    assert router.recorded_messages is not None
    user_msg = next(m for m in router.recorded_messages if m["role"] == "user")
    parsed_payload = json.loads(user_msg["content"])
    assert "task" in parsed_payload
    assert "goal" in parsed_payload["task"]
    assert parsed_payload["task"]["goal"] == "Magic wand system"
    assert len(updated["decisions"]) >= 1
