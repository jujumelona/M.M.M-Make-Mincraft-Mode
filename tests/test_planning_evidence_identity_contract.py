from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_detail_contract import validate_evidence_refs
from minecraft_mod_ai.planning_state_contract import (
    SCHEMA,
    _hash_without,
    _sha,
    validate_planning_state,
)


def _state_with_atomic_evidence() -> dict:
    prompt = "Build an economy-backed modular spaceship progression system."
    source_kinds = [
        "repository",
        "existing_mods",
        "minecraft_docs",
        "minecraft_source",
        "project_rag",
    ]
    state = {
        "schema_version": SCHEMA,
        "original_prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "goal": {"statement": "Build the requested progression system."},
        "known": [
            {
                "known_id": "known_001",
                "statement": "The spaceship is assembled and upgraded through progression.",
            }
        ],
        "references": [],
        "scope_status": "explicit",
        "unresolved": [
            {
                "unresolved_id": "u_001",
                "question": "How is the requirement implemented?",
                "reason": "implementation_method",
                "blocks": ["implementation_plan"],
                "information_needed": "Verified implementation evidence.",
                "resolution_route": "implementation_research",
                "source_kinds": source_kinds,
                "status": "resolved",
                "research_ref": "r_001",
            }
        ],
        "research_queue": [
            {
                "research_id": "r_001",
                "resolves": ["u_001"],
                "objective": "Find a grounded implementation pattern.",
                "information_needed": "Verified implementation evidence.",
                "source_kinds": source_kinds,
                "queries": ["Minecraft implementation pattern"],
                "status": "complete",
            }
        ],
        "evidence": [
            {
                "research_ref": "r_001",
                "claims": [{"fact": "Verified fact", "evidence_refs": ["source:atomic-001"]}],
                "evidence_refs": ["source:atomic-001"],
                "sufficient": True,
                "source": "grounded_materialized_pages",
            }
        ],
        "resolved": [
            {
                "unresolved_id": "u_001",
                "resolution": [{"fact": "Verified fact"}],
                "basis": "grounded_research",
                "evidence_refs": ["source:atomic-001"],
            }
        ],
        "decisions": [],
        "implementation_candidates": [],
        "coverage": [],
        "blockers": [],
        "plan_ready": False,
        "state_sha256": "",
    }
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def test_planning_state_accepts_evidence_rows_without_container_ids() -> None:
    state = _state_with_atomic_evidence()

    validate_planning_state(state, prompt=state["original_prompt"])

    assert "evidence_id" not in state["evidence"][0]
    assert state["evidence"][0]["evidence_refs"] == ["source:atomic-001"]


def test_container_style_evidence_id_is_not_an_allowed_atomic_reference() -> None:
    with pytest.raises(ValueError, match="unknown evidence refs: e_001"):
        validate_evidence_refs(
            ["e_001"],
            {"source:atomic-001"},
            field="regression",
            require=True,
        )
