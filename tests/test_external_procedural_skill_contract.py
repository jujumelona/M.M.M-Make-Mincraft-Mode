from __future__ import annotations

from minecraft_mod_ai.external_procedural_skill_contract import (
    _compile_skillbank,
    _consolidate_skills,
    _sanitize_procedure,
    _select_skills,
)


def _procedure(name: str, activation: str, step: str, ref: str, confidence: float = 0.9):
    return {
        "name": name,
        "activate_when": [activation],
        "contraindications": [],
        "steps": [step],
        "constraints": ["verify current version"],
        "output_contract": "validated change",
        "evidence_refs": [ref],
        "confidence": confidence,
    }


def test_external_procedure_requires_evidence() -> None:
    raw = _procedure(
        "repair registry",
        "registry compile error",
        "inspect registry API",
        "ref:1",
    )
    raw["evidence_refs"] = []
    assert _sanitize_procedure(raw, "fabric_api") is None


def test_skillbank_compiles_relations_and_retrieves_relevant_skill() -> None:
    bank = _compile_skillbank(
        [
            {
                "domain_id": "fabric_api",
                "procedures": [
                    _procedure(
                        "repair registry mapping",
                        "registry compile error",
                        "inspect exact mapping before patch",
                        "ref:registry",
                    ),
                    _procedure(
                        "verify registry repair",
                        "registry build failure",
                        "run compiler and tests",
                        "ref:test",
                        0.8,
                    ),
                ],
            }
        ]
    )
    assert bank["schema_version"] == "mmm/external-procedural-skillbank-v1"
    assert len(bank["skills"]) == 2
    assert bank["consolidated_skill_count"] == 0
    assert bank["relation_graph"]
    selected = _select_skills("fix registry compile error", bank["skills"], limit=1)
    assert len(selected) == 1
    assert "registry" in selected[0]["name"]


def test_consolidation_keeps_only_shared_evidence_backed_prefix() -> None:
    first = _sanitize_procedure(
        {
            "name": "repair registry mapping",
            "activate_when": ["registry compile error"],
            "contraindications": ["do not patch unknown loader versions"],
            "steps": [
                "inspect exact registry mapping",
                "apply compatible registry patch",
            ],
            "constraints": ["verify current version"],
            "output_contract": "validated registry change",
            "evidence_refs": ["ref:registry-a"],
            "confidence": 0.9,
        },
        "fabric_api",
    )
    second = _sanitize_procedure(
        {
            "name": "verify registry mapping",
            "activate_when": ["registry build error"],
            "contraindications": [],
            "steps": [
                "inspect registry mapping exactly",
                "run compiler and tests",
            ],
            "constraints": ["preserve loader compatibility"],
            "output_contract": "validated registry change",
            "evidence_refs": ["ref:registry-b"],
            "confidence": 0.7,
        },
        "fabric_api",
    )
    assert first is not None and second is not None

    consolidated = _consolidate_skills([first, second])
    assert len(consolidated) == 1
    skill = consolidated[0]
    assert skill["skill_kind"] == "consolidated"
    assert len(skill["steps"]) == 1
    assert skill["steps"][0] in {first["steps"][0], second["steps"][0]}
    assert set(skill["evidence_refs"]) == {"ref:registry-a", "ref:registry-b"}
    assert set(skill["member_skill_ids"]) == {first["skill_id"], second["skill_id"]}
    assert skill["confidence"] == 0.7
    assert skill["consolidation_policy"] == "common_supported_prefix_only"


def test_consolidation_never_crosses_domains_or_invents_common_steps() -> None:
    first = _sanitize_procedure(
        _procedure(
            "repair registry mapping",
            "registry compile error",
            "inspect registry API",
            "ref:fabric",
        ),
        "fabric_api",
    )
    other_domain = _sanitize_procedure(
        _procedure(
            "repair registry mapping",
            "registry compile error",
            "inspect registry API",
            "ref:gradle",
        ),
        "gradle",
    )
    divergent = _sanitize_procedure(
        _procedure(
            "verify registry mapping",
            "registry compile error",
            "run compiler tests",
            "ref:verify",
        ),
        "fabric_api",
    )
    assert first is not None and other_domain is not None and divergent is not None

    assert _consolidate_skills([first, other_domain]) == []
    assert _consolidate_skills([first, divergent]) == []
