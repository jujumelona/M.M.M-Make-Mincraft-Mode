from __future__ import annotations

from minecraft_mod_ai.external_procedural_skill_contract import (
    _compile_skillbank,
    _sanitize_procedure,
    _select_skills,
)


def _procedure(
    name: str,
    activation: str,
    step: str,
    ref: str,
    confidence: float = 0.9,
    *,
    requires=(),
    provides=(),
):
    return {
        "name": name,
        "activate_when": [activation],
        "contraindications": [],
        "steps": [step],
        "constraints": ["verify current version"],
        "output_contract": "validated change",
        "evidence_refs": [ref],
        "confidence": confidence,
        "requires": list(requires),
        "provides": list(provides),
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


def test_skillbank_keeps_source_procedures_separate_and_retrieves_relevant_skill() -> None:
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
                        provides=("registry_mapping",),
                    ),
                    _procedure(
                        "verify registry repair",
                        "registry build failure",
                        "run compiler and tests",
                        "ref:test",
                        0.8,
                        requires=("registry_mapping",),
                        provides=("verified_registry",),
                    ),
                ],
            }
        ]
    )

    assert bank["schema_version"] == "mmm/external-procedural-skillbank-v1"
    assert bank["source_skill_count"] == 2
    assert len(bank["skills"]) == 2
    assert bank["relation_graph"] == []
    assert all(skill.get("skill_kind") != "consolidated" for skill in bank["skills"])

    selected = _select_skills("fix registry compile error", bank["skills"], limit=1)
    assert len(selected) == 1
    assert "registry" in selected[0]["name"]


def test_sanitizer_preserves_only_explicit_requires_and_provides() -> None:
    skill = _sanitize_procedure(
        _procedure(
            "verify registry mapping",
            "registry build failure",
            "run compiler and tests",
            "ref:test",
            requires=("registry_mapping",),
            provides=("verified_registry",),
        ),
        "fabric_api",
    )

    assert skill is not None
    assert skill["requires"] == ["registry_mapping"]
    assert skill["provides"] == ["verified_registry"]

    no_edges = _sanitize_procedure(
        _procedure(
            "inspect mapping",
            "mapping question",
            "inspect current mapping",
            "ref:mapping",
        ),
        "fabric_api",
    )
    assert no_edges is not None
    assert no_edges["requires"] == []
    assert no_edges["provides"] == []


def test_skillbank_never_invents_cross_procedure_consolidation() -> None:
    bank = _compile_skillbank(
        [
            {
                "domain_id": "fabric_api",
                "procedures": [
                    _procedure(
                        "repair registry mapping",
                        "registry compile error",
                        "inspect registry API",
                        "ref:fabric",
                    ),
                    _procedure(
                        "verify registry mapping",
                        "registry compile error",
                        "run compiler tests",
                        "ref:verify",
                    ),
                ],
            },
            {
                "domain_id": "gradle",
                "procedures": [
                    _procedure(
                        "repair registry mapping",
                        "registry compile error",
                        "inspect registry API",
                        "ref:gradle",
                    )
                ],
            },
        ]
    )

    assert len(bank["skills"]) == 3
    assert {skill["domain_id"] for skill in bank["skills"]} == {"fabric_api", "gradle"}
    assert all("member_skill_ids" not in skill for skill in bank["skills"])
