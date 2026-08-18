from __future__ import annotations

from minecraft_mod_ai.external_procedural_skill_contract import _compile_skillbank, _sanitize_procedure, _select_skills


def _procedure(name: str, activation: str, step: str, ref: str, confidence: float = 0.9):
    return {"name": name, "activate_when": [activation], "contraindications": [], "steps": [step], "constraints": ["verify current version"], "output_contract": "validated change", "evidence_refs": [ref], "confidence": confidence}


def test_external_procedure_requires_evidence() -> None:
    raw = _procedure("repair registry", "registry compile error", "inspect registry API", "ref:1")
    raw["evidence_refs"] = []
    assert _sanitize_procedure(raw, "fabric_api") is None


def test_skillbank_compiles_relations_and_retrieves_relevant_skill() -> None:
    bank = _compile_skillbank([{"domain_id": "fabric_api", "procedures": [_procedure("repair registry mapping", "registry compile error", "inspect exact mapping before patch", "ref:registry"), _procedure("verify registry repair", "registry build failure", "run compiler and tests", "ref:test", 0.8)]}])
    assert bank["schema_version"] == "mmm/external-procedural-skillbank-v1"
    assert len(bank["skills"]) == 2
    assert bank["relation_graph"]
    selected = _select_skills("fix registry compile error", bank["skills"], limit=1)
    assert len(selected) == 1
    assert "registry" in selected[0]["name"]
