from __future__ import annotations

from minecraft_mod_ai import external_procedural_skill_contract as skills
from minecraft_mod_ai import procedural_skill_identity_contract as identity


def _row() -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "mmm/external-procedural-skillbank-v1",
        "domain_id": "fabric",
        "name": "register item",
        "activate_when": ["register an item"],
        "contraindications": [],
        "steps": ["register the item in the reviewed registry"],
        "constraints": [],
        "output_contract": "registered item",
        "evidence_refs": ["fixture:1"],
        "confidence": 0.9,
        "rule": "fixture",
        "requires": [],
        "provides": ["item_registered"],
    }
    row["skill_id"] = identity._committed_identity(row)
    return row


def test_persistent_skill_content_must_match_declared_sha256_identity() -> None:
    valid = _row()
    tampered = dict(valid)
    tampered["steps"] = ["run an unrelated command"]

    assert identity.verified_persistent_skills([valid, tampered]) == [valid]


def test_identical_persistent_skill_rows_are_deduplicated_by_committed_identity() -> None:
    valid = _row()

    assert identity.verified_persistent_skills([valid, dict(valid)]) == [valid]


def test_persistent_skill_loader_is_bound_to_identity_validation() -> None:
    assert getattr(
        skills._load_persistent_skills,
        "_mmm_persistent_skill_identity_v1",
        False,
    ) is True
