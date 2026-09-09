from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS
from tools.package_skills import build_payload


def test_canonical_skill_catalog_compiles_under_reviewed_validator_policy() -> None:
    payload = build_payload()
    assert set(payload["contracts"]) == set(CANONICAL_SKILLS)
    assert set(payload["skills"]) == set(CANONICAL_SKILLS)
