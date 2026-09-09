from pathlib import Path

from minecraft_mod_ai.skill_catalog import compile_skill_catalog, load_skill_catalog


SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"


def test_canonical_skill_catalog_compiles_under_reviewed_validator_policy() -> None:
    catalog = load_skill_catalog(SKILLS_ROOT)

    compiled = compile_skill_catalog(catalog)

    assert set(compiled) == set(catalog)
