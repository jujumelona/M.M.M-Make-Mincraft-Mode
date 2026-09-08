from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.skill_catalog import (
    CANONICAL_SKILLS,
    REVIEWED_VALIDATORS,
    SkillPolicyError,
    compile_skill_catalog,
    compile_skill_contract,
)


def test_entire_canonical_skill_catalog_compiles() -> None:
    contracts = compile_skill_catalog()
    assert tuple(contracts) == CANONICAL_SKILLS
    for contract in contracts.values():
        assert contract.validators
        assert set(contract.validators) <= REVIEWED_VALIDATORS


def test_generate_fabric_core_uses_canonical_validator_ids() -> None:
    contract = compile_skill_contract("generate-fabric-core")
    assert contract.validators == (
        "approval_and_fidelity",
        "path_containment",
        "version_lock",
        "source_validation",
        "capability_receipts",
    )


def test_skill_contract_reports_all_unreviewed_validators_at_once(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "generate-fabric-core"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    source = source.replace(
        "  - approval_and_fidelity\n",
        "  - validator-not-reviewed-one\n  - validator-not-reviewed-two\n",
        1,
    )
    target = tmp_path / "generate-fabric-core" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")

    with pytest.raises(SkillPolicyError) as error:
        compile_skill_contract("generate-fabric-core", tmp_path)

    message = str(error.value)
    assert "validator-not-reviewed-one" in message
    assert "validator-not-reviewed-two" in message
