from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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

    # Mutate the policy structurally rather than depending on a particular YAML
    # indentation style. The canonical Skill migration intentionally rewrites
    # policies with safe_dump, whose block-sequence indentation can differ from
    # the hand-authored source while remaining semantically identical.
    start = source.index("```yaml") + len("```yaml")
    end = source.index("```", start)
    policy = yaml.safe_load(source[start:end])
    validators = list(policy["validators"])
    validators[validators.index("approval_and_fidelity")] = "validator-not-reviewed-one"
    validators.insert(1, "validator-not-reviewed-two")
    policy["validators"] = validators
    replacement = "\n" + yaml.safe_dump(
        policy,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    source = source[:start] + replacement + source[end:]

    target = tmp_path / "generate-fabric-core" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")

    with pytest.raises(SkillPolicyError) as error:
        compile_skill_contract("generate-fabric-core", tmp_path)

    message = str(error.value)
    assert "validator-not-reviewed-one" in message
    assert "validator-not-reviewed-two" in message
