from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.skill_catalog import (
    CANONICAL_SKILLS,
    POLICY_NATIVE_SKILLS,
    REVIEWED_TOOL_STAGES,
    SkillPolicyError,
    compile_skill_catalog,
    compile_skill_contract,
    validate_skill_catalog,
)


def test_catalog_compiles_every_skill_into_runtime_contracts() -> None:
    report = validate_skill_catalog()
    assert report["passed"], report["findings"]
    assert len(CANONICAL_SKILLS) == 27
    assert set(report["contracts"]) == set(CANONICAL_SKILLS)

    contracts = compile_skill_catalog()
    assert set(contracts) == set(CANONICAL_SKILLS)
    assert all(contract.allowed_tools for contract in contracts.values())
    assert all(contract.tool_routes for contract in contracts.values())
    assert all(contract.validators for contract in contracts.values())


@pytest.mark.parametrize("skill", sorted(POLICY_NATIVE_SKILLS))
def test_new_skill_frontmatter_is_minimal_and_has_no_readme(skill: str) -> None:
    directory = Path("skills") / skill
    text = (directory / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    keys = {
        line.split(":", 1)[0].strip()
        for line in frontmatter.splitlines()
        if ":" in line
    }
    assert keys == {"name", "description"}
    assert not (directory / "README.md").exists()
    assert f"Use ${skill}" in (
        directory / "agents" / "openai.yaml"
    ).read_text(encoding="utf-8")


def test_massive_graph_policy_denies_tools_stages_and_missing_approval() -> None:
    contract = compile_skill_contract("compile-massive-work-graph")
    assert contract.tool_routes["plan_complete_game"] == "planning"
    assert contract.tool_routes["execute_complete_project"] == "generation"

    denied_tool = contract.authorize_tool("runtime_start_server", "runtime")
    assert not denied_tool.allowed
    assert denied_tool.reason == "tool_not_allowlisted"

    denied_stage = contract.authorize_tool("plan_complete_game", "generation")
    assert not denied_stage.allowed
    assert denied_stage.reason == "tool_not_exposed_in_stage"

    denied_write = contract.authorize_tool(
        "execute_complete_project",
        "generation",
    )
    assert not denied_write.allowed
    assert denied_write.reason == "write_approval_required"

    allowed = contract.authorize_tool(
        "execute_complete_project",
        "generation",
        write_approved=True,
    )
    assert allowed.allowed

    denied_runtime = contract.authorize_tool(
        "execute_complete_project",
        "generation",
        write_approved=True,
        runtime_requested=True,
    )
    assert not denied_runtime.allowed
    assert denied_runtime.reason == "runtime_approval_required"


def test_adaptive_evidence_policy_is_read_only_and_fail_closed() -> None:
    contract = compile_skill_contract("gather-adaptive-minecraft-evidence")
    assert contract.stages == ("research",)
    assert contract.authorize_tool("search_code_rag", "research").allowed
    assert not contract.authorize_tool(
        "apply_source_patch",
        "generation",
        write_approved=True,
    ).allowed

    context = {validator: True for validator in contract.validators}
    assert contract.failed_validators(context) == ()
    context["source_provenance"] = False
    assert contract.failed_validators(context) == ("source_provenance",)


def test_ai_technique_policy_is_read_only_and_fail_closed() -> None:
    contract = compile_skill_contract("select-compatible-ai-technique")
    assert contract.stages == ("planning", "research")
    assert contract.authorize_tool(
        "build_technology_radar", "planning"
    ).allowed
    assert contract.authorize_tool(
        "inspect_huggingface_model", "research"
    ).allowed
    assert contract.authorize_tool(
        "assess_technology_compatibility", "research"
    ).allowed
    assert not contract.authorize_tool(
        "execute_complete_project",
        "generation",
        write_approved=True,
    ).allowed

    context = {validator: True for validator in contract.validators}
    assert contract.failed_validators(context) == ()
    context["data_flow_and_consent"] = False
    assert contract.failed_validators(context) == (
        "data_flow_and_consent",
    )


def test_retry_and_exit_contracts_are_executable() -> None:
    contract = compile_skill_contract("resume-production-run")
    retry = contract.retry
    assert retry.allows_retry(
        attempts_started=1,
        error_signature="compile:missing-method",
        prior_error_signatures=(),
        fresh_evidence=True,
    )
    assert not retry.allows_retry(
        attempts_started=1,
        error_signature="compile:missing-method",
        prior_error_signatures=("compile:missing-method",),
        fresh_evidence=True,
    )
    assert not retry.allows_retry(
        attempts_started=1,
        error_signature="compile:new",
        fresh_evidence=False,
    )
    assert not retry.allows_retry(
        attempts_started=retry.max_attempts,
        error_signature="compile:new",
        fresh_evidence=True,
    )

    assert contract.exit.resolve(
        validators_passed=True,
        receipts_complete=True,
    ) == "success"
    assert contract.exit.resolve(
        validators_passed=False,
        receipts_complete=False,
        unresolved_external=("java17",),
    ) == "blocked"
    assert contract.exit.resolve(
        validators_passed=False,
        receipts_complete=False,
        attempts_exhausted=True,
    ) == "failed"
    assert contract.exit.resolve(
        validators_passed=True,
        receipts_complete=False,
    ) == "in_progress"


def test_unknown_tool_or_validator_cannot_compile(tmp_path: Path) -> None:
    source = Path("skills") / "gather-adaptive-minecraft-evidence" / "SKILL.md"
    text = source.read_text(encoding="utf-8")
    skill_dir = tmp_path / "gather-adaptive-minecraft-evidence"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        text.replace("  - search_code_rag\n", "  - arbitrary_shell\n"),
        encoding="utf-8",
    )
    with pytest.raises(SkillPolicyError, match="unreviewed tools"):
        compile_skill_contract("gather-adaptive-minecraft-evidence", tmp_path)

    (skill_dir / "SKILL.md").write_text(
        text.replace("  - source_provenance\n", "  - trust_the_model\n"),
        encoding="utf-8",
    )
    with pytest.raises(SkillPolicyError, match="unreviewed validator"):
        compile_skill_contract("gather-adaptive-minecraft-evidence", tmp_path)


def test_packaged_skill_text_and_contracts_are_current() -> None:
    packaged = json.loads(
        Path("minecraft_mod_ai/packaged_skills.json").read_text(encoding="utf-8")
    )
    assert packaged["schema_version"] == "mmm/packaged-skills-v3"
    assert set(packaged["skills"]) == set(CANONICAL_SKILLS)
    assert set(packaged["contracts"]) == set(CANONICAL_SKILLS)
    for name, contract in compile_skill_catalog().items():
        assert packaged["contracts"][name] == contract.to_dict()
        assert packaged["skills"][name] == (
            Path("skills") / name / "SKILL.md"
        ).read_text(encoding="utf-8")


def test_reviewed_tool_stage_map_matches_mcp_surface() -> None:
    from minecraft_mod_ai.mcp_server import _TOOL_STAGES

    mcp_stages = {
        tool: frozenset(stage for stage in stages if stage != "all")
        for tool, stages in _TOOL_STAGES.items()
    }
    assert REVIEWED_TOOL_STAGES == mcp_stages
