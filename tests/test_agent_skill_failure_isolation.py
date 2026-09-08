from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.agent_capability_context as capability_context
from minecraft_mod_ai.skill_catalog import SkillPolicyError


def test_invalid_reachable_skill_does_not_disable_sibling_skills(monkeypatch) -> None:
    def fake_compile(skill: str):
        if skill == "broken-skill":
            raise SkillPolicyError("synthetic invalid contract")
        return SimpleNamespace(name=skill, stages=("generation",))

    monkeypatch.setattr(capability_context, "_role_skill_contract", fake_compile)
    policy = capability_context._RolePolicySnapshot(
        model_role="coder",
        routes=(object(),),
        skills=frozenset({"broken-skill", "working-skill"}),
        mcp_servers=frozenset(),
    )

    contracts = capability_context._request_contracts_from_policy("generation", policy)

    assert [contract.name for contract in contracts] == ["working-skill"]


def test_invalid_skill_is_not_exposed_as_authority(monkeypatch) -> None:
    def fake_compile(skill: str):
        raise SkillPolicyError(f"invalid: {skill}")

    monkeypatch.setattr(capability_context, "_role_skill_contract", fake_compile)
    policy = capability_context._RolePolicySnapshot(
        model_role="coder",
        routes=(object(),),
        skills=frozenset({"broken-skill"}),
        mcp_servers=frozenset(),
    )

    assert capability_context._request_contracts_from_policy("generation", policy) == ()
