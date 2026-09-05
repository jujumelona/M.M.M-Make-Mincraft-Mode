from __future__ import annotations


def test_package_runtime_installs_authoritative_semantic_request_path():
    import minecraft_mod_ai  # noqa: F401 - package import is the production bootstrap
    from minecraft_mod_ai import evidence_first_planning
    from minecraft_mod_ai import evidence_request_guard
    from minecraft_mod_ai import semantic_requirement_authority
    from minecraft_mod_ai.game_design import GameDesignPlanner

    assert getattr(
        GameDesignPlanner.plan,
        "__mmm_request_contract_guard__",
        False,
    ) is True
    assert (
        evidence_request_guard.build_authoritative_request_catalog
        is semantic_requirement_authority.build_approved_requirement_catalog
    )
    assert getattr(
        evidence_first_planning._validate_request_catalog,
        "__mmm_approved_requirement_authority__",
        False,
    ) is True
