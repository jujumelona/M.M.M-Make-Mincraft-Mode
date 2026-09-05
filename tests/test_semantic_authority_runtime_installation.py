from __future__ import annotations

import pytest


def test_package_runtime_installs_authoritative_semantic_request_path():
    import minecraft_mod_ai  # noqa: F401 - package import is the production bootstrap
    from minecraft_mod_ai.game_design import GameDesignPlanner
    from minecraft_mod_ai.runtime_preflight import _assert_authoritative_requirement_path
    from minecraft_mod_ai.semantic_batching_contract import (
        build_bounded_requirement_catalog,
    )

    assert getattr(
        GameDesignPlanner.plan,
        "__mmm_request_contract_guard__",
        False,
    ) is True
    assert getattr(
        build_bounded_requirement_catalog,
        "__mmm_bounded_semantic_batching__",
        False,
    ) is True
    _assert_authoritative_requirement_path()


def test_runtime_preflight_rejects_semantic_authority_bypass(monkeypatch):
    import minecraft_mod_ai  # noqa: F401 - package import is the production bootstrap
    from minecraft_mod_ai import evidence_request_guard
    from minecraft_mod_ai.runtime_preflight import (
        RuntimePreflightError,
        _assert_authoritative_requirement_path,
    )

    monkeypatch.setattr(
        evidence_request_guard,
        "build_authoritative_request_catalog",
        lambda prompt, router=None: {},
    )
    with pytest.raises(RuntimePreflightError, match="bounded request catalog owner"):
        _assert_authoritative_requirement_path()
