from __future__ import annotations

import pytest

from minecraft_mod_ai.authored_production import _bound_target


def test_bound_target_allows_completely_absent_target() -> None:
    assert _bound_target({"authored_plan": {"text": "saved"}}) == {}


def test_bound_target_preserves_complete_top_level_target() -> None:
    target = {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "1.21.1+build.3",
    }
    assert _bound_target(target) == target


def test_bound_target_reads_platform_selection_target() -> None:
    target = {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "1.21.1+build.3",
    }
    assert _bound_target({"_platform_selection": {"target": target}}) == target


def test_bound_target_rejects_partial_target() -> None:
    with pytest.raises(ValueError, match="incomplete platform target"):
        _bound_target(
            {"target": {"minecraft_version": "1.21.1", "loader": "fabric"}}
        )
