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
    with pytest.raises(ValueError, match="TARGET_MAPPINGS_REQUIRED"):
        _bound_target(
            {"target": {"minecraft_version": "1.21.1", "loader": "fabric"}}
        )


@pytest.mark.parametrize("version", ["1.21.11", "26.2"])
def test_bound_target_consumes_real_provider_receipt(version):
    from minecraft_mod_ai.platform_catalog import adapter_for_target

    adapter = adapter_for_target(version, "fabric")
    assert _bound_target({"_platform_selection": {"target": adapter.public_dict()}}) == {
        "minecraft_version": version,
        "loader": "fabric",
        "mappings": adapter.yarn_mappings,
    }


def test_bound_target_round_trips_every_published_host_target():
    from minecraft_mod_ai.host_version_catalog import host_target, host_versions

    for version in host_versions(1000):
        adapter = host_target(version)
        assert _bound_target({"_platform_selection": {"target": adapter.public_dict()}}) == {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        }, version


def test_bound_target_preserves_host_selection_over_old_design_coordinates():
    assert _bound_target({
        "minecraft_version": "1.21.1", "loader": "fabric", "mappings": "old",
        "_platform_selection": {"target": {"minecraft_version": "26.2", "loader": "fabric"}},
    }) == {"minecraft_version": "26.2", "loader": "fabric", "mappings": ""}


@pytest.mark.parametrize("target", [
    {"minecraft_version": "26.2", "loader": "fabric", "mappings": "old"},
    {"minecraft_version": "26.2", "loader": "fabric", "mappings_applicable": True},
    {"minecraft_version": "1.21.1", "loader": "fabric", "mappings": "one", "yarn_mappings": "two"},
])
def test_bound_target_rejects_contradictory_host_receipt(target):
    with pytest.raises(ValueError, match="TARGET_MAPPINGS"):
        _bound_target({"_platform_selection": {"target": target}})
