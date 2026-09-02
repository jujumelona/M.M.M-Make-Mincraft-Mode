from __future__ import annotations

import json

from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai import fabric_official_template_provider as fabric_provider
from minecraft_mod_ai import platform_catalog
from minecraft_mod_ai.platform_catalog import PlatformAdapter


def _adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_live_27_0",
        edition="java",
        loader="fabric",
        minecraft_version="27.0",
        java_version="21",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
        fabric_loader="0.18.2",
        fabric_api="0.140.0+27.0",
        fabric_loom="1.11.8",
        gradle="9.1.0",
        gradle_sha256="a" * 64,
        data_pack_version="100.0",
        resource_pack_version="100.0",
        resource_pack_format=100,
        release_metadata_url="https://piston-meta.mojang.com/mc/game/version_manifest_v2.json",
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset({"item", "block"}),
    )


def test_balanced_think_block_is_removed_without_losing_core_loop() -> None:
    raw = """
<think>I need to reason about ordering before answering.</think>
- Gather ore and trade it for credits.
- Buy hull parts and assemble the ship.
- Upgrade the drive, launch, and travel to another planet.
"""
    assert agentic._parse_field_output(raw, "core_loop") == [
        "Gather ore and trade it for credits.",
        "Buy hull parts and assemble the ship.",
        "Upgrade the drive, launch, and travel to another planet.",
    ]


def test_identity_wrapper_is_removed_but_semantic_title_and_pitch_are_kept() -> None:
    assert (
        agentic._parse_field_output("Here is the title: Starbound Foundry", "title")
        == "Starbound Foundry"
    )
    assert (
        agentic._parse_field_output(
            "The pitch is: Mine, trade, build, upgrade, and explore "
            "as one continuous progression loop.",
            "pitch",
        )
        == "Mine, trade, build, upgrade, and explore as one continuous progression loop."
    )


def test_unbalanced_think_marker_still_fails_closed() -> None:
    try:
        agentic._parse_field_output("<think>I need to hide this\n- Gather ore", "core_loop")
    except Exception as exc:
        assert "internal model reasoning/meta output" in str(exc)
    else:  # pragma: no cover - contract failure
        raise AssertionError("unbalanced hidden reasoning must remain fail-closed")


def test_official_bootstrap_writer_preserves_full_immutable_receipt(tmp_path) -> None:
    adapter = _adapter()
    adapter.validate()
    receipt = {
        "schema_version": "mmm/fabric-official-template-v2",
        "provider": "fabricmc.net/cli",
        "project_manifest_sha256": "sha256:" + "b" * 64,
    }

    fabric_provider._write_platform_lock(tmp_path, adapter, receipt)

    payload = json.loads(
        (tmp_path / ".minecraft_ai" / "platform-lock.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "mmm/generated-platform-lock-v4"
    assert payload["adapter_id"] == adapter.adapter_id
    assert payload["mappings_kind"] == adapter.mappings_kind
    assert payload["mappings_version"] == adapter.mappings_version
    assert payload["gradle_sha256"] == adapter.gradle_sha256
    assert payload["gradle_distribution_url"].endswith("gradle-9.1.0-bin.zip")
    assert payload["data_pack_version"] == adapter.data_pack_version
    assert payload["resource_pack_version"] == adapter.resource_pack_version
    assert payload["release_metadata_url"] == adapter.release_metadata_url
    assert payload["deterministic_module_kinds"] == ["block", "item"]
    assert payload["receipt_sha256"].startswith("sha256:")
    assert payload["bootstrap"] == receipt

    rebound = platform_catalog.adapter_from_project(tmp_path)
    assert rebound == adapter
