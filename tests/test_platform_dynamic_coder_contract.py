from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    CustomModuleGenerator,
)
from minecraft_mod_ai.platform_catalog import FABRIC_1201, FABRIC_1211, PlatformAdapter
from minecraft_mod_ai.platform_custom_coder_contract import _rewrite_message
from minecraft_mod_ai.platform_resolver import resolve_platform
from minecraft_mod_ai import repair_engine


def _future_adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_live_27_0_test",
        edition="java",
        loader="fabric",
        minecraft_version="27.0",
        java_version="25",
        yarn_mappings="mojang",
        fabric_loader="0.20.0",
        fabric_api="0.200.0+27.0",
        fabric_loom="1.20-SNAPSHOT",
        gradle="9.7",
        gradle_sha256="a" * 64,
        resource_pack_format=0,
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def test_custom_coder_structured_request_uses_exact_live_target() -> None:
    adapter = _future_adapter()
    message = {
        "role": "user",
        "content": json.dumps(
            {
                "phase": "generate_patch",
                "target": {
                    "minecraft_version": "1.20.1",
                    "loader": "fabric",
                    "mappings": "1.20.1+build.1",
                    "java": "17",
                },
                "module": {"module_id": "probe"},
            }
        ),
    }
    rewritten = _rewrite_message(message, adapter)
    payload = json.loads(rewritten["content"])
    assert payload["target"] == {
        "minecraft_version": "27.0",
        "loader": "fabric",
        "mappings": "mojang",
        "java": "25",
        "fabric_loader": "0.20.0",
        "fabric_api": "0.200.0+27.0",
        "loom": "1.20-SNAPSHOT",
        "gradle": "9.7",
    }


def test_custom_coder_system_prompt_cannot_leak_legacy_version() -> None:
    rewritten = _rewrite_message(
        {
            "role": "system",
            "content": (
                "Implement compilable Minecraft Java 1.20.1 Fabric code and data. "
                "You are an expert Minecraft 1.20.1 Fabric Java mod developer."
            ),
        },
        _future_adapter(),
    )
    assert "1.20.1" not in rewritten["content"]
    assert "27.0" in rewritten["content"]


def test_custom_patch_scope_allows_kotlin_gradle_metadata_but_not_arbitrary_files() -> None:
    generator = object.__new__(CustomModuleGenerator)
    generator._validate_operations(
        [
            {"operation": "create", "path": "build.gradle.kts", "content": ""},
            {"operation": "create", "path": "settings.gradle.kts", "content": ""},
            {"operation": "create", "path": "gradle/libs.versions.toml", "content": ""},
        ]
    )
    with pytest.raises(CustomModuleGenerationError):
        generator._validate_operations(
            [{"operation": "create", "path": "scripts/migrate.sh", "content": ""}]
        )


def test_repair_scope_supports_kotlin_gradle_metadata() -> None:
    assert ".kts" in repair_engine._ALLOWED_SUFFIXES


def test_revise_explicit_version_port_is_bound_as_migration() -> None:
    selection = resolve_platform(
        "Minecraft 1.21.1로 버전 업데이트해서 포팅해줘",
        existing_version="1.20.1",
        existing_loader="fabric",
    )
    assert selection.adapter.adapter_id == FABRIC_1211.adapter_id
    assert selection.migration_requested is True
    assert selection.preserved_existing_target is False
    assert selection.to_dict()["migration_requested"] is True


def test_revise_without_port_request_preserves_existing_target() -> None:
    selection = resolve_platform(
        "기존 모드에 아이템 하나 추가해줘",
        existing_version="1.20.1",
        existing_loader="fabric",
    )
    assert selection.adapter.adapter_id == FABRIC_1201.adapter_id
    assert selection.migration_requested is False
    assert selection.preserved_existing_target is True
