from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import platform_resolver as resolver
from minecraft_mod_ai import repair_engine
from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    CustomModuleGenerator,
)
from minecraft_mod_ai.platform_catalog import PlatformAdapter, adapter_for_target
from minecraft_mod_ai.platform_custom_coder_contract import _bind_target
from minecraft_mod_ai.platform_evidence_pipeline import PlatformOptimization, TargetEvidence
from minecraft_mod_ai.platform_resolver import resolve_platform


def _future_adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_live_27_0_test",
        edition="java",
        loader="fabric",
        minecraft_version="27.0",
        java_version="25",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
        fabric_loader="0.20.0",
        fabric_api="0.200.0+27.0",
        fabric_loom="1.20-SNAPSHOT",
        gradle="9.7",
        gradle_sha256="a" * 64,
        data_pack_version="100.0",
        resource_pack_version="100.0",
        resource_pack_format=100,
        release_metadata_url="https://www.minecraft.net/en-us/article/minecraft-java-edition-27-0",
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def _optimization(adapter: PlatformAdapter) -> PlatformOptimization:
    evidence = TargetEvidence(
        adapter=adapter,
        requested_capabilities=(),
        covered_capabilities=(),
        exact_projects=(),
        exact_versions=1,
        verified_hash_files=1,
        dependency_edges=0,
        maintenance_signals=1,
        adoption=0,
        freshness=0.0,
        evidence_quality=1.0,
        integration_risk=0.0,
        residual_cost=0,
        dependency_complexity=0,
    )
    optimization = PlatformOptimization(
        selected=adapter,
        evidence=evidence,
        candidates=(evidence,),
        capability_queries=("migration capability",),
        discovery_mode="test-host-reuse",
    )
    object.__setattr__(
        optimization,
        "_mmm_reuse_plan",
        {
            "target": {
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
            },
            "capabilities": [],
        },
    )
    return optimization


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
    rewritten = _bind_target(message, adapter)
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


def test_custom_coder_prose_prompt_is_not_rewritten() -> None:
    message = {
        "role": "system",
        "content": (
            "Implement compilable Minecraft Java 1.20.1 Fabric code and data. "
            "You are an expert Minecraft 1.20.1 Fabric Java mod developer."
        ),
    }
    assert _bind_target(message, _future_adapter()) == message


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


def test_revise_explicit_version_is_hint_and_migration_uses_optimizer(monkeypatch) -> None:
    winner = _future_adapter()
    monkeypatch.setattr(
        resolver,
        "_optimize",
        lambda *_args, **_kwargs: _optimization(winner),
    )

    selection = resolve_platform(
        "Minecraft 1.21.1로 버전 업데이트해서 포팅해줘",
        existing_version="1.20.1",
        existing_loader="fabric",
    )

    assert selection.adapter.adapter_id == winner.adapter_id
    assert selection.adapter.minecraft_version == "27.0"
    assert selection.adapter.minecraft_version != "1.21.1"
    assert selection.explicit_version is True
    assert selection.migration_requested is True
    assert selection.preserved_existing_target is False
    assert selection.to_dict()["migration_requested"] is True


def test_revise_without_port_request_preserves_existing_target() -> None:
    selection = resolve_platform(
        "기존 모드에 아이템 하나 추가해줘",
        existing_version="1.20.1",
        existing_loader="fabric",
    )
    assert selection.adapter.adapter_id == adapter_for_target("1.20.1", "fabric").adapter_id
    assert selection.migration_requested is False
    assert selection.preserved_existing_target is True
