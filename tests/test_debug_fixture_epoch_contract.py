"""Verify actual target-epoch registry ABI, no model or Gradle required."""
from __future__ import annotations

import pytest

from minecraft_mod_ai import debug_fixture_host as fixture


def _mock_grounding(keyed: bool):
    key = {
        "template_id": "fabric/item/key",
        "render_body": (
            "public static final ResourceKey<Item> {{java_constant}}_KEY = "
            "ResourceKey.create(Registries.ITEM, Identifier.of(\"{{mod_id}}\", \"{{registry_path}}\"));"
        ),
        "symbol_usage": ["resource_key_create"],
        "dependencies": [],
    }
    register = {
        "template_id": "fabric/item/register_keyed" if keyed else "fabric/item/register_direct",
        "render_body": (
            "public static final Item {{java_constant}} = Registry.register("
            "BuiltInRegistries.ITEM, "
            + ("ModItemIds.{{java_constant}}_KEY" if keyed else
               "Identifier.of(\"{{mod_id}}\", \"{{registry_path}}\")")
            + ", new Item(new Item.Properties()));"
        ),
        "symbol_usage": ["register_item"],
        "dependencies": ["{{subject}}.key_symbol"] if keyed else [],
    }
    return {
        "facts": [{
            "api_symbols": {
                "register_item": {"owner": "net.minecraft.registry.Registry"},
                **({"resource_key_create": {
                    "owner": "net.minecraft.resources.ResourceKey",
                }} if keyed else {}),
            },
            "templates": [key, register] if keyed else [register],
            "required_imports": [],
        }],
        "grounding_sha256": "sha256:" + "a" * 64,
    }


@pytest.mark.parametrize("keyed", [False, True])
def test_fixture_contract_follows_actual_registry_key_requirement(monkeypatch, keyed):
    monkeypatch.setattr(
        fixture, "_debug_grounding",
        lambda *, minecraft_version: _mock_grounding(keyed),
    )
    contract = fixture.debug_fixture_source_contract(
        package_name="example.fixture", minecraft_version="26.2" if keyed else "1.20.1",
    )
    symbols = set(contract["required_host_symbol_keys"])
    assert "register_item" in symbols
    assert ("resource_key_create" in symbols) is keyed
    source = fixture.render_debug_fixture_source(
        package_name="example.fixture",
        mod_id="probe",
        minecraft_version="26.2" if keyed else "1.20.1",
    )
    assert "DEBUG_TOKEN" in source
    assert "Registry.register(" in source
    assert ("DEBUG_TOKEN_KEY" in source) is keyed
    assert "{{" not in source


def test_debug_fixture_uses_real_host_grounding_on_integrity_target():
    # Read the actual source registry instead of mocking a renderer. This is
    # the exact boundary that CI formerly missed until long Fabric-integrity.
    contract = fixture.debug_fixture_source_contract(
        package_name="dev.mmm.debugfixture",
        minecraft_version="1.20.1",
    )
    assert contract["required_host_symbol_specs"]
    source = fixture.render_debug_fixture_source(
        package_name="dev.mmm.debugfixture",
        mod_id="debug_fixture",
        minecraft_version="1.20.1",
    )
    assert "Registry.register(" in source
    assert "DEBUG_TOKEN" in source
    assert "{{" not in source


def test_integrity_gradle_mapping_matches_host_generated_debug_source():
    """Pre-Gradle contract: probe and host-rendered Java use one naming regime."""
    from tools import verify_integrity_minecraft as integrity

    build = integrity._BUILD_GRADLE
    assert "mappings loom.officialMojangMappings()" in build
    assert "net.fabricmc:yarn:" not in build
    # Loom's transitive third-party plugins (ASM, Guava, Gson) are hosted in
    # Maven Central, not reliably mirrored by Fabric's own repository.
    settings = integrity._project_files(
        integrity._probe_spec(),
        {"item_registry_artifact": integrity._PROBE_SOURCE},
    )["settings.gradle"]
    assert "mavenCentral()" in settings
    assert "maven.fabricmc.net" in settings

    generated = fixture.render_debug_fixture_source(
        package_name="dev.mmm.debugfixture",
        mod_id="mmm_debug_fixture",
        minecraft_version="1.20.1",
    )
    sources = (
        generated,
        integrity._PROBE_SOURCE,
        integrity._PROBE_TEST_SOURCE,
    )
    # A Yarn project and official Mojang-mapped source cannot compile together.
    for source in sources:
        assert "net.minecraft.registry." not in source
        assert "net.minecraft.util.Identifier" not in source
        assert "net.minecraft.test." not in source
        assert "net.minecraft.item." not in source
    assert "net.minecraft.core.Registry" in generated
    assert "net.minecraft.core.Registry" in integrity._PROBE_SOURCE
    assert "net.minecraft.gametest.framework.GameTestHelper" in integrity._PROBE_TEST_SOURCE
    assert "context.succeed()" in integrity._PROBE_TEST_SOURCE


def test_unverified_keyed_registry_symbol_rejected(monkeypatch):
    invalid = _mock_grounding(True)
    invalid["facts"][0]["api_symbols"].pop("resource_key_create")
    monkeypatch.setattr(
        fixture, "_debug_grounding",
        lambda *, minecraft_version: invalid,
    )
    with pytest.raises(fixture.DebugFixtureHostError, match="SYMBOLS_INCOMPLETE"):
        fixture.debug_fixture_source_contract(
            package_name="example.fixture", minecraft_version="26.2",
        )
