from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.artifact_expansion import _canonical_candidate_inputs
from minecraft_mod_ai.canonical_java_target_contract import assert_canonical_java_target


def _spec(version: str, *, side: str = "CLIENT") -> dict:
    source = _canonical_candidate_inputs(
        SimpleNamespace(
            source_clause="Render module status",
            display_name="Module status",
            fact_type=SimpleNamespace(value="gui_exists"),
        ),
        canonical_leaf="minecraft/screen/registration",
        mod_id="testmod",
        package_name="example.mod",
        package_path="example/mod",
        subject="module_status",
        minecraft_version=version,
        context_id="test-context",
    )
    return source["screen_registration_input"]


def test_mojang_client_candidate_is_bound_to_client_source_set():
    modern = _spec("26.1.2")
    legacy = _spec("1.21.5")
    assert modern["target_path"].startswith("src/client/java/")
    assert legacy["target_path"].startswith("src/main/java/")
    assert modern["bindings"]["package_name"] == legacy["bindings"]["package_name"]


def test_reject_yarn_client_gui_before_materialization():
    spec = _spec("26.1.2")
    java = (
        "package example.mod.client.generated;\n"
        "import net.minecraft.client.gui.screen.Screen;\n"
        "import net.minecraft.text.Text;\n"
        "public final class ModuleStatusRegistration implements "
        "net.fabricmc.api.ClientModInitializer {\n"
        "public void onInitializeClient() {}\n}\n"
    )
    with pytest.raises(ValueError, match="CANONICAL_MOJANG_API_REQUIRED"):
        assert_canonical_java_target(java, spec)
    assert_canonical_java_target(java, _spec("1.21.5"))


def test_mojang_client_registration_refuses_screen_inheritance():
    spec = _spec("26.1.2")
    java = (
        "package example.mod.client.generated;\n"
        "import net.minecraft.client.gui.screens.Screen;\n"
        "public final class ModuleStatusRegistration extends Screen "
        "implements net.fabricmc.api.ClientModInitializer {\n"
        "public void onInitializeClient() {}\n}\n"
    )
    with pytest.raises(ValueError, match="CANONICAL_SCREEN_REGISTRATION_CLASS_INVALID"):
        assert_canonical_java_target(java, spec)


def test_mojang_client_source_location_is_proven_not_assumed():
    spec = _spec("26.1.2")
    spec["target_path"] = spec["target_path"].replace("src/client/java/", "src/main/java/")
    with pytest.raises(ValueError, match="CANONICAL_CLIENT_SOURCE_SET_REQUIRED"):
        assert_canonical_java_target(
            "package example.mod.client.generated; public final class A {}",
            spec,
        )


def test_valid_mojang_client_initializer_is_not_rejected():
    java = (
        "package example.mod.client.generated;\n"
        "import net.fabricmc.api.ClientModInitializer;\n"
        "public final class ModuleStatusRegistration implements ClientModInitializer {\n"
        "  public void onInitializeClient() {}\n"
        "}\n"
    )
    assert_canonical_java_target(java, _spec("26.1.2"))
