from __future__ import annotations

from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.artifact_expansion import _canonical_candidate_inputs
from minecraft_mod_ai.canonical_java_target_contract import assert_canonical_java_target
from minecraft_mod_ai.canonical_schema_compiler import compile_leaf_schemas
from minecraft_mod_ai.host_screen_shell import (
    basic_screen_candidate_contract,
    screen_command_name,
)
from minecraft_mod_ai.implementation_template_renderer import render_template
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema


def _canonical_spec(version: str = "26.1.2") -> dict:
    facts = _canonical_candidate_inputs(
        SimpleNamespace(
            source_clause="Show the spaceship blueprint database interface",
            display_name="Blueprint Database",
            fact_type=SimpleNamespace(value="GUI_EXISTS"),
            parent_requirement="",
        ),
        canonical_leaf="minecraft/screen/registration",
        mod_id="ships",
        package_name="example.ships",
        package_path="example/ships",
        subject="blueprint_db",
        minecraft_version=version,
        context_id="candidate-26.1",
    )
    input_schema, _ = compile_leaf_schemas("minecraft/screen/registration")
    Draft202012Validator(input_schema).validate(facts)
    return facts["screen_registration_input"]


def test_small_ai_only_owns_two_bounded_display_strings():
    spec = _canonical_spec()
    assert [slot["name"] for slot in spec["slots"]] == ["ui_title", "ui_body"]
    assert "artifact_source" not in [slot["name"] for slot in spec["slots"]]
    assert spec["bindings"]["host_screen_capability"] == "basic_client_screen"
    for slot in spec["slots"]:
        assert_atomic_model_schema(slot["schema"], surface="host GUI tiny slot")
    assert "src/client/java/example/ships/client/generated/" in spec["target_path"]


def test_rendered_screen_has_real_client_command_and_screen_lifecycle():
    spec = _canonical_spec()
    generated = render_template(
        {"render": spec["render_mold"]},
        {"ui_title": "Blueprint Database", "ui_body": "View ship modules"},
    )
    assert "MMM:HOST_26_SCREEN_SHELL" in generated
    assert "ClientCommandRegistrationCallback.EVENT.register(" in generated
    assert "ClientCommands.literal(" in generated
    assert "net.minecraft.client.gui.screens.Screen" in generated
    assert "class GeneratedView extends Screen" in generated
    assert "Button.builder(" in generated
    assert "graphics.text(" in generated
    assert "void onInitializeClient()" in generated
    assert "super.init()" not in generated
    assert "net.minecraft.client.gui.screen" not in generated.replace(
        "net.minecraft.client.gui.screens.Screen", ""
    )
    assert f'ClientCommands.literal("{spec["bindings"]["host_screen_command"]}")' in generated
    assert_canonical_java_target(generated, spec)
    Draft202012Validator(spec["output_schema"]).validate(generated)


def test_model_cannot_inject_java_source_or_newlines_through_labels():
    spec = _canonical_spec()
    title, body = (slot["schema"] for slot in spec["slots"])
    validator_title = Draft202012Validator(title)
    validator_body = Draft202012Validator(body)
    assert validator_title.is_valid("宇宙船 데이터 화면")
    assert validator_body.is_valid("Show available blueprints")
    assert not validator_title.is_valid('Hello"; System.exit(0); //')
    assert not validator_body.is_valid("first\nsecond")
    assert not validator_body.is_valid("System\\nexit")


def test_legacy_screen_remains_source_generation_contract():
    spec = _canonical_spec("1.21.5")
    assert [slot["name"] for slot in spec["slots"]] == ["artifact_source"]
    assert spec["render_mold"] == "{{artifact_source}}"


def test_same_subject_has_stable_command_and_different_subjects_are_distinct():
    one = screen_command_name("ships", "blueprint_db")
    assert one == screen_command_name("ships", "blueprint_db")
    assert one != screen_command_name("ships", "engine_modules")
    assert one.startswith("mmm_")
    assert len(one) == 20


def test_host_rejects_malformed_shell_identifiers():
    with pytest.raises(ValueError, match="HOST_SCREEN_MOD_ID_INVALID"):
        screen_command_name("Bad.Mod", "blueprint")
    with pytest.raises(ValueError, match="HOST_SCREEN_PACKAGE_INVALID"):
        basic_screen_candidate_contract(
            package_name="com.invalid;Runtime", class_name="SomeRegistration",
            mod_id="ships", subject="blueprint", default_title="Blueprint",
        )
