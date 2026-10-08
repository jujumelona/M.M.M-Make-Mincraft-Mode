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


def test_gui_has_zero_model_slots_and_full_host_rendered_static_copy():
    spec = _canonical_spec()
    assert spec["slots"] == []
    assert spec["bindings"]["host_screen_capability"] == "basic_client_screen"
    assert "src/client/java/example/ships/client/generated/" in spec["target_path"]
    assert 'SCREEN_TITLE = "Blueprint Database"' in spec["render_mold"]
    assert 'SCREEN_BODY = "Planned: Show the spaceship blueprint database interface"' in spec["render_mold"]
    assert "{{ui_title}}" not in spec["render_mold"]
    assert "{{ui_body}}" not in spec["render_mold"]


def test_rendered_screen_has_real_client_command_and_screen_lifecycle():
    spec = _canonical_spec()
    generated = render_template({"render": spec["render_mold"]}, {})
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


def test_host_java_copy_escapes_untrusted_long_labels_without_model_inference():
    attack = 'Queue GUI"; java.lang.Runtime.getRuntime().exec("oops"); //'
    long_requirement = "Displays feature names " * 200
    contract = basic_screen_candidate_contract(
        package_name="example.ships.client.generated",
        class_name="QueueRegistration",
        mod_id="ships", subject="queue",
        default_title=attack,
        requirement=long_requirement + "\\n next line",
    )
    assert contract["slots"] == []
    source = render_template({"render": contract["render_mold"]}, {})
    assert "Runtime.getRuntime().exec" in source
    # Injection-like source text remains confined to an escaped string literal,
    # not executable Java expressions or code after the string terminator.
    assert '\"; java.lang.Runtime' not in source
    assert 'SCREEN_TITLE = "Queue GUI' in source
    assert '\\n next line' not in source
    body = source.split("SCREEN_BODY = ", 1)[1].split(";", 1)[0]
    assert len(body) <= 142
    assert body.startswith('"Planned: ')
    assert "…" in body


def test_gui_generation_keeps_original_unbounded_requirement_in_contract():
    spec = _canonical_spec()
    assert spec["requirement"] == "Show the spaceship blueprint database interface"
    assert spec["bindings"]["source_requirement"] == spec["requirement"]

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



def test_26_2_screen_api_uses_minecraft_gui_after_fabric_migration():
    spec = _canonical_spec("26.2")
    source = render_template({"render": spec["render_mold"]}, {})
    assert "client.gui.setScreen(" in source
    assert "client.setScreen(" not in source
    assert_canonical_java_target(source, spec)
    legacy26 = _canonical_spec("26.1.2")
    old_source = render_template({"render": legacy26["render_mold"]}, {})
    assert "client.setScreen(" in old_source
    assert "client.gui.setScreen(" not in old_source


def test_future_unreviewed_gui_apis_fail_closed_instead_of_using_old_template():
    with pytest.raises(ValueError, match="HOST_SCREEN_UNREVIEWED_MINECRAFT_EPOCH"):
        _canonical_spec("26.3")


def test_canonical_26_1_screen_generator_succeeds_without_any_ai_router():
    """Real canonical leaf execution never reaches the fixed-template model call."""
    from minecraft_mod_ai.canonical_generators import generate_canonical_leaf

    leaf = "minecraft/screen/registration"
    facts = _canonical_candidate_inputs(
        SimpleNamespace(
            source_clause="Starship queue, currency, parts and progress UI.",
            display_name="Starship Build Queue GUI",
            fact_type=SimpleNamespace(value="GUI_EXISTS"),
            parent_requirement="",
        ),
        canonical_leaf=leaf,
        mod_id="ships",
        package_name="example.ships",
        package_path="example/ships",
        subject="starship_build_queue_gui",
        minecraft_version="26.1.2",
        context_id="candidate-26.1",
    )
    input_schema, output_schema = compile_leaf_schemas(leaf)

    class ActualSchemas:
        def validate_input(self, _, payload):
            Draft202012Validator(input_schema).validate(payload)

        def validate_output(self, _, payload):
            Draft202012Validator(output_schema).validate(payload)

    output = generate_canonical_leaf(
        facts, leaf_id=leaf, router=None,
        authority=SimpleNamespace(types=ActualSchemas()),
    )
    java = output["screen_registration_artifact"]
    assert "Starship Build Queue GUI" in java
    assert 'Planned: Starship queue, currency, parts and progress UI.' in java
    assert "ClientCommandRegistrationCallback" in java
    assert output["screen_registration_receipt"]["content_sha256"].startswith("sha256:")
