from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_ports import PortKind, PortRegistry, TypedPort
from minecraft_mod_ai.task_template_catalog import load_template
from minecraft_mod_ai.task_template_runner import execute_artifact_template


def test_leaf_template_register_basic_runs_deterministically():
    registry = PortRegistry()
    job = ArtifactJob(
        job_id="raw_lunite.register_basic",
        template_id="fabric/item/register_basic",
        owner_module="raw_lunite",
        target_path="src/main/java/com/foo/space/registry/ModItems.java",
        anchor="mod_items_registry",
        deterministic_inputs={
            "mod_id": "space",
            "package_path": "com/foo/space",
            "java_constant": "RAW_LUNITE",
            "registry_path": "raw_lunite",
        },
    )

    receipt = execute_artifact_template(job, port_registry=registry)

    assert receipt["status"] == "PASS"
    assert "Registry.register" in receipt["rendered_output"]
    assert "Registries.ITEM" in receipt["rendered_output"]
    assert "RAW_LUNITE" in receipt["rendered_output"]
    assert "raw_lunite" in receipt["rendered_output"]
    assert receipt["target_file"] == "src/main/java/com/foo/space/registry/ModItems.java"
    assert receipt["anchor"] == "mod_items_registry"

    # Validators ran
    assert len(receipt["validations"]) == 2
    assert all(v["status"] == "PASS" for v in receipt["validations"])

    # Ports published
    assert registry.has("item_registry_id")
    port = registry.resolve("item_registry_id", PortKind.REGISTRY_ID, "Item")
    assert port.value == "space:raw_lunite"

    assert registry.has("item_symbol")
    port_sym = registry.resolve("item_symbol", PortKind.JAVA_SYMBOL, "Item")
    assert port_sym.value == "ModItems.RAW_LUNITE"


def test_chained_leaf_jobs_through_typed_ports():
    registry = PortRegistry()

    # Job 1: Register item
    job1 = ArtifactJob(
        job_id="raw_lunite.register_basic",
        template_id="fabric/item/register_basic",
        owner_module="raw_lunite",
        deterministic_inputs={
            "mod_id": "space",
            "package_path": "com/foo/space",
            "java_constant": "RAW_LUNITE",
            "registry_path": "raw_lunite",
        },
    )
    r1 = execute_artifact_template(job1, port_registry=registry)
    assert r1["status"] == "PASS"

    # Job 2: Model basic (requires mod_id, registry_path)
    job2 = ArtifactJob(
        job_id="raw_lunite.model_basic",
        template_id="fabric/item/model_basic",
        owner_module="raw_lunite",
        deterministic_inputs={
            "mod_id": "space",
            "registry_path": "raw_lunite",
        },
    )
    r2 = execute_artifact_template(job2, port_registry=registry)
    assert r2["status"] == "PASS"
    assert "minecraft:item/generated" in r2["rendered_output"]
    assert "space:item/raw_lunite" in r2["rendered_output"]
    assert registry.has("model_ref")

    # Job 3: Lang EN (requires mod_id, registry_path, display_name)
    job3 = ArtifactJob(
        job_id="raw_lunite.lang_en",
        template_id="fabric/item/lang_en",
        owner_module="raw_lunite",
        deterministic_inputs={
            "mod_id": "space",
            "registry_path": "raw_lunite",
            "display_name": "Raw Lunite",
        },
    )
    r3 = execute_artifact_template(job3, port_registry=registry)
    assert r3["status"] == "PASS"
    assert "item.space.raw_lunite" in r3["rendered_output"]
    assert "Raw Lunite" in r3["rendered_output"]
    assert registry.has("translation_key")


def test_leaf_template_fixtures_pass():
    """Verify that all declared leaf template fixtures render and pass their expected output."""
    templates = [
        "fabric/item/register_basic",
        "fabric/item/settings_max_stack",
        "fabric/item/model_basic",
        "fabric/item/lang_en",
    ]

    for template_id in templates:
        tmpl = load_template(template_id)
        fixture = tmpl.get("fixture")
        assert fixture is not None, f"Template {template_id} must declare a fixture"
        inputs = fixture["input"]
        expected = fixture["expected_contains"]

        job = ArtifactJob(
            job_id=f"test.{template_id}",
            template_id=template_id,
            owner_module="test",
            deterministic_inputs=inputs,
        )
        receipt = execute_artifact_template(job)
        assert receipt["status"] == "PASS"
        for text in expected:
            assert text in receipt["rendered_output"], f"Expected {text!r} in {template_id} output"
