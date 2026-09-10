from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_ports import PortKind, PortRegistry
from minecraft_mod_ai.task_template_catalog import load_template
from minecraft_mod_ai.task_template_runner import execute_artifact_template


def _register_job() -> ArtifactJob:
    return ArtifactJob(
        job_id="raw_lunite.register_basic",
        template_id="fabric/item/register_basic",
        owner_module="raw_lunite",
        target_path="src/main/java/com/foo/space/registry/ModItems.java",
        anchor="mod_items_registry",
        produces=("raw_lunite.registry_id", "raw_lunite.java_symbol"),
        deterministic_inputs={
            "mod_id": "space",
            "package_path": "com/foo/space",
            "java_constant": "RAW_LUNITE",
            "registry_path": "raw_lunite",
        },
    )


def test_leaf_publishes_job_scoped_ports_not_global_names():
    registry = PortRegistry()
    receipt = execute_artifact_template(_register_job(), port_registry=registry)
    assert receipt["status"] == "PASS"
    assert registry.has("raw_lunite.registry_id")
    assert registry.has("raw_lunite.java_symbol")
    assert not registry.has("item_registry_id")
    assert registry.resolve(
        "raw_lunite.registry_id", PortKind.REGISTRY_ID, "Item"
    ).value == "space:raw_lunite"


def test_declared_job_dependency_is_actually_required():
    model = ArtifactJob(
        job_id="raw_lunite.model_basic",
        template_id="fabric/item/model_basic",
        owner_module="raw_lunite",
        requires=("raw_lunite.registry_id",),
        produces=("raw_lunite.model_ref",),
        deterministic_inputs={
            "mod_id": "space",
            "registry_path": "raw_lunite",
        },
    )
    with pytest.raises(ValueError, match="TEMPLATE_JOB_DEPENDENCY_MISSING"):
        execute_artifact_template(model, port_registry=PortRegistry())


def test_scoped_dependency_chain_executes_without_fuzzy_selection():
    registry = PortRegistry()
    execute_artifact_template(_register_job(), port_registry=registry)
    model = ArtifactJob(
        job_id="raw_lunite.model_basic",
        template_id="fabric/item/model_basic",
        owner_module="raw_lunite",
        requires=("raw_lunite.registry_id",),
        produces=("raw_lunite.model_ref",),
        deterministic_inputs={
            "mod_id": "space",
            "registry_path": "raw_lunite",
        },
    )
    receipt = execute_artifact_template(model, port_registry=registry)
    assert receipt["status"] == "PASS"
    assert registry.has("raw_lunite.model_ref")
    assert not registry.has("model_ref")


def test_template_output_arity_must_match_scoped_job_outputs():
    bad = _register_job()
    bad.produces = ("raw_lunite.registry_id",)
    with pytest.raises(ValueError, match="TEMPLATE_PORT_ARITY"):
        execute_artifact_template(bad, port_registry=PortRegistry())


def test_declared_leaf_fixtures_render():
    for template_id in (
        "fabric/item/register_basic",
        "fabric/item/settings_max_stack",
        "fabric/item/model_basic",
        "fabric/item/lang_en",
    ):
        template = load_template(template_id)
        fixture = template.get("fixture")
        assert isinstance(fixture, dict)
        job = ArtifactJob(
            job_id=f"fixture.{template_id}",
            template_id=template_id,
            owner_module="fixture",
            deterministic_inputs=dict(fixture["input"]),
        )
        receipt = execute_artifact_template(job)
        for expected in fixture["expected_contains"]:
            assert expected in receipt["rendered_output"]


def test_migrated_leaf_has_no_central_port_semantics_dependency(monkeypatch):
    import minecraft_mod_ai.task_template_runner as runner
    monkeypatch.setattr(runner, "_STANDARD_PORT_DEFINITIONS", {})
    registry = PortRegistry()
    execute_artifact_template(_register_job(), port_registry=registry)
    assert registry.resolve("raw_lunite.registry_id", PortKind.REGISTRY_ID, "Item").value == "space:raw_lunite"


@pytest.mark.parametrize("change", [{"kind": "TYPO"}, {"target_type": ""}, {"value": ""}])
def test_malformed_port_declaration_fails_before_publish(monkeypatch, change):
    import minecraft_mod_ai.task_template_catalog as catalog
    original = catalog.load_template
    def load(identifier):
        template = original(identifier)
        template["produces"][0].update(change)
        return template
    monkeypatch.setattr(catalog, "load_template", load)
    registry = PortRegistry()
    with pytest.raises(ValueError):
        execute_artifact_template(_register_job(), port_registry=registry)
    assert registry.all_ports() == {}
