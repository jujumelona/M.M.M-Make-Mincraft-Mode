"""Deterministic template and port tests, separate from HOST-authorized execution.

Render/port helpers can be checked without runtime authority. Executable jobs
must still reject a missing immutable version snapshot; tests must not revive
the obsolete unrestricted runtime to exercise pure template behavior.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_ports import PortKind, PortRegistry
from minecraft_mod_ai.artifact_template_runner import (
    _bind_job_dependencies,
    _logical_port,
)
from minecraft_mod_ai.implementation_template_renderer import render_template
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


def _model_job() -> ArtifactJob:
    return ArtifactJob(
        job_id="raw_lunite.model_basic",
        template_id="minecraft/resource/item/model_generated",
        owner_module="raw_lunite",
        requires=("raw_lunite.registry_id",),
        produces=("raw_lunite.model_ref",),
        deterministic_inputs={"mod_id": "space", "registry_path": "raw_lunite"},
    )


def _register_scoped_ports(registry: PortRegistry) -> None:
    job = _register_job()
    template = load_template(job.template_id)
    for logical, scoped in zip(template["produces"], job.produces, strict=True):
        registry.publish(_logical_port(logical, scoped, job.deterministic_inputs, job.template_id))


def test_execution_requires_host_snapshot_before_template_side_effects():
    registry = PortRegistry()
    with pytest.raises(ValueError, match="VERSION_CONTEXT_REQUIRED"):
        execute_artifact_template(_register_job(), port_registry=registry)
    assert registry.all_ports() == {}


def test_leaf_declares_job_scoped_ports_not_global_names():
    registry = PortRegistry()
    _register_scoped_ports(registry)
    assert registry.has("raw_lunite.registry_id")
    assert registry.has("raw_lunite.java_symbol")
    assert not registry.has("item_registry_id")
    assert registry.resolve("raw_lunite.registry_id", PortKind.REGISTRY_ID, "Item").value == "space:raw_lunite"


def test_declared_job_dependency_is_actually_required():
    with pytest.raises(ValueError, match="TEMPLATE_JOB_DEPENDENCY_MISSING"):
        _bind_job_dependencies(_model_job(), {}, PortRegistry())


def test_scoped_dependency_chain_has_no_fuzzy_selection():
    registry = PortRegistry()
    _register_scoped_ports(registry)
    model = _model_job()
    values = dict(model.deterministic_inputs)
    _bind_job_dependencies(model, values, registry)
    assert values["registry_id"] == "space:raw_lunite"
    template = load_template(model.template_id)
    rendered = render_template(template, model.deterministic_inputs)
    assert "space:item/raw_lunite" in rendered
    registry.publish(_logical_port(template["produces"][0], model.produces[0], values, model.template_id))
    assert registry.has("raw_lunite.model_ref")
    assert not registry.has("model_ref")


def test_invalid_scoped_output_arity_does_not_bypass_host_snapshot():
    bad = _register_job()
    bad.produces = ("raw_lunite.registry_id",)
    registry = PortRegistry()
    with pytest.raises(ValueError, match="VERSION_CONTEXT_REQUIRED"):
        execute_artifact_template(bad, port_registry=registry)
    assert registry.all_ports() == {}


def test_declared_leaf_fixtures_render_without_execution_authority():
    for template_id in (
        "fabric/item/register_basic",
        "fabric/item/settings_max_stack",
        "minecraft/resource/item/model_generated",
        "fabric/item/lang_en",
    ):
        template = load_template(template_id)
        fixture = template.get("fixture")
        assert isinstance(fixture, dict)
        rendered = render_template(template, fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in rendered


def test_migrated_leaf_ports_do_not_depend_on_central_name_table(monkeypatch):
    import minecraft_mod_ai.artifact_template_runner as runner

    monkeypatch.setattr(runner, "_STANDARD_PORT_DEFINITIONS", {})
    registry = PortRegistry()
    _register_scoped_ports(registry)
    assert registry.resolve("raw_lunite.registry_id", PortKind.REGISTRY_ID, "Item").value == "space:raw_lunite"


@pytest.mark.parametrize("change", [{"kind": "TYPO"}, {"target_type": ""}, {"value": ""}])
def test_malformed_port_declaration_rejected_before_publication(change):
    template = load_template("fabric/item/register_basic")
    declaration = deepcopy(template["produces"][0])
    declaration.update(change)
    registry = PortRegistry()
    with pytest.raises(ValueError):
        registry.publish(_logical_port(declaration, "raw_lunite.registry_id",
                                       _register_job().deterministic_inputs,
                                       "fabric/item/register_basic"))
    assert registry.all_ports() == {}
