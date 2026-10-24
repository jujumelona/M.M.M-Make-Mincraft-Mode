"""Regression for concrete entity classes and approved producer Java handoff."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.artifact_job import validate_artifact_job_graph
from minecraft_mod_ai.artifact_ports import PortKind, PortRegistry, TypedPort
from minecraft_mod_ai.host_version_catalog import host_target
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact
from minecraft_mod_ai.request_runtime_records import (
    generated_owner_source_context,
    verified_graph_java_source_paths,
)


def test_entity_fact_creates_concrete_type_before_registry():
    target = host_target("auto")
    jobs = expand_facts_to_jobs(
        [PromptFact(fact_id="entity_alien", fact_type=FactType.ENTITY_EXISTS,
                    subject="alien_combat_hostile_entity")],
        mod_id="space", package_name="example.space",
        version_context=target.version_context,
    )
    validate_artifact_job_graph(jobs, module_ids=["alien_combat_hostile_entity"])
    by_leaf = {job.canonical_leaf: job for job in jobs}
    assert set(by_leaf) == {"minecraft/entity/type", "minecraft/entity/registry"}
    entity = by_leaf["minecraft/entity/type"]
    registry = by_leaf["minecraft/entity/registry"]
    assert entity.produces[-1] == "alien_combat_hostile_entity.entity_type_artifact"
    assert registry.requires == ("alien_combat_hostile_entity.entity_type_artifact",)
    assert registry.required_ports == ({
        "name": registry.requires[0], "kind": "GENERIC", "target_type": "code_fragment"},)
    type_input = entity.deterministic_inputs["_canonical_inputs"]["entity_type_input"]
    registry_input = registry.deterministic_inputs["_canonical_inputs"]["entity_registry_input"]
    assert type_input["target_path"].endswith("/AlienCombatHostileEntity.java")
    assert registry_input["target_path"].endswith("/AlienCombatHostileEntityRegistry.java")
    from minecraft_mod_ai.artifact_graph_executor import _declared_java_source_ports
    paths = _declared_java_source_ports(jobs)
    assert paths[registry.requires[0]] == type_input["target_path"]


def _sources(tmp_path, *, disk_source=None, port_source=None):
    relative = "src/main/java/example/space/generated/AlienCombatHostileEntity.java"
    source = "package example.space.generated; public class AlienCombatHostileEntity {}\n"
    file = tmp_path / relative
    file.parent.mkdir(parents=True)
    file.write_text(disk_source if disk_source is not None else source, encoding="utf-8")
    registry = PortRegistry()
    port = "alien.entity_type_artifact"
    registry.publish(TypedPort(port, PortKind.GENERIC, "code_fragment",
                               json.dumps(port_source if port_source is not None else source)))
    job = SimpleNamespace(job_id="alien.registry", canonical_leaf="minecraft/entity/registry",
                          requires=(port,))
    context = {"_artifact_java_source_ports": {port: relative}}
    return job, context, registry, relative


def test_required_materialized_type_is_passed_to_java_author(tmp_path):
    job, context, registry, relative = _sources(tmp_path)
    required = verified_graph_java_source_paths(
        job, context=context, port_registry=registry, project_root=tmp_path,
    )
    assert required == (relative,)
    details = generated_owner_source_context(tmp_path, {"bindings": {}},
                                            required_paths=required)
    assert len(details) == 1
    assert details[0]["path"] == relative
    assert "class AlienCombatHostileEntity" in details[0]["source"]
    assert details[0]["authority"] == "materialized_prerequisite_source"


def test_missing_type_or_forged_port_cannot_reach_coder(tmp_path):
    job, context, registry, relative = _sources(tmp_path, port_source="fake Java")
    with pytest.raises(ValueError, match="ARTIFACT_REQUIRED_SOURCE_CONTENT_MISMATCH"):
        verified_graph_java_source_paths(
            job, context=context, port_registry=registry, project_root=tmp_path)

    (tmp_path / relative).unlink()
    with pytest.raises(ValueError, match="ARTIFACT_REQUIRED_JAVA_SOURCE_NOT_MATERIALIZED"):
        generated_owner_source_context(tmp_path, {"bindings": {}},
                                       required_paths=(relative,))


def test_registry_without_concrete_producer_is_rejected_before_model(tmp_path):
    job = SimpleNamespace(job_id="alien.registry", canonical_leaf="minecraft/entity/registry",
                          requires=())
    with pytest.raises(ValueError, match="ARTIFACT_ENTITY_TYPE_SOURCE_DEPENDENCY_MISSING"):
        verified_graph_java_source_paths(
            job, context={}, port_registry=PortRegistry(), project_root=tmp_path)


def test_entity_type_requires_concrete_supertype_and_constructor():
    from minecraft_mod_ai.construction_file_author import (
        ConstructionFileError, _check_concrete_entity,
    )

    context = {"authoring_contract": {"canonical_leaf": "minecraft/entity/type"}}
    name = "AlienCombatHostileEntity"
    with pytest.raises(ConstructionFileError, match="SUPERCLASS_REQUIRED"):
        _check_concrete_entity(
            "package example; public class AlienCombatHostileEntity { public void ping() {} }",
            context, name,
        )
    with pytest.raises(ConstructionFileError, match="CONSTRUCTOR_REQUIRED"):
        _check_concrete_entity(
            "package example; public class AlienCombatHostileEntity extends Entity { public void ping() {} }",
            context, name,
        )
    _check_concrete_entity(
        "package example; public class AlienCombatHostileEntity extends Entity { "
        "public AlienCombatHostileEntity(EntityType<?> type, Level level) {} }",
        context, name,
    )


def test_feature_requires_materialized_prior_gui_producer_source(tmp_path):
    from minecraft_mod_ai.request_runtime_records import generated_owner_source_context

    relative = "src/main/java/example/space/ShipyardScreen.java"
    spec = {"bindings": {
        "content_dependency_paths_json": json.dumps([relative]),
        "required_content_source_paths_json": json.dumps([relative]),
    }}
    with pytest.raises(ValueError, match="ARTIFACT_REQUIRED_JAVA_SOURCE_NOT_MATERIALIZED"):
        generated_owner_source_context(tmp_path, spec)
    file = tmp_path / relative
    file.parent.mkdir(parents=True)
    file.write_text(
        "package example.space; public class ShipyardScreen {"
        "public void placeSection() {} }",
        encoding="utf-8",
    )
    sources = generated_owner_source_context(tmp_path, spec)
    assert len(sources) == 1
    assert sources[0]["path"] == relative
    assert "placeSection" in sources[0]["public_declarations"]
    assert sources[0]["authority"] == "materialized_prerequisite_source"
