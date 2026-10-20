"""Replay production-logs (14): a 9B model invented writes to Entity.type.

The host must own this structural Java owner from inspected Fabric 26.2
classpath signatures and still require the real compilation gate.
"""
import json

import pytest

from minecraft_mod_ai.host_verified_entity_type import verified_entity_type_source
from minecraft_mod_ai.construction_host_source import compose_host_java_source


MONSTER = "net.minecraft.world.entity.monster.Monster"
ENTITY_TYPE = "net.minecraft.world.entity.EntityType"
LEVEL = "net.minecraft.world.level.Level"

SPEC = {
    "language": "java",
    "target_path": "src/main/java/example/Alien.java",
    "side": "COMMON",
    "requirement": "Fight aliens in space",
    "bindings": {
        "package_name": "example", "class_name": "Alien",
        "module_config_json": json.dumps({
            "behavior": "hostile_melee", "category": "creature",
            "max_health": 100, "archetype": "flying",
        }),
    },
}
CONTRACT = {"canonical_leaf": "minecraft/entity/type",
            "responsibility": "Implement the entity type owner"}
EVIDENCE = {
    "class_members": [{
        "verified_class": MONSTER,
        "declaration": {"kind": "CLASS", "owner": MONSTER, "access": 1057},
        "members": [{
            "kind": "CONSTRUCTOR", "owner": MONSTER,
            "access": 4, "parameters": [ENTITY_TYPE, LEVEL],
            "generic_parameters": [ENTITY_TYPE + "<? extends " + MONSTER + ">", LEVEL],
            "throws": [],
        }],
        "evidence": "actual_target_symbol_index",
    }],
}


def test_host_constructs_exact_inspected_super_call_without_any_invented_field():
    java = verified_entity_type_source(
        SPEC, CONTRACT, EVIDENCE, package="example", name="Alien")
    assert "extends " + MONSTER in java
    assert "super(type, level);" in java
    assert "this.type" not in java
    assert "ExposeType" not in java
    assert "register(" not in java
    assert "getType()" not in java


@pytest.mark.parametrize("mutate", [
    lambda e: e["class_members"][0].update({"evidence": "model_guess"}),
    lambda e: e["class_members"][0]["members"][0].update({"parameters": [ENTITY_TYPE]}),
    lambda e: e["class_members"][0]["members"][0].update({"access": 2}),
    lambda e: e["class_members"][0]["members"][0].update({"generic_parameters": ["unknown", LEVEL]}),
])
def test_host_refuses_missing_or_inaccessible_real_superclass_proof(mutate):
    import copy
    evidence = copy.deepcopy(EVIDENCE)
    mutate(evidence)
    with pytest.raises(ValueError, match="HOST_ENTITY_"):
        verified_entity_type_source(SPEC, CONTRACT, evidence,
                                    package="example", name="Alien")


def test_nonhostile_entity_falls_back_to_general_authoring():
    spec = {**SPEC, "bindings": {
        **SPEC["bindings"],
        "module_config_json": json.dumps({"behavior": "passive"}),
    }}
    assert verified_entity_type_source(
        spec, CONTRACT, EVIDENCE, package="example", name="Alien") is None


def test_entity_type_shell_does_not_suppress_assigned_gameplay_behaviors():
    assert verified_entity_type_source(
        SPEC, CONTRACT, EVIDENCE, package="example", name="Alien",
        behaviors=[{"behavior_id": "alien.attack", "intent": "Attack players"}]) is None


def test_real_host_route_skips_java_planner_but_still_invokes_full_compiler(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "verified-entity-test",
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._evidence",
        lambda *args, **kwargs: EVIDENCE,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._read_materialized_context",
        lambda *args, **kwargs: [],
    )
    compiled = []
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._verify_full_source",
        lambda java, *args, **kwargs: compiled.append(java),
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_coverage.record_authored_source_calls",
        lambda *args, **kwargs: None,
    )

    class NoModel:
        def generate_structured_data(self, *args, **kwargs):
            raise AssertionError("Host should own verified entity skeleton")
        def generate_java_source(self, *args, **kwargs):
            raise AssertionError("Host should own verified constructor")

    ctx = {
        "target": {"loader": "fabric", "minecraft_version": "26.2", "java_version": 25},
        "project_root": str(tmp_path),
        "prerequisite_hashes": {},
        "java_classpath": ["/checked/classpath"],
        "authoring_contract": CONTRACT,
        "symbols": {}, "binding_pages": {},
    }
    java = compose_host_java_source(
        NoModel(), SPEC, ctx, package="example", name="Alien", owner="example.Alien",
    )
    assert java == compiled[0]
    assert "super(type, level);" in java
    assert len(compiled) == 1
