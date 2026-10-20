"""Regression from production-logs (15): never model-author an EntityType holder."""
import json

import pytest

from minecraft_mod_ai.host_verified_entity_registry import (
    verified_entity_registry_source,
)
from minecraft_mod_ai.construction_host_source import compose_host_java_source

E = "net.minecraft.world.entity.EntityType"
B = E + ".Builder"
C = "net.minecraft.world.entity.MobCategory"
R = "net.minecraft.core.Registry"
BI = "net.minecraft.core.registries.BuiltInRegistries"
RS = "net.minecraft.core.registries.Registries"
K = "net.minecraft.resources.ResourceKey"
I = "net.minecraft.resources.Identifier"
SPEC = {
    "language": "java",
    "target_path": "src/main/java/example/AlienRegistry.java",
    "side": "COMMON",
    "requirement": "Register hostile alien",
    "bindings": {
        "package_name": "example", "class_name": "AlienRegistry",
        "module_config_json": json.dumps({
            "spawn_group": "creature", "entity_width": 1.8, "entity_height": 1.8,
            "behavior": "hostile_melee", "name": "Alien",
        }),
    },
}
CONTRACT = {
    "canonical_leaf": "minecraft/entity/registry",
    "bindings": {"mod_id": "testmod", "registry_path": "alien"},
    "host_requirements": {"java_class_header": "plain"},
}


def _member(owner, kind, name, parameters=None, static=True):
    result = {"owner": owner, "kind": kind, "name": name,
              "access": 9 if static else 1}
    if kind == "METHOD":
        result["parameters"] = parameters or []
    return result


EVIDENCE = {
    "source": [{
        "authority": "inspected_materialized_source",
        "signature": "public class Alien extends net.minecraft.world.entity.monster.Monster",
        "source_sha256": "verified-source-sha",
    }],
    "class_members": [{
        "evidence": "actual_target_symbol_index",
        "verified_class": R,
        "members": [_member(R, "METHOD", "register", [R, I, "java.lang.Object"])],
    }],
}
REQUIRED_SYMBOL_ROWS = [
    _member(B, "METHOD", "of", [E + ".EntityFactory", C]),
    _member(B, "METHOD", "sized", ["float", "float"], static=False),
    _member(B, "METHOD", "build", [K], static=False),
    _member(I, "METHOD", "fromNamespaceAndPath",
            ["java.lang.String", "java.lang.String"]),
    _member(K, "METHOD", "create", [K, I]),
    _member(BI, "FIELD", "ENTITY_TYPE"),
    _member(RS, "FIELD", "ENTITY_TYPE"),
    _member(C, "FIELD", "CREATURE"),
]
SYMBOLS = {str(i): row for i, row in enumerate(REQUIRED_SYMBOL_ROWS)}


def test_entity_registry_host_uses_only_verified_registration_contract():
    java = verified_entity_registry_source(
        SPEC, CONTRACT, EVIDENCE, SYMBOLS, package="example",
        name="AlienRegistry",
    )
    assert "public final class AlienRegistry {" in java
    assert "implements " not in java
    assert "public static final " + E + "<Alien> ENTITY_TYPE" in java
    assert "Alien::new" in java
    assert ".sized(1.8F, 1.8F)" in java
    assert R + ".register(" in java
    assert "fromNamespaceAndPath" in java
    assert "super(" not in java
    assert "public static void register()" in java
    assert "new net.minecraft.world.entity.EntityType.EntityFactory" not in java


def test_registry_must_have_proven_builder_and_materialized_entity():
    import copy
    evidence = copy.deepcopy(EVIDENCE)
    evidence["source"] = []
    with pytest.raises(ValueError, match="PREREQUISITE_UNVERIFIED"):
        verified_entity_registry_source(
            SPEC, CONTRACT, evidence, SYMBOLS, package="example",
            name="AlienRegistry",
        )
    symbols = dict(SYMBOLS)
    symbols.pop("1")  # missing sized(float,float) - don't guess API
    with pytest.raises(ValueError, match="API_UNVERIFIED"):
        verified_entity_registry_source(
            SPEC, CONTRACT, EVIDENCE, symbols, package="example",
            name="AlienRegistry",
        )


def test_registry_host_route_never_calls_model_and_keeps_compile_gate(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "verified-registry-test",
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._evidence",
        lambda *args, **kwargs: EVIDENCE,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._read_materialized_context",
        lambda *args, **kwargs: [],
    )
    checked = []
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_file_author._verify_full_source",
        lambda source, *args, **kwargs: checked.append(source),
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_coverage.record_authored_source_calls",
        lambda *args, **kwargs: None,
    )

    class NoModel:
        def generate_structured_data(self, *args, **kwargs):
            raise AssertionError("Model cannot redefine verified entity registry")
        def generate_java_source(self, *args, **kwargs):
            raise AssertionError("Model cannot invent entity factory")

    context = {
        "target": {"loader": "fabric", "minecraft_version": "26.2"},
        "project_root": str(tmp_path),
        "java_classpath": ["/verified/fabric/classpath"],
        "symbols": SYMBOLS,
        "prerequisite_hashes": {"entity": "verified-source-sha"},
        "authoring_contract": CONTRACT, "binding_pages": {},
    }
    generated = compose_host_java_source(
        NoModel(), SPEC, context,
        package="example", name="AlienRegistry", owner="example.AlienRegistry",
    )
    assert generated == checked[0]
    assert "Alien::new" in generated
    assert len(checked) == 1
