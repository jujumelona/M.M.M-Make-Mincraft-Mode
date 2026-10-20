"""Compile-gated structural entity type source derived from inspected APIs.

Only the entity/type leaf is covered. Registration, AI goals, attacks, world
generation and other gameplay behaviors remain separate feature obligations.
This is not a substitute for Minecraft runtime behavior verification.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping


_JAVA_ID = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_ENTITY_TYPE = "net.minecraft.world.entity.EntityType"
_LEVEL = "net.minecraft.world.level.Level"
_MONSTER = "net.minecraft.world.entity.monster.Monster"


def _hostile_entity_metadata(spec: Mapping, contract: Mapping) -> bool:
    if contract.get("canonical_leaf") != "minecraft/entity/type":
        return False
    bindings = spec.get("bindings") or {}
    if not isinstance(bindings, Mapping):
        return False
    raw = bindings.get("module_config_json")
    if not isinstance(raw, str):
        return False
    try:
        module = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if not isinstance(module, dict):
        return False
    # This is explicitly declared in the artifact specification; never infer
    # hostility from filenames, English prose or guessed Minecraft versions.
    behavior = module.get("behavior")
    return isinstance(behavior, str) and behavior.startswith("hostile_")


def verified_entity_type_source(
    spec: Mapping, contract: Mapping, inspected: Mapping,
    *, package: str, name: str, behaviors=(),
) -> str | None:
    """Produce the canonical hostile entity shell without a Java model call.

    The parent and signature MUST be present in inspected target-classpath
    evidence. If the user requires gameplay callbacks in this compilation
    unit, let their separate authoring path handle them rather than claiming
    that an empty entity shell implemented those behaviors.
    """
    if not _hostile_entity_metadata(spec, contract) or behaviors:
        return None
    if not _JAVA_ID.fullmatch(name) or any(
        not _JAVA_ID.fullmatch(part) for part in package.split(".")
    ):
        raise ValueError("HOST_ENTITY_IDENTITY_INVALID")
    if str(spec.get("side") or "COMMON").upper() != "COMMON":
        raise ValueError("HOST_ENTITY_TYPE_COMMON_SIDE_REQUIRED")
    if not isinstance(inspected, Mapping):
        raise ValueError("HOST_ENTITY_TARGET_EVIDENCE_MISSING")
    classes = inspected.get("class_members")
    if not isinstance(classes, list):
        raise ValueError("HOST_ENTITY_TARGET_EVIDENCE_MISSING")
    matching = [
        row for row in classes
        if isinstance(row, dict) and row.get("verified_class") == _MONSTER
        and row.get("evidence") == "actual_target_symbol_index"
    ]
    if len(matching) != 1:
        raise ValueError("HOST_ENTITY_SUPERCLASS_UNVERIFIED: " + _MONSTER)
    parent = matching[0]
    declaration = parent.get("declaration") or {}
    if declaration.get("kind") != "CLASS" or declaration.get("owner") != _MONSTER:
        raise ValueError("HOST_ENTITY_SUPERCLASS_UNVERIFIED: " + _MONSTER)
    if int(declaration.get("access") or 0) & 1 == 0:
        raise ValueError("HOST_ENTITY_SUPERCLASS_INACCESSIBLE")
    constructors = [
        row for row in parent.get("members", ())
        if isinstance(row, dict) and row.get("kind") == "CONSTRUCTOR"
        and row.get("owner") == _MONSTER
        and row.get("parameters") == [_ENTITY_TYPE, _LEVEL]
        and (int(row.get("access") or 0) & 5) != 0
        and not row.get("throws")
    ]
    if len(constructors) != 1:
        raise ValueError("HOST_ENTITY_CONSTRUCTOR_UNVERIFIED: " + _MONSTER)
    parent_args = constructors[0].get("generic_parameters")
    if not isinstance(parent_args, list) or len(parent_args) != 2:
        raise ValueError("HOST_ENTITY_GENERIC_SIGNATURE_MISSING")
    if parent_args[0] != (_ENTITY_TYPE + "<? extends " + _MONSTER + ">"):
        raise ValueError("HOST_ENTITY_GENERIC_SIGNATURE_UNVERIFIED")
    if parent_args[1] != _LEVEL:
        raise ValueError("HOST_ENTITY_GENERIC_SIGNATURE_UNVERIFIED")
    # This is exactly the inspected superclass constructor; the generated
    # class does not write private Entity fields or invent a type accessor.
    return (
        f"package {package};\n\n"
        f"public class {name} extends {_MONSTER} {{\n"
        f"    public {name}({_ENTITY_TYPE}<? extends {_MONSTER}> type,\n"
        f"            {_LEVEL} level) {{\n"
        "        super(type, level);\n"
        "    }\n"
        "}\n"
    )
