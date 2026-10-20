"""Host-owned Fabric entity registry generated only from inspected 26.2 APIs.

Never ask a 9B coder to reinvent a Registry holder's superclass, field type,
lambda factory, ResourceKey or builder syntax. The existing project compiler
still verifies every emitted byte against the actual target classpath.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

from .java_operation_ir import parameter_types

_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_ID_PATH = re.compile(r"[a-z0-9_.-]+")
_ENTITY = "net.minecraft.world.entity.EntityType"
_BUILDER = _ENTITY + ".Builder"
_FACTORY = _ENTITY + ".EntityFactory"
_CATEGORY = "net.minecraft.world.entity.MobCategory"
_REGISTRY = "net.minecraft.core.Registry"
_BUILTIN = "net.minecraft.core.registries.BuiltInRegistries"
_REGISTRIES = "net.minecraft.core.registries.Registries"
_RESOURCE_KEY = "net.minecraft.resources.ResourceKey"
_IDENTIFIER = "net.minecraft.resources.Identifier"
_MOB = "net.minecraft.world.entity.Mob"
_LIVING = "net.minecraft.world.entity.LivingEntity"
_MONSTER = "net.minecraft.world.entity.monster.Monster"
_ATTR_REGISTRY = "net.fabricmc.fabric.api.object.builder.v1.entity.FabricDefaultAttributeRegistry"
_ATTR_BUILDER = "net.minecraft.world.entity.ai.attributes.AttributeSupplier.Builder"


def _model_free_registry_fields(spec: Mapping, contract: Mapping):
    if contract.get("canonical_leaf") != "minecraft/entity/registry":
        return None
    bindings = spec.get("bindings") or {}
    raw = bindings.get("module_config_json")
    if not isinstance(raw, str):
        raise ValueError("JAVA_HOST_REGISTRY_CONFIG_REQUIRED")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValueError("JAVA_HOST_REGISTRY_CONFIG_INVALID") from error
    if not isinstance(data, dict):
        raise ValueError("JAVA_HOST_REGISTRY_CONFIG_INVALID")
    identity = contract.get("bindings") or {}
    modid = identity.get("mod_id")
    registry_path = identity.get("registry_path")
    if not isinstance(modid, str) or not _ID_PATH.fullmatch(modid):
        raise ValueError("JAVA_HOST_REGISTRY_MOD_ID_UNVERIFIED")
    if not isinstance(registry_path, str) or not _ID_PATH.fullmatch(registry_path):
        raise ValueError("JAVA_HOST_REGISTRY_PATH_UNVERIFIED")
    category = str(data.get("spawn_group") or data.get("category") or "").upper()
    if not _IDENT.fullmatch(category):
        raise ValueError("JAVA_HOST_REGISTRY_CATEGORY_UNVERIFIED")
    width = data.get("entity_width", data.get("width"))
    height = data.get("entity_height", data.get("height"))
    try:
        width, height = float(width), float(height)
    except (TypeError, ValueError) as error:
        raise ValueError("JAVA_HOST_REGISTRY_DIMENSIONS_REQUIRED") from error
    if not all(math.isfinite(x) and x > 0 for x in (width, height)):
        raise ValueError("JAVA_HOST_REGISTRY_DIMENSIONS_INVALID")
    return modid, registry_path, category, width, height


def _members(inspected: Mapping, symbols: Mapping):
    seen = set()
    for cls in inspected.get("class_members", ()):
        if not isinstance(cls, Mapping):
            continue
        if cls.get("evidence") != "actual_target_symbol_index":
            continue
        for member in cls.get("members", ()):
            if isinstance(member, Mapping):
                row = dict(member)
                key = (row.get("owner"), row.get("kind"), row.get("name"),
                       tuple(row.get("parameters", ())))
                if key not in seen:
                    seen.add(key)
                    yield row
    for row in symbols.values():
        if isinstance(row, Mapping) and row.get("provenance") != "construction_plan":
            if row.get("owner") not in {
                _BUILDER, _CATEGORY, _REGISTRY, _BUILTIN,
                _REGISTRIES, _RESOURCE_KEY, _IDENTIFIER,
                _MOB, _LIVING, _ATTR_REGISTRY,
            }:
                continue
            key = (row.get("owner"), row.get("kind"), row.get("name"),
                   tuple(parameter_types(row) if row.get("kind") in {"METHOD", "CONSTRUCTOR"}
                         else ()))
            if key not in seen:
                seen.add(key)
                yield row


def verified_entity_registry_source(
    spec: Mapping, contract: Mapping, inspected: Mapping, symbols: Mapping,
    *, package: str, name: str, behaviors=(),
) -> str | None:
    args = _model_free_registry_fields(spec, contract)
    if args is None or behaviors:
        return None
    if (not _IDENT.fullmatch(name)
            or not package or any(not _IDENT.fullmatch(x) for x in package.split("."))):
        raise ValueError("JAVA_HOST_REGISTRY_IDENTITY_INVALID")
    if str(spec.get("side") or "COMMON").upper() != "COMMON":
        raise ValueError("JAVA_HOST_REGISTRY_SERVER_SIDE_REQUIRED")
    if not isinstance(inspected, Mapping) or not isinstance(symbols, Mapping):
        raise ValueError("JAVA_HOST_REGISTRY_SYMBOL_INDEX_REQUIRED")
    modid, path, category, width, height = args
    source = inspected.get("source") or []
    entity_name = name.removesuffix("Registry")
    if not entity_name or not _IDENT.fullmatch(entity_name):
        raise ValueError("JAVA_HOST_REGISTRY_ENTITY_NAME_REQUIRED")
    # An actual materialized prerequisite must contain the entity factory
    # constructor. A word match in the model prompt is not sufficient proof.
    if not any(isinstance(row, Mapping)
               and row.get("authority") == "inspected_materialized_source"
               and str(row.get("signature") or "").startswith("public class " + entity_name + " ")
               and row.get("source_sha256")
               for row in source):
        raise ValueError("JAVA_HOST_REGISTRY_ENTITY_PREREQUISITE_UNVERIFIED")
    # The prerequisite source, not a prompt keyword, proves this type is a
    # Mob/Monster that must receive default attributes during initialization.
    owner_declarations = [
        str(row.get("signature") or "") for row in source
        if isinstance(row, Mapping)
        and row.get("authority") == "inspected_materialized_source"
        and str(row.get("signature") or "").startswith("public class " + entity_name + " ")
    ]
    inherited = next((
        match.group(1) for declaration in owner_declarations
        if (match := re.search(r"\bextends\s+([\w.$]+)", declaration))
    ), "")
    # Walk actual class hierarchy, not English hints or a hand-written list
    # of mod prompts. This also covers PathfinderMob and other descendants.
    attribute_owner = None
    visited = set()
    while inherited and inherited not in visited:
        visited.add(inherited)
        if inherited in {_MONSTER, _MOB}:
            attribute_owner = _MOB
            break
        if inherited == _LIVING:
            attribute_owner = _LIVING
            break
        if inherited == "net.minecraft.world.entity.Entity":
            inherited = ""
            break
        parent = next((
            row.get("superclass") for row in symbols.values()
            if isinstance(row, Mapping)
            and row.get("kind") == "CLASS"
            and row.get("owner") == inherited
            and row.get("provenance") != "construction_plan"
        ), None)
        if not isinstance(parent, str) or not parent:
            raise ValueError("JAVA_HOST_ENTITY_SUPERCLASS_LINEAGE_UNVERIFIED: " + inherited)
        inherited = parent
    if inherited in visited and not attribute_owner:
        raise ValueError("JAVA_HOST_ENTITY_SUPERCLASS_CYCLE")
    rows = list(_members(inspected, symbols))

    def proven(owner, kind, member, params=()):
        for row in rows:
            if row.get("owner") != owner or row.get("kind") != kind:
                continue
            if row.get("name") != member:
                continue
            actual = (row.get("parameters") if "parameters" in row else
                      parameter_types(row) if kind in {"METHOD", "CONSTRUCTOR"} else [])
            if kind == "METHOD" and list(actual or ()) != list(params):
                continue
            if not int(row.get("access") or 0) & 1:
                continue
            if kind == "METHOD" and not (int(row.get("access") or 0) & 8
                                        or owner == _BUILDER and member in {"build", "sized"}):
                continue
            return True
        return False

    # These exact callable contracts are checked on the target version before
    # generation, never chosen from a broad class name by the model.
    required = [
        (_BUILDER, "of", [_FACTORY, _CATEGORY]),
        (_BUILDER, "sized", ["float", "float"]),
        (_BUILDER, "build", [_RESOURCE_KEY]),
        (_REGISTRY, "register", [_REGISTRY, _IDENTIFIER, "java.lang.Object"]),
        (_IDENTIFIER, "fromNamespaceAndPath", ["java.lang.String", "java.lang.String"]),
        (_RESOURCE_KEY, "create", [_RESOURCE_KEY, _IDENTIFIER]),
    ]
    for owner, method, params in required:
        if not proven(owner, "METHOD", method, params):
            raise ValueError("JAVA_HOST_REGISTRY_API_UNVERIFIED: " + owner + "." + method)
    for owner, field in (
        (_BUILTIN, "ENTITY_TYPE"),
        (_REGISTRIES, "ENTITY_TYPE"),
        (_CATEGORY, category),
    ):
        if not proven(owner, "FIELD", field):
            raise ValueError("JAVA_HOST_REGISTRY_FIELD_UNVERIFIED: " + owner + "." + field)
    if attribute_owner:
        # Verify both signatures in the selected version, rather than
        # assuming Mob and LivingEntity expose identical builders.
        if not proven(_ATTR_REGISTRY, "METHOD", "register", [_ENTITY, _ATTR_BUILDER]):
            raise ValueError("JAVA_HOST_ENTITY_ATTRIBUTES_API_UNVERIFIED")
        builder_method = ("createMobAttributes" if attribute_owner == _MOB
                          else "createLivingAttributes")
        if not proven(attribute_owner, "METHOD", builder_method, []):
            raise ValueError("JAVA_HOST_ENTITY_ATTRIBUTE_BUILDER_UNVERIFIED")
    else:
        builder_method = ""
    attributes = (
        f"        {_ATTR_REGISTRY}.register(ENTITY_TYPE, {attribute_owner}.{builder_method}());\n"
        if attribute_owner else ""
    )
    ident = f'{_IDENTIFIER}.fromNamespaceAndPath("{modid}", "{path}")'
    return (
        f"package {package};\n\n"
        f"public final class {name} {{\n"
        f"    private {name}() {{}}\n"
        f"    public static final {_ENTITY}<{entity_name}> ENTITY_TYPE =\n"
        f"        {_REGISTRY}.register(\n"
        f"            {_BUILTIN}.ENTITY_TYPE, {ident},\n"
        f"            {_BUILDER}.of({entity_name}::new, {_CATEGORY}.{category})\n"
        f"                .sized({width:g}F, {height:g}F)\n"
        f"                .build({_RESOURCE_KEY}.create({_REGISTRIES}.ENTITY_TYPE, {ident})));\n"
        f"    public static void register() {{\n"
        "        java.util.Objects.requireNonNull(ENTITY_TYPE);\n"
        f"{attributes}"
        "    }\n"
        "}\n"
    )
