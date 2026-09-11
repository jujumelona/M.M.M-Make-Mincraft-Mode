"""Evidence-backed HOST facts for Minecraft item registration API epochs.

This module contains deterministic target facts only.  Callers must not infer
or substitute a neighbouring Minecraft release when a target is unsupported.
"""

from __future__ import annotations

from packaging.version import Version


_DIRECT_RESOURCE_LOCATION = {
    "id": "direct_resource_location",
    "registry_owner": "net.minecraft.core.Registry",
    "builtin_registries_owner": "net.minecraft.core.registries.BuiltInRegistries",
    "registries_owner": "net.minecraft.core.registries.Registries",
    "resource_key_owner": "net.minecraft.resources.ResourceKey",
    "resource_identifier_owner": "net.minecraft.resources.ResourceLocation",
    "resource_identifier_factory": "fromNamespaceAndPath",
    "requires_resource_key": False,
    "requires_set_id": False,
}

_KEYED_RESOURCE_LOCATION = {
    "id": "keyed_resource_location",
    "registry_owner": "net.minecraft.core.Registry",
    "builtin_registries_owner": "net.minecraft.core.registries.BuiltInRegistries",
    "registries_owner": "net.minecraft.core.registries.Registries",
    "resource_key_owner": "net.minecraft.resources.ResourceKey",
    "resource_identifier_owner": "net.minecraft.resources.ResourceLocation",
    "resource_identifier_factory": "fromNamespaceAndPath",
    "requires_resource_key": True,
    "requires_set_id": True,
}

_KEYED_IDENTIFIER = {
    "id": "keyed_identifier",
    "registry_owner": "net.minecraft.core.Registry",
    "builtin_registries_owner": "net.minecraft.core.registries.BuiltInRegistries",
    "registries_owner": "net.minecraft.core.registries.Registries",
    "resource_key_owner": "net.minecraft.resources.ResourceKey",
    "resource_identifier_owner": "net.minecraft.resources.Identifier",
    "resource_identifier_factory": "fromNamespaceAndPath",
    "requires_resource_key": True,
    "requires_set_id": True,
}


def item_registration_epoch(minecraft_version: str) -> dict[str, object]:
    """Return the reviewed Mojang-mapped item registration facts for a target.

    Boundaries are deliberately explicit:
    * through 1.21.1: direct ResourceLocation registration
    * 1.21.2 through 1.21.10: ResourceKey + ResourceLocation + setId
    * 1.21.11 and later: ResourceKey + Identifier + setId

    A fresh dictionary is returned so downstream code cannot mutate HOST
    authority shared by another resolved context.
    """

    version = Version(minecraft_version)
    if version < Version("1.21.2"):
        return dict(_DIRECT_RESOURCE_LOCATION)
    if version < Version("1.21.11"):
        return dict(_KEYED_RESOURCE_LOCATION)
    return dict(_KEYED_IDENTIFIER)
