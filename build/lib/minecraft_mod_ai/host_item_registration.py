"""Evidence-backed HOST facts for Minecraft item registration API epochs.

This module contains deterministic target facts only. Callers must not infer
or substitute a neighbouring Minecraft release when a target is unsupported.
"""

from __future__ import annotations

from packaging.version import Version


_COMMON = {
    "registry_owner": "net.minecraft.core.Registry",
    "builtin_registries_owner": "net.minecraft.core.registries.BuiltInRegistries",
    "registries_owner": "net.minecraft.core.registries.Registries",
    "resource_key_owner": "net.minecraft.resources.ResourceKey",
}


def item_registration_epoch(minecraft_version: str) -> dict[str, object]:
    """Return reviewed Mojang-mapped item registration facts for a target.

    Registration boundaries:
    * through 1.21.1: direct ResourceLocation registration
    * 1.21.2 through 1.21.10: ResourceKey + ResourceLocation + setId
    * 1.21.11 and later: ResourceKey + Identifier + setId

    ResourceLocation construction has its own boundary: its public constructor
    is used before 1.21, while 1.21+ uses fromNamespaceAndPath.
    """

    version = Version(minecraft_version)
    facts: dict[str, object] = dict(_COMMON)

    if version < Version("1.21.2"):
        facts.update(
            id="direct_resource_location",
            resource_identifier_owner="net.minecraft.resources.ResourceLocation",
            resource_identifier_factory=("<init>" if version < Version("1.21") else "fromNamespaceAndPath"),
            resource_identifier_factory_static=version >= Version("1.21"),
            requires_resource_key=False,
            requires_set_id=False,
        )
        return facts

    if version < Version("1.21.11"):
        facts.update(
            id="keyed_resource_location",
            resource_identifier_owner="net.minecraft.resources.ResourceLocation",
            resource_identifier_factory="fromNamespaceAndPath",
            resource_identifier_factory_static=True,
            requires_resource_key=True,
            requires_set_id=True,
        )
        return facts

    facts.update(
        id="keyed_identifier",
        resource_identifier_owner="net.minecraft.resources.Identifier",
        resource_identifier_factory="fromNamespaceAndPath",
        resource_identifier_factory_static=True,
        requires_resource_key=True,
        requires_set_id=True,
    )
    return facts
