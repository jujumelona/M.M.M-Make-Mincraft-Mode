from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.spec import (
    SpecValidationError,
    _validate_host_facts_namespace,
)


def _facts(*owners: str) -> str:
    return json.dumps(
        {
            "schema_version": "mmm/host-facts-v1",
            "api_symbols": {
                f"symbol_{index}": {"owner": owner}
                for index, owner in enumerate(owners)
            },
        },
        sort_keys=True,
    )


def test_host_facts_reject_mixed_mojang_and_yarn_namespaces() -> None:
    with pytest.raises(
        SpecValidationError,
        match="mix Mojang and Yarn API namespaces",
    ):
        _validate_host_facts_namespace(
            _facts(
                "net.minecraft.core.registries.BuiltInRegistries",
                "net.minecraft.registry.Registries",
            ),
            mappings_kind="mojang",
            native_names=False,
        )


def test_mojang_target_rejects_yarn_host_api_names() -> None:
    with pytest.raises(
        SpecValidationError,
        match="Yarn API namespaces for a Mojang-mapped target",
    ):
        _validate_host_facts_namespace(
            _facts(
                "net.minecraft.registry.Registry",
                "net.minecraft.util.Identifier",
            ),
            mappings_kind="mojang",
            native_names=False,
        )


def test_yarn_target_rejects_mojang_host_api_names() -> None:
    with pytest.raises(
        SpecValidationError,
        match="Mojang API namespaces for a Yarn-mapped target",
    ):
        _validate_host_facts_namespace(
            _facts(
                "net.minecraft.core.Registry",
                "net.minecraft.world.item.Item",
            ),
            mappings_kind="yarn",
            native_names=False,
        )


@pytest.mark.parametrize(
    ("mappings_kind", "owners"),
    [
        (
            "mojang",
            (
                "net.minecraft.core.Registry",
                "net.minecraft.world.item.Item",
            ),
        ),
        (
            "yarn",
            (
                "net.minecraft.registry.Registry",
                "net.minecraft.item.Item",
            ),
        ),
    ],
)
def test_host_facts_accept_one_consistent_mapping_family(
    mappings_kind: str,
    owners: tuple[str, ...],
) -> None:
    _validate_host_facts_namespace(
        _facts(*owners),
        mappings_kind=mappings_kind,
        native_names=False,
    )
