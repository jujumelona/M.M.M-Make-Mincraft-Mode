from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_ports import (
    PortConnectionError,
    PortKind,
    PortRegistry,
    TypedPort,
    validate_port_compatibility,
)


def test_typed_port_serialization():
    port = TypedPort(
        name="space.raw_lunite.registry_id",
        port_kind=PortKind.REGISTRY_ID,
        target_type="Item",
        value="space:raw_lunite",
    )
    data = port.to_dict()
    assert data["port_kind"] == "REGISTRY_ID"
    assert data["target_type"] == "Item"
    assert data["value"] == "space:raw_lunite"

    reconstructed = TypedPort.from_dict(data)
    assert reconstructed == port


def test_typed_port_compatibility_validation():
    item_port = TypedPort(
        name="test_item",
        port_kind=PortKind.REGISTRY_ID,
        target_type="Item",
        value="mod:item_foo",
    )

    # Valid match
    validate_port_compatibility(item_port, PortKind.REGISTRY_ID, "Item")
    validate_port_compatibility(item_port, "REGISTRY_ID", "item")

    # Mismatched PortKind
    with pytest.raises(PortConnectionError, match="PORT_KIND_MISMATCH"):
        validate_port_compatibility(item_port, PortKind.JAVA_SYMBOL, "Item")

    # Mismatched TargetType
    with pytest.raises(PortConnectionError, match="PORT_TARGET_TYPE_MISMATCH"):
        validate_port_compatibility(item_port, PortKind.REGISTRY_ID, "Block")


def test_port_registry_publish_and_resolve():
    registry = PortRegistry()
    port = TypedPort(
        name="raw_lunite.registry_id",
        port_kind=PortKind.REGISTRY_ID,
        target_type="Item",
        value="space:raw_lunite",
    )
    registry.publish(port)

    # Resolving with matching types works
    resolved = registry.resolve("raw_lunite.registry_id", PortKind.REGISTRY_ID, "Item")
    assert resolved == port

    # Resolving missing port raises
    with pytest.raises(PortConnectionError, match="PORT_MISSING"):
        registry.resolve("nonexistent_port", PortKind.REGISTRY_ID, "Item")

    # Publishing conflicting duplicate raises
    duplicate_conflict = TypedPort(
        name="raw_lunite.registry_id",
        port_kind=PortKind.REGISTRY_ID,
        target_type="Item",
        value="space:different_value",
    )
    with pytest.raises(PortConnectionError, match="PORT_DUPLICATE"):
        registry.publish(duplicate_conflict)
