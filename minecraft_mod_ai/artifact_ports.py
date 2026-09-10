from __future__ import annotations

"""Typed ports for artifact connection and composition.

Artifacts communicate across Minecraft domain boundaries using statically typed
ports (RegistryId<Item>, TextureRef<Block>, JavaSymbol<Item>, etc.) preventing
invalid inter-artifact connections.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PortKind(str, Enum):
    REGISTRY_ID = "REGISTRY_ID"
    JAVA_SYMBOL = "JAVA_SYMBOL"
    TEXTURE_REF = "TEXTURE_REF"
    MODEL_REF = "MODEL_REF"
    SCREEN_HANDLER_TYPE = "SCREEN_HANDLER_TYPE"
    PAYLOAD_TYPE = "PAYLOAD_TYPE"
    TRANSLATION_KEY = "TRANSLATION_KEY"


class PortConnectionError(ValueError):
    """Raised when an incompatible port is connected or required port is missing."""
    pass


@dataclass(frozen=True)
class TypedPort:
    name: str
    port_kind: PortKind
    target_type: str  # e.g. "Item", "Block", "EntityType", "C2S", etc.
    value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "port_kind": self.port_kind.value,
            "target_type": self.target_type,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypedPort:
        return cls(
            name=str(data["name"]),
            port_kind=PortKind(data["port_kind"]),
            target_type=str(data["target_type"]),
            value=str(data["value"]),
        )

    def matches(self, expected_kind: PortKind | str, expected_target_type: str) -> bool:
        kind = PortKind(expected_kind) if isinstance(expected_kind, str) else expected_kind
        return self.port_kind == kind and self.target_type.casefold() == expected_target_type.casefold()


def validate_port_compatibility(
    port: TypedPort,
    expected_kind: PortKind | str,
    expected_target_type: str,
) -> None:
    expected = PortKind(expected_kind) if isinstance(expected_kind, str) else expected_kind
    if port.port_kind != expected:
        raise PortConnectionError(
            f"PORT_KIND_MISMATCH: Port {port.name} has kind {port.port_kind.value}, "
            f"but consumer expected {expected.value}"
        )
    if port.target_type.casefold() != expected_target_type.casefold():
        raise PortConnectionError(
            f"PORT_TARGET_TYPE_MISMATCH: Port {port.name} targets type {port.target_type}, "
            f"but consumer expected {expected_target_type}"
        )


class PortRegistry:
    """Session container for all published and resolved typed ports."""

    def __init__(self) -> None:
        self._ports: dict[str, TypedPort] = {}

    def publish(self, port: TypedPort) -> None:
        key = port.name
        if key in self._ports:
            existing = self._ports[key]
            if existing != port:
                raise PortConnectionError(
                    f"PORT_DUPLICATE: Port {key} is already published with {existing.to_dict()}, "
                    f"conflicting with {port.to_dict()}"
                )
        self._ports[key] = port

    def resolve(
        self,
        name: str,
        expected_kind: PortKind | str,
        expected_target_type: str,
    ) -> TypedPort:
        if name not in self._ports:
            raise PortConnectionError(f"PORT_MISSING: Required port {name!r} has not been published")
        port = self._ports[name]
        validate_port_compatibility(port, expected_kind, expected_target_type)
        return port

    def get(self, name: str) -> TypedPort | None:
        return self._ports.get(name)

    def has(self, name: str) -> bool:
        return name in self._ports

    def all_ports(self) -> dict[str, TypedPort]:
        return dict(self._ports)
