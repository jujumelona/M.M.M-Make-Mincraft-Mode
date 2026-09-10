from __future__ import annotations

"""Typed artifact ports used to connect leaf jobs without semantic guessing."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PortKind(str, Enum):
    REGISTRY_ID = "REGISTRY_ID"
    JAVA_SYMBOL = "JAVA_SYMBOL"
    TEXTURE_REF = "TEXTURE_REF"
    MODEL_REF = "MODEL_REF"
    CLIENT_ITEM_REF = "CLIENT_ITEM_REF"
    SCREEN_HANDLER_TYPE = "SCREEN_HANDLER_TYPE"
    PAYLOAD_TYPE = "PAYLOAD_TYPE"
    TRANSLATION_KEY = "TRANSLATION_KEY"


class PortConnectionError(ValueError):
    pass


@dataclass(frozen=True)
class TypedPort:
    name: str
    port_kind: PortKind
    target_type: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "port_kind": self.port_kind.value,
            "target_type": self.target_type,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedPort":
        return cls(
            name=str(data["name"]),
            port_kind=PortKind(data["port_kind"]),
            target_type=str(data["target_type"]),
            value=str(data["value"]),
        )

    def matches(self, expected_kind: PortKind | str, expected_target_type: str) -> bool:
        kind = PortKind(expected_kind) if isinstance(expected_kind, str) else expected_kind
        return (
            self.port_kind == kind
            and self.target_type.casefold() == expected_target_type.casefold()
        )


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
            f"PORT_TARGET_TYPE_MISMATCH: Port {port.name} targets {port.target_type}, "
            f"but consumer expected {expected_target_type}"
        )


class PortRegistry:
    """Session-local immutable-by-name port table."""

    def __init__(self) -> None:
        self._ports: dict[str, TypedPort] = {}

    def publish(self, port: TypedPort) -> None:
        if not port.name or not port.value:
            raise PortConnectionError("PORT_EMPTY: published ports need non-empty name and value")
        existing = self._ports.get(port.name)
        if existing is not None and existing != port:
            raise PortConnectionError(
                f"PORT_DUPLICATE: Port {port.name} already has {existing.to_dict()}, "
                f"conflicting with {port.to_dict()}"
            )
        self._ports[port.name] = port

    def resolve(
        self,
        name: str,
        expected_kind: PortKind | str,
        expected_target_type: str,
    ) -> TypedPort:
        port = self._ports.get(name)
        if port is None:
            raise PortConnectionError(
                f"PORT_MISSING: Required port {name!r} has not been published"
            )
        validate_port_compatibility(port, expected_kind, expected_target_type)
        return port

    def get(self, name: str) -> TypedPort | None:
        return self._ports.get(name)

    def has(self, name: str) -> bool:
        return name in self._ports

    def all_ports(self) -> dict[str, TypedPort]:
        return dict(self._ports)
