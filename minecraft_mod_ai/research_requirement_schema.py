from __future__ import annotations

"""Shared fixed schema for research-derived implementation facets."""

from typing import Any

FACETS = (
    "state_lifecycle",
    "interfaces_integration",
    "persistence_reload",
    "server_network_authority",
    "registration_data_resources",
    "failure_edge_cases",
    "verification_testing",
)

DISPOSITIONS = frozenset(
    {"derived", "already_covered", "not_applicable", "unresolved"}
)

FACET_HINTS: dict[str, tuple[str, ...]] = {
    "state_lifecycle": (
        "lifecycle",
        "state transition",
        "state",
        "init",
        "tick",
        "update",
        "cleanup",
        "dispose",
        "ownership",
    ),
    "interfaces_integration": (
        "interface",
        "integration",
        "api",
        "hook",
        "event",
        "callback",
        "service",
        "binding",
    ),
    "persistence_reload": (
        "persist",
        "persistent",
        "save",
        "load",
        "reload",
        "serialize",
        "codec",
        "nbt",
        "world state",
    ),
    "server_network_authority": (
        "server-authoritative",
        "server authoritative",
        "server",
        "client",
        "network",
        "packet",
        "payload",
        "sync",
        "multiplayer",
    ),
    "registration_data_resources": (
        "registry",
        "register",
        "resource",
        "datapack",
        "tag",
        "recipe",
        "loot",
        "model",
        "language",
        "worldgen",
    ),
    "failure_edge_cases": (
        "failure",
        "invalid",
        "reject",
        "insufficient",
        "incompatible",
        "missing",
        "exception",
        "fallback",
        "edge case",
        "locked",
    ),
    "verification_testing": (
        "test",
        "gametest",
        "verify",
        "verification",
        "validation",
        "assert",
        "regression",
        "runtime check",
    ),
}

STRUCTURAL_HINTS: dict[str, tuple[str, ...]] = {
    "state_lifecycle": (
        "state",
        "transition",
        "lifecycle",
        "tick",
        "update",
        "ownership",
        "damage",
        "transaction",
        "behavior",
        "assignment",
        "placement",
    ),
    "interfaces_integration": (
        "service",
        "binding",
        "integration",
        "surface",
        "handler",
        "menu",
        "screen",
        "interaction",
        "world_transition",
        "domain_service",
    ),
    "persistence_reload": FACET_HINTS["persistence_reload"],
    "server_network_authority": FACET_HINTS["server_network_authority"],
    "registration_data_resources": FACET_HINTS["registration_data_resources"],
    "failure_edge_cases": FACET_HINTS["failure_edge_cases"],
    "verification_testing": FACET_HINTS["verification_testing"],
}

STOPWORDS = frozenset(
    {
        "add",
        "make",
        "create",
        "implement",
        "minecraft",
        "mod",
        "mechanic",
        "system",
        "feature",
        "with",
        "from",
        "into",
        "that",
        "this",
        "then",
        "and",
        "for",
        "the",
        "a",
        "an",
        "capability",
        "requirement",
    }
)

FACET_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facet": {"type": "string", "enum": list(FACETS)},
        "disposition": {
            "type": "string",
            "enum": [
                "derived",
                "already_covered",
                "not_applicable",
                "unresolved",
            ],
        },
        "statement": {"type": "string"},
        "rationale": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "acceptance": {"type": "array", "items": {"type": "string"}},
        "implementation_obligations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "facet",
        "disposition",
        "statement",
        "rationale",
        "evidence_refs",
        "acceptance",
        "implementation_obligations",
    ],
    "additionalProperties": False,
}

REQUIREMENT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facets": {
            "type": "array",
            "minItems": len(FACETS),
            "maxItems": len(FACETS),
            "items": FACET_ITEM_SCHEMA,
        }
    },
    "required": ["facets"],
    "additionalProperties": False,
}

__all__ = [
    "DISPOSITIONS",
    "FACET_HINTS",
    "FACET_ITEM_SCHEMA",
    "FACETS",
    "REQUIREMENT_RESPONSE_SCHEMA",
    "STOPWORDS",
    "STRUCTURAL_HINTS",
]
