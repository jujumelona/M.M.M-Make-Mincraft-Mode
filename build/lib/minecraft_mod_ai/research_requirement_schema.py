from __future__ import annotations

"""Fixed host-owned research facet taxonomy and structural matching hints.

This module intentionally contains no language-model response schema.  Model-owned
seven-facet generation was removed; bounded model slots live in
``research_requirement_template`` and cannot redefine these host facets.
"""

FACETS = (
    "state_lifecycle",
    "interfaces_integration",
    "persistence_reload",
    "server_network_authority",
    "registration_data_resources",
    "failure_edge_cases",
    "verification_testing",
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

__all__ = [
    "FACET_HINTS",
    "FACETS",
    "STOPWORDS",
    "STRUCTURAL_HINTS",
]
