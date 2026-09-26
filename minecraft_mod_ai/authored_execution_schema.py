from __future__ import annotations

"""Single host authority for authored-design execution roles and concern taxonomy."""

from copy import deepcopy
from typing import Any


from .authored_concern_catalog import load_authored_concern_contracts
from .authored_section_ids import (
    CONTEXT_SECTION_SET,
    DOCUMENT_SECTION_ORDER,
    DOCUMENT_SECTION_SET,
    EXECUTION_SECTION_ORDER,
    EXECUTION_SECTION_SET,
    REQUIRED_EXECUTION_SECTIONS,
)

SECTION_SPECS: dict[str, dict[str, Any]] = {
    "state_model": {
        "symbol": "AuthoredStateModel",
        "depends_on": (),
        "responsibility": "Own the authored domain state, invariants, and state transitions.",
        "instruction": (
            "Implement domain state containers, invariants, and transition helpers only. "
            "Do not perform Fabric lifecycle registration, networking, UI, or persistence."
        ),
    },
    "behavior_contract": {
        "symbol": "AuthoredBehaviorContract",
        "depends_on": ("state_model",),
        "responsibility": "Implement the authored player/system behavior contract.",
        "instruction": (
            "Implement bounded gameplay operations and precondition/failure guards over the "
            "authored state. Do not perform Fabric lifecycle registration."
        ),
    },
    "algorithm": {
        "symbol": "AuthoredAlgorithm",
        "depends_on": ("state_model", "behavior_contract"),
        "responsibility": "Implement deterministic algorithms from the authored design.",
        "instruction": (
            "Implement deterministic calculations and algorithms only; keep Minecraft/Fabric "
            "registration and UI/network wiring out of this unit."
        ),
    },
    "authority_and_network": {
        "symbol": "AuthoredAuthorityNetwork",
        "depends_on": ("state_model", "behavior_contract"),
        "responsibility": "Implement server-authoritative synchronization and network-facing rules.",
        "instruction": (
            "Implement only authority/synchronization/network-facing behavior required by the "
            "approved design. Use exact host-provided platform facts; never invent API names."
        ),
    },
    "persistence": {
        "symbol": "AuthoredPersistence",
        "depends_on": ("state_model",),
        "responsibility": "Implement persistence boundaries for authored state.",
        "instruction": (
            "Implement serialization/persistence boundaries for the authored state. "
            "Do not perform unrelated lifecycle registration."
        ),
    },
    "resources_and_ui": {
        "symbol": "AuthoredResourcesUi",
        "depends_on": ("state_model", "behavior_contract"),
        "responsibility": "Implement the bounded UI/resource-facing behavior in the authored design.",
        "instruction": (
            "Implement only UI/resource-facing coordination described by the approved design. "
            "Do not redesign gameplay or invent unavailable platform APIs."
        ),
    },
    "failure_and_limits": {
        "symbol": "AuthoredFailureLimits",
        "depends_on": ("state_model",),
        "responsibility": "Implement failure handling, limits, and invariant guards.",
        "instruction": (
            "Implement validation, limits, and failure guards as deterministic Java logic. "
            "Do not perform Fabric lifecycle registration."
        ),
    },
    "integration": {
        "symbol": "AuthoredIntegration",
        "depends_on": (
            "state_model", "behavior_contract", "algorithm", "authority_and_network",
            "persistence", "resources_and_ui", "failure_and_limits",
        ),
        "responsibility": "Wire the authored runtime systems into the host-owned mod lifecycle.",
        "instruction": (
            "This is the integration unit. Wire already implemented authored systems into the "
            "host-owned initialize() hook. Use only exact host-provided Minecraft/Fabric facts "
            "and dependency source; never create another mod entrypoint."
        ),
    },
}

_SYMBOL_TO_SECTION = {
    str(spec["symbol"]): section for section, spec in SECTION_SPECS.items()
}


def section_spec(section: str) -> dict[str, Any] | None:
    value = SECTION_SPECS.get(str(section or "").strip())
    return deepcopy(value) if value is not None else None


def section_for_symbol(symbol: str) -> str:
    return _SYMBOL_TO_SECTION.get(str(symbol or "").strip(), "")


def concern_contracts(section: str) -> tuple[dict[str, Any], ...]:
    """Return fixed host-ordered concern contracts from the canonical YAML catalog."""
    name = str(section or "").strip()
    if name not in EXECUTION_SECTION_SET:
        return ()
    return load_authored_concern_contracts(name)


def concern_names(section: str) -> tuple[str, ...]:
    return tuple(item["concern"] for item in concern_contracts(section))


__all__ = [
    "CONTEXT_SECTION_SET",
    "DOCUMENT_SECTION_ORDER",
    "DOCUMENT_SECTION_SET",
    "EXECUTION_SECTION_ORDER",
    "EXECUTION_SECTION_SET",
    "REQUIRED_EXECUTION_SECTIONS",
    "SECTION_SPECS",
    "concern_contracts",
    "concern_names",
    "section_for_symbol",
    "section_spec",
]
