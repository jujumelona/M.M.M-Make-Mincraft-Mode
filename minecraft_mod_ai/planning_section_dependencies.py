from __future__ import annotations

"""Host-owned dependency order for canonical structured design sections."""

SECTION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "behavior_contract": (),
    "state_model": ("behavior_contract",),
    "integration": ("behavior_contract",),
    "resources_and_ui": ("behavior_contract", "integration"),
    "algorithm": ("behavior_contract", "state_model"),
    "authority_and_network": ("behavior_contract", "state_model", "integration"),
    "persistence": ("state_model", "integration"),
    "reuse_assessment": ("integration",),
    "failure_and_limits": (
        "behavior_contract",
        "state_model",
        "algorithm",
        "integration",
    ),
    "verification": (
        "behavior_contract",
        "algorithm",
        "integration",
        "failure_and_limits",
    ),
}

__all__ = ["SECTION_DEPENDENCIES"]
