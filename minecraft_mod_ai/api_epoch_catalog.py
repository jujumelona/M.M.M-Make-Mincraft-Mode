"""API epoch catalog for version-specific template selection.

P0-5: Different Minecraft/Fabric API epochs require different template implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class APIEpoch:
    """An API epoch representing a set of compatible Minecraft versions."""
    epoch_id: str
    minecraft_versions: list[str]
    description: str
    key_api_changes: list[str]


# P0-5: Define API epochs based on major API changes
API_EPOCHS = {
    "registry_v1": APIEpoch(
        epoch_id="registry_v1",
        minecraft_versions=["1.20.1", "1.20.2", "1.20.3", "1.20.4"],
        description="DeferredRegister API",
        key_api_changes=[
            "Uses DeferredRegister for registry",
            "Items registered via DeferredRegister.register()",
            "No .setId() method on properties",
        ]
    ),
    "registry_v2": APIEpoch(
        epoch_id="registry_v2",
        minecraft_versions=["1.20.5", "1.20.6", "1.21", "1.21.1", "1.21.2", "1.21.3", "1.21.4"],
        description="BuiltInRegistries API",
        key_api_changes=[
            "Uses BuiltInRegistries.ITEM",
            "Direct Registry.register() calls",
            "No .setId() yet",
        ]
    ),
    "registry_v3": APIEpoch(
        epoch_id="registry_v3",
        minecraft_versions=["1.21.5"],
        description="Modern Registry with setId",
        key_api_changes=[
            "BuiltInRegistries with Registry.register()",
            "Item.Properties().setId() method available",
            "Simplified registration API",
        ]
    ),
}

# P0-5: Map canonical leaves to epoch-specific templates
LEAF_TEMPLATE_EPOCHS = {
    "minecraft/item/registry": {
        "registry_v1": "fabric/item/register_deferred_v1",
        "registry_v2": "fabric/item/register_builtin_v2",
        "registry_v3": "fabric/item/register_modern_v3",
    },
    "minecraft/block/registry": {
        "registry_v1": "fabric/block/register_deferred_v1",
        "registry_v2": "fabric/block/register_builtin_v2",
        "registry_v3": "fabric/block/register_modern_v3",
    },
}

# P0-5: Compilation evidence required for each epoch × version
EPOCH_COMPILE_EVIDENCE = {
    ("registry_v1", "1.20.1"): "required",
    ("registry_v1", "1.20.4"): "required",
    ("registry_v2", "1.21.1"): "required",
    ("registry_v2", "1.21.4"): "required",
    ("registry_v3", "1.21.5"): "required",
}


def determine_api_epoch(minecraft_version: str) -> str:
    """Determine which API epoch a Minecraft version belongs to.
    
    P0-5: Replaces threshold-based version checks.
    
    Args:
        minecraft_version: e.g., "1.21.5"
        
    Returns:
        Epoch ID (e.g., "registry_v3")
        
    Raises:
        ValueError: If version not in any epoch
    """
    for epoch_id, epoch in API_EPOCHS.items():
        if minecraft_version in epoch.minecraft_versions:
            return epoch_id
    
    raise ValueError(
        f"Minecraft version {minecraft_version} not mapped to any API epoch. "
        f"Add to API_EPOCHS or mark as unsupported."
    )


def get_template_for_leaf(
    canonical_leaf: str,
    minecraft_version: str,
) -> str:
    """Get epoch-appropriate template for a canonical leaf.
    
    P0-5: Returns version-specific template, not universal one.
    
    Args:
        canonical_leaf: e.g., "minecraft/item/registry"
        minecraft_version: e.g., "1.21.5"
        
    Returns:
        Template ID (e.g., "fabric/item/register_modern_v3")
        
    Raises:
        ValueError: If no template available for this leaf+version
    """
    epoch_id = determine_api_epoch(minecraft_version)
    
    epoch_templates = LEAF_TEMPLATE_EPOCHS.get(canonical_leaf)
    if not epoch_templates:
        raise ValueError(
            f"No epoch templates defined for canonical leaf: {canonical_leaf}"
        )
    
    template_id = epoch_templates.get(epoch_id)
    if not template_id:
        raise ValueError(
            f"No template for leaf {canonical_leaf} in epoch {epoch_id} "
            f"(Minecraft {minecraft_version})"
        )
    
    return template_id


def requires_compile_evidence(
    epoch_id: str,
    minecraft_version: str,
) -> bool:
    """Check if compile evidence is required for this epoch+version.
    
    P0-5: Only admit templates with compile evidence.
    """
    return EPOCH_COMPILE_EVIDENCE.get((epoch_id, minecraft_version)) == "required"


def validate_template_compatibility(
    template_id: str,
    minecraft_version: str,
    compile_evidence: Any = None,
) -> tuple[bool, str]:
    """Validate template is compatible with Minecraft version.
    
    P0-5: Checks:
    1. Template belongs to correct epoch
    2. Compile evidence exists if required
    3. No API mismatches
    
    Returns:
        (is_valid, reason)
    """
    try:
        epoch_id = determine_api_epoch(minecraft_version)
    except ValueError as e:
        return False, str(e)
    
    # Check if template is for this epoch
    template_epoch = None
    for leaf, epochs in LEAF_TEMPLATE_EPOCHS.items():
        if template_id in epochs.values():
            template_epoch = next(
                (eid for eid, tid in epochs.items() if tid == template_id),
                None
            )
            break
    
    if template_epoch and template_epoch != epoch_id:
        return False, (
            f"Template {template_id} is for epoch {template_epoch}, "
            f"but version {minecraft_version} is in epoch {epoch_id}"
        )
    
    # Check compile evidence
    if requires_compile_evidence(epoch_id, minecraft_version):
        if not compile_evidence:
            return False, (
                f"Compile evidence required for {template_id} "
                f"on {minecraft_version} (epoch {epoch_id})"
            )
        
        if compile_evidence.get("status") != "PASS":
            return False, (
                f"Compile evidence exists but status is not PASS: "
                f"{compile_evidence.get('status')}"
            )
    
    return True, "OK"


def list_supported_versions() -> dict[str, list[str]]:
    """List all supported Minecraft versions by epoch."""
    return {
        epoch_id: epoch.minecraft_versions
        for epoch_id, epoch in API_EPOCHS.items()
    }
