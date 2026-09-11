"""Product support matrix - explicit definition of supported versions and required leaves.

P0-3: This replaces implicit scope derived from first bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Explicitly supported Minecraft versions
SUPPORTED_MINECRAFT_VERSIONS = [
    "1.21.5",
    "1.21.4",
    "1.21.3",
    "1.21.1",
    "1.21",
    "1.20.6",
    "1.20.4",
    "1.20.1",
]

# Core canonical leaves required for all supported versions
REQUIRED_CANONICAL_LEAVES = [
    "fabric/item/key",
    "fabric/item/register_basic",
    "fabric/item/client_item",
    "fabric/item/model_basic",
    "fabric/item/lang_en",
    "fabric/item/initializer",
    "fabric/item/settings_max_stack",
    "fabric/block/key",
    "fabric/block/register_basic",
    "fabric/block/blockstate_basic",
    "fabric/block/model_cube_all",
    "fabric/block/lang_en",
    "fabric/block/initializer",
    "fabric/recipe/shaped",
    "fabric/recipe/shapeless",
    "fabric/recipe/smelting",
    "fabric/tag/registry",
    "fabric/loot/block_drop",
]

# Version-specific requirements (additions to core set)
VERSION_SPECIFIC_REQUIREMENTS: dict[str, list[str]] = {
    "1.21.5": [
        "fabric/modern_registry/item",
        "fabric/modern_registry/block",
    ],
    "1.21.4": [
        "fabric/modern_registry/item",
        "fabric/modern_registry/block",
    ],
    # Older versions don't have modern registry
}

# Versions that explicitly don't support certain leaves
VERSION_SPECIFIC_UNSUPPORTED: dict[str, list[str]] = {
    "1.20.1": [
        "fabric/modern_registry/item",
        "fabric/modern_registry/block",
    ],
}


@dataclass
class SupportMatrixError(Exception):
    """Error in support matrix validation."""
    failures: list[dict[str, Any]]
    
    def __str__(self):
        lines = ["Support matrix validation failed:"]
        for failure in self.failures:
            lines.append(
                f"  - {failure['version']}: {failure['leaf']} -> {failure['status']} "
                f"({failure.get('reason', 'no reason')})"
            )
        return "\n".join(lines)


def get_required_leaves_for_version(minecraft_version: str) -> list[str]:
    """Get all required canonical leaves for a specific version."""
    required = list(REQUIRED_CANONICAL_LEAVES)
    
    # Add version-specific requirements
    if minecraft_version in VERSION_SPECIFIC_REQUIREMENTS:
        required.extend(VERSION_SPECIFIC_REQUIREMENTS[minecraft_version])
    
    return required


def is_leaf_supported_for_version(leaf_id: str, minecraft_version: str) -> bool:
    """Check if a leaf should be supported for a specific version."""
    if minecraft_version in VERSION_SPECIFIC_UNSUPPORTED:
        if leaf_id in VERSION_SPECIFIC_UNSUPPORTED[minecraft_version]:
            return False
    return True


def validate_support_matrix(bundles: list) -> None:
    """Validate that all required leaves are admitted for all supported versions.
    
    P0-3: This replaces the weak audit that only checked first bundle.
    
    Raises:
        SupportMatrixError: If any required leaf is not admitted
    """
    from .resolved_version_context import ResolvedVersionContext
    
    failures = []
    
    for version in SUPPORTED_MINECRAFT_VERSIONS:
        # Find context for this version
        context = None
        for bundle in bundles:
            if isinstance(bundle, ResolvedVersionContext) and bundle.minecraft == version:
                context = bundle
                break
        
        if context is None:
            failures.append({
                "version": version,
                "leaf": "N/A",
                "status": "missing_context",
                "reason": f"No bundle found for supported version {version}",
            })
            continue
        
        # Check all required leaves
        required_leaves = get_required_leaves_for_version(version)
        
        for leaf_id in required_leaves:
            # Get leaf binding status
            binding = context.facts.get("artifact_rules", {}).get(leaf_id, {})
            status = binding.get("status", "not_reviewed")
            
            # P0-3: Both unsupported and not_reviewed fail production readiness
            if status in ["unsupported", "not_reviewed", "unsupported_by_target"]:
                failures.append({
                    "version": version,
                    "leaf": leaf_id,
                    "status": status,
                    "reason": binding.get("reason", "no reason provided"),
                })
    
    if failures:
        raise SupportMatrixError(failures)
