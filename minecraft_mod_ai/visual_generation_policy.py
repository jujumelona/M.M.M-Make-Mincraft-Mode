"""M.M.M visual production policy.

Standalone texture/resource packs and FLUX/other generative-image asset
production are deliberately removed from the supported production contract.
Fabric mod-owned resource JSON and existing/proven vanilla texture references
are still supported: those are game assets, not standalone texture-pack output.
"""
from __future__ import annotations

AI_TEXTURE_GENERATION_ENABLED = False
STANDALONE_RESOURCE_PACK_GENERATION_ENABLED = False

AI_TEXTURE_GENERATION_REMOVED = (
    "AI_TEXTURE_GENERATION_REMOVED: FLUX/other generative-image textures and "
    "standalone texture packs are no longer produced. Use existing mod-owned "
    "resources or validated vanilla Minecraft texture references."
)


def reject_generated_assets(assets: object) -> None:
    """Fail closed instead of loading FLUX or silently dropping old assets."""
    if assets:
        raise ValueError(AI_TEXTURE_GENERATION_REMOVED)
