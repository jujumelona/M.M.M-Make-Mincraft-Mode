---
name: create-minecraft-assets
description: Generate texture concepts and Minecraft-sized PNG candidates with exclusive GPU scheduling and visual review gates.
---

# Create Minecraft Assets

## activate_when
Use for item/block texture concepts, icons, GUI references, environment concepts, or visual consistency review.

## inputs
- One to sixteen snake_case asset IDs.
- Concrete visual brief for each ID.
- Optional palette/reference images.
- Deterministic seed.

## required_rag
No web RAG is required for original art. Research external visual references only when licensing and provenance are recorded.

## allowed_tools
- `mmm-local.generate_assets`
- Planner/visual-critic model roles through the configured router

## output_schema
Return concept PNG path/hash, 16x16 texture PNG path/hash, seed, model role, and visual-review warning for every asset.

## validators
- FLUX/image role runs with exclusive GPU ownership.
- Output path is workspace-contained.
- PNG decodes successfully and texture size is exactly 16x16.
- IDs match resource paths and have no collision.
- VisualCritic review occurs before release use.

## retry_policy
One regeneration is allowed for transparent failure, unreadable silhouette, text/watermark, or severe style mismatch. Preserve both candidates and hashes.

## approval_required
Asset generation writes files and requires an approved plan when attached to a release workflow.

## forbidden
- Running image generation concurrently with a resident T4 LLM.
- Copying copyrighted game assets or logos.
- Describing procedural color pixels as AI-generated art.
- Claiming Blockbench/GeckoLib export or UV validation without executing those tools.

## exit_conditions
Exit when all PNGs pass format checks and are either approved by VisualCritic or explicitly rejected.
