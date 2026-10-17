---
name: generate-textures
description: RETIRED — M.M.M does not generate AI textures or standalone resource packs.
---

This skill is **retired** and must not be selected for new or resumed production.

- Do not load FLUX, image diffusion, rembg/BiRefNet, or run `generate_assets`.
- Do not emit standalone `generated-resource-pack.zip`.
- Fabric mod-owned model JSON, blockstates, recipes, and existing PNGs remain normal mod assets.
- For new generated content, use supported vanilla Minecraft texture references as an explicitly generic visual fallback. Do not assert custom art was generated.
- If a user requires original custom artwork, report that asset generation is unsupported; do not fabricate an alternative image output or claim completion.

No activation, tool invocation, retry, or success condition exists for this retired skill.
