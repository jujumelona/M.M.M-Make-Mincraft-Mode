---
name: design-minecraft-worlds
description: Produce validated region and route IR while refusing to claim unsupported Jigsaw, NBT, or world-file compilation.
---

# Design Minecraft Worlds

## activate_when
Use for towns, fields, dungeons, routes, biome placement, quest flow, structures, or map reference images.

## inputs
- World brief.
- Optional reference images.
- Required regions, routes, structures, progression, and traversal constraints.

## required_rag
Use `minecraft-dev` for 1.20.1 structure, biome, Jigsaw, NBT, and worldgen APIs. Use local RAG for pinned platform evidence.

## allowed_tools
- `mmm-local.generate_world_ir`
- `mmm-local.search_project_rag`
- `minecraft-dev` read-only tools

## output_schema
Return `mmm/world-ir-v1` with `regions`, `routes`, `structures`, `quests`, and `constraints`, plus compiler status.

## validators
- Region and structure IDs are unique snake_case.
- Every route references known regions.
- Every structure references a known region.
- Required gameplay loop locations are reachable.
- Compiler status remains `blocked` until an actual Jigsaw/NBT compiler and runtime test exist.

## retry_policy
Repair invalid graph references at most twice. Do not replace missing locations with invented implemented systems.

## approval_required
IR generation may write only the declared IR file. World compilation and runtime placement require a separate approved implementation plugin.

## forbidden
- Claiming a world ZIP, NBT structure, Jigsaw pool, village, or dungeon was produced from IR alone.
- Modifying a user world.
- Using a runtime MCP whose Minecraft version has not passed 1.20.1 compatibility tests.

## exit_conditions
Exit when the IR validates and every noncompiled output is explicitly labeled planning-only.
