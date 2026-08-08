---
name: make-minecraft-mod
description: Plan, revise, build, validate, or resume a Minecraft Java 1.20.1 Fabric mod with M.M.M. Use when the user describes any Minecraft mod idea, asks to change an existing source project, wants a small or very large game-scale mod, or wants the Colab production run continued.
---

# Make Minecraft Mod

Start from the user's words. Do not inject unrequested content or an example
template into the design.

1. Use the `mmm-frontdoor` tools to produce a readable game plan. Show the
   player fantasy, loop, progression, requested mod content, systems, and
   acceptance criteria in normal language.
2. Keep revising that plan through conversation until the user accepts the
   direction. Keep proposal IDs, hashes, and machine contracts out of the
   conversational response.
3. Apply `route-generic-game-research` to broad or unfamiliar requests. Let the
   central planner derive the research domains, then page compatible Modrinth,
   GitHub and license-aware media evidence for the domains that actually need
   them. Never use a named example game as a fixed feature template.
4. Before implementation facts are uncertain, apply
   `gather-adaptive-minecraft-evidence`. Pin Minecraft 1.20.1, Fabric, Yarn,
   Java, dependency, license, and project-local evidence.
5. For accepted work, apply `compile-massive-work-graph`. Preserve every
   requested deliverable, split work into bounded dependency shards, and save
   durable checkpoints instead of truncating scope. In the planning stage,
   keep the opaque `proposal_ref` and page `read_complete_plan_section` until
   `next_cursor` is empty; do not copy a giant proposal object between tools.
6. Use the stage-specific generation and quality MCP servers. A generated file
   is not proof of completion: require the selected source, Gradle, GameTest,
   JAR, Blockbench, runtime, playtest, and visual receipts.
7. If a run or Colab session stops, apply `resume-production-run`. Reuse only
   hash-valid outputs and retry the failed or changed shard.
8. For an existing mod, accept one authorized source/release ZIP only when the
   user explicitly selected modification mode. Inspect and hash it before
   extraction; never execute archive contents during inspection.

Do not claim an infinite machine. M.M.M has no fixed project-wide feature,
module, entity, asset, or audio count cap; physical formats,
available memory, storage, model context, API quotas, and runtime time remain
resource boundaries handled through more shards and sessions.
