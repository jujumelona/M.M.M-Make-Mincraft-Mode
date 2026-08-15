# Minecraft Fabric Mod Development Methods

## Product boundary

M.M.M creates or patches Minecraft Java on the host-selected executable target mod projects.

It does **not** create standalone world saves, downloadable map ZIPs, schematics, Litematica files, or block-delta handoffs for an external map Builder. World generation remains available only as code and data owned by a requested mod: structures, biomes, dimensions, ores, configured features, placed features, processors, tags, and bootstrap or registration code.

## Method resolution

Before generation, `resolve_mod_development_methods` classifies the request and returns:

- the common production methods required by every mod;
- optional methods selected by the requested behavior;
- the evidence needed before code or dependencies are reused;
- the release gates that must pass;
- an explicit `standalone_map_generation: false` boundary.

The same catalog is exposed through the mod-only MCP server:

```text
python -m minecraft_mod_ai.mod_generation_mcp_server
```

Its `mmm://mod-development/methods` resource describes the reviewed method catalog, and `resolve_mod_methods` selects methods for one request.

## Common baseline

### 1. Fabric project contract

Lock the exact Minecraft, Fabric Loader, Fabric API, Yarn, Loom, Gradle, and Java versions before source generation. Generated projects must include a coherent `settings.gradle`, `build.gradle`, `gradle.properties`, `fabric.mod.json`, dependency license receipts, and immutable dependency hashes.

Required gates:

- Gradle dependency resolution;
- Java 17 compilation;
- version and mapping consistency;
- no dependency copied before compatibility and license verification.

### 2. Client and server boundary

Common initializers, registries, commands, networking handlers, persistence, and gameplay state must remain dedicated-server safe. Rendering, screens, key bindings, model layers, and client-only callbacks belong in the client initializer and client source set.

Required gates:

- dedicated-server classloading without client classes;
- client startup when client features are requested;
- environment annotation and entrypoint validation.

### 3. Registries and data generation

Use typed registration modules and data generation instead of manually duplicating repeated JSON. Depending on the request, this includes items, blocks, block entities, entities, status effects, enchantments, sounds, recipes, loot tables, tags, models, blockstates, language files, advancements, and worldgen data.

Required gates:

- registry presence tests;
- resource identifier and JSON validation;
- recipe, loot, tag, and model reference checks;
- GameTests for requested observable behavior.

### 4. Validation and release

A generated source tree is not considered a verified release merely because files exist. Release evidence includes static policy validation, Eclipse JDT LS diagnostics, Gradle, GameTest, dedicated-server loading, JAR inspection, requested runtime checks, SBOM, provenance, and license receipts.

## Optional methods

| Method | Selected when | Implementation rule | Required checks |
|---|---|---|---|
| `content_registry` | Items, blocks, crops, food, tools, weapons, armor, machines, effects, enchantments | Typed registrars plus generated resources | Compile, registry presence, recipe/loot/resource validation |
| `events_mixins_access` | Vanilla behavior hooks, events, Mixins, access changes | Prefer Fabric events; use Mixins or access wideners only where public APIs are insufficient | Mapping-pinned target validation, dedicated-server loading, behavior GameTest |
| `gui_and_networking` | GUI, HUD, menus, packets, synchronized actions | Screen handlers and typed packets; every mutation remains server-authoritative | Decode validation, permissions, replay/rate-limit tests, two-sided runtime check |
| `persistent_game_state` | Quests, classes, skills, economy, parties, progression, saved state | Server-owned state with schema version, atomic persistence, migration, fallback | Restart persistence, migration, corruption fallback, multiplayer authority |
| `configuration` | User or server options, tuning, gamerules | Validated schema, defaults, ownership rules, migration | Invalid-config fallback and server ownership tests |
| `entity_rendering_animation` | Mobs, bosses, entities, custom rendering, GeckoLib | Entity type, attributes, goals, renderer/model bindings, animation assets | Dedicated-server compile, spawn GameTest, runtime animation review |
| `fabric_worldgen` | Structures, biomes, dimensions, ores, configured or placed features | Mod-owned code and datapack resources only; never a world save or map artifact | Datapack schema validation, fresh-world generation, upgrade compatibility |
| `commands_permissions_multiplayer` | Commands, permissions, servers, multiplayer operations | Brigadier commands, explicit permissions, server-side state mutation, concurrency-safe operations | Permission, two-client, concurrency, and restart tests |
| `existing_project_patch` | Existing owned source project modification or porting | Inspect source and Gradle graph, apply SHA-256 guarded transactional patches, keep rollback receipts | Compile before and after, regression tests, rollback test |

## Events, Mixins, and access wideners

The decision order is fixed:

1. use a supported Fabric API event or callback;
2. use a normal registry, command, networking, data, or lifecycle API;
3. use an access widener for a narrowly required visibility change;
4. use a Mixin only when the requested behavior cannot be implemented through the public API.

A Mixin must pin the target Minecraft and mappings version, validate the exact target descriptor, avoid client classes in common code, and have a behavior test. Broad overwrite-style patches are not the default generation method.

## Networking and authority

Client input is never treated as trusted game state. Client-to-server packets require bounded decoding, identifier validation, permission checks, range or ownership checks, cooldown or rate limiting, and execution on the correct server thread. Rewards, inventories, balances, quests, classes, skills, parties, and world mutations remain server-owned.

## Persistence and migration

Persistent data must include a schema version. Writes must be atomic where the storage layer permits it, and upgrades must either migrate known versions or fail with a recorded fallback instead of silently discarding data. Persistence features require save, stop, restart, reload, and multiplayer-authority tests.

## Mod-owned world generation

Worldgen is selected only when explicit mod functionality needs it. Valid outputs include:

- configured and placed features;
- biome additions or modifications;
- structures, structure sets, template pools, and processors;
- dimensions and dimension types;
- biome and placement tags;
- registration or bootstrap code;
- tests in a newly generated disposable world.

Invalid product outputs include:

- complete playable world save folders;
- world ZIP downloads;
- `.schem` or `.litematic` map artifacts;
- external Builder `BuildSpec` or NPZ block-delta handoffs.

## Existing project modification

Only source projects the user owns or may modify are patchable. A JAR can be inspected but is not represented as editable source. The patch flow indexes the project, records current hashes, applies exact guarded changes, runs diagnostics and regression checks, and rolls back if the approved transaction fails.

## Release decision

A release is ready only when all gates selected by the request pass. Unrequested external gates are not invented, but requested runtime, visual, multiplayer, persistence, performance, accessibility, AI, or voice behavior must have corresponding evidence. Missing executables, credentials, consent, licenses, compatibility evidence, or runtime receipts remain unresolved rather than being reported as success.
