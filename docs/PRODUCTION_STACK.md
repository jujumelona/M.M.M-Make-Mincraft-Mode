# Complete Minecraft production stack

The default path is `CompleteGameDesignPlanner -> immutable CompleteProposal -> CompleteProductionOrchestrator`.
The old item/block slice remains only as an explicit compatibility command.

## One approved execution graph

1. Parse a multimodal brief and inspect an optional existing source ZIP.
2. Produce a dependency-checked module graph covering content, entities, systems, assets, audio, custom Java and explicitly requested native Minecraft modules.
3. Freeze the complete proposal and all external-input hashes.
4. Generate or safely patch a the approved Fabric target project.
5. Bind extended content, persistent quests/classes/economy/GUI/party systems, GeckoLib entities, requested native Minecraft modules, textures and SoundEvents.
6. Run deterministic validation and JDT LS.
7. Run Gradle and GameTest, applying at most three hash-guarded minimal repair transactions from new diagnostics.
8. Independently inspect the JAR.
9. Launch a disposable Fabric server/client, Mineflayer tasks and visual review when the reviewed external environment is connected.
10. Package source, validated JAR, manifests and optional Modrinth/CurseForge upload receipts.

## Safety and correctness

- All writes remain under the approved workspace.
- Existing files require their exact SHA-256 before replacement or editing.
- A multi-file patch validates completely before the first write and rolls back on failure.
- Retrieved text and model output never grant authorization.
- No silent model substitution, loader mixing or success simulation is allowed.
- A complete binary run fails closed when JDT, Gradle, GameTest, JAR, runtime, Blockbench, Mineflayer or visual evidence required by the proposal is unavailable.

## Environment boundary

CPU CI proves Python contracts, generators, parsers, notebooks and MCP behavior. Actual T4 weights, Java dependencies, Minecraft, Blockbench and publishing credentials require the manual `Production integration` workflow on reviewed self-hosted runners. Their absence is reported as a missing gate, never as a pass.
