# M.M.M Make Minecraft Mode

[한국어](README_KO.md)

M.M.M is a fail-closed Minecraft Java Fabric 1.20.1 production system. Its default workflow takes one approved multimodal brief through complete planning, source generation or modification, repair, build, GameTest, JAR inspection, disposable runtime playtesting, visual review and release packaging.

## Default complete workflow

```bash
python -m pip install -e '.[dev,ui,local-model,rag,image,speech,training,production-audio]'
mmm plan "Create the complete mod" --profile t4_local --save complete-proposal.json
mmm validate-proposal complete-proposal.json
mmm execute complete-proposal.json --approve <sha256> --output mmm-output --server-launcher <fabric-server-launch.jar> --accept-eula --screenshot <runtime.png>
```

The complete proposal can describe items, blocks, food, weapons, tools, armor, crops, machines, effects, enchantments, commands, recipes, advancements, loot, animated entities, quests, classes, skills, economy, shops, GUI/networking, parties/guilds, structures, audio and bounded custom Java modules. Dependencies and acceptance tests are part of the immutable approval hash.

## Implemented production stages

- Qwen role-routed planning/coding/RAG, FLUX assets and Whisper input
- exact-hash transactional source patching and finite diagnostic repair
- real Fabric registrations for extended content and gameplay systems
- persistent quest, class, economy and party data
- GeckoLib entity, renderer, attributes, goals and animation bindings
- gzipped structure NBT, Jigsaw pools, structure sets and world resources
- OGG synthesis/import, `sounds.json`, subtitles and `SoundEvent` registration
- JDT LS, Gradle, GameTest and independent JAR validation
- disposable Minecraft 1.20.1 server/client, Mineflayer and visual review adapters
- source/JAR distribution bundles and optional reviewed Modrinth/CurseForge upload
- 21 executable Skills and a real FastMCP server

## Verification boundary

Ordinary GitHub CI verifies the CPU contracts. T4 inference, Minecraft, Blockbench and publishing require the manual self-hosted `Production integration` workflow. Missing executables, EULA acceptance, endpoints, credentials or evidence fail closed.

See [docs/PRODUCTION_STACK.md](docs/PRODUCTION_STACK.md).
