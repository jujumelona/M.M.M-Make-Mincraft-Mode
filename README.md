# M.M.M Make Mincraft Mode

[한국어](README_KO.md)

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

M.M.M turns a natural-language request into a Minecraft Java 1.20.1 Fabric mod plan, lets you revise that plan in conversation, and then creates or patches the mod project. It does not create standalone maps, world saves, world ZIPs, schematics, Litematica files, or external Builder block-delta jobs.

World generation is treated as an ordinary mod capability, not as a separate map product. Structures, biomes, dimensions, ores, and configured or placed features are generated only when the requested mod actually needs them, and the outputs stay inside the Fabric source project and its data resources.

For each request, the central agent reclassifies the required systems, code, libraries, images, 3D, animation, audio, licensing, and test evidence, then searches exact-version RAG and compatible open-source or media candidates. It does not install or copy a search result before its origin license and immutable file hash are verified.

Each request is kept as a traceable requirement connected to implementation and an observable test. Large jobs continue in resumable slices until every relevant quality check has fresh evidence, or the result names the real external input or runtime that is still missing. OpenAlex and Crossref searches can supply current research candidates, but papers and search results are never treated as proof that the mod works.

If the request needs AI or voice, M.M.M dynamically searches current runtimes and model catalogs instead of fixing one product in the engine. It separates inference, ASR, VAD, TTS, translation, transport, and optional voice adaptation, then checks the exact Minecraft, Fabric, and Java boundary, immutable model revision, code, model, and data licenses, hardware and latency measurements, privacy, and fallback. The approved plan adds one bounded, token-authenticated asynchronous localhost bridge for the requested executable capabilities; it does not bundle a model before its gates pass or let model output mutate game state directly. Voice adaptation stays disabled unless the speaker is authorized and explicit consent, provenance, revocation, and deletion all pass.

## Mod development methods

Every request starts with the same production baseline and adds only the methods required by that mod:

- lock Minecraft, Fabric Loader, Fabric API, Yarn, Loom, Gradle, and Java versions;
- split common/server-safe code from client rendering, screens, keybinds, and model registration;
- generate typed registries and data resources for items, blocks, recipes, loot, tags, models, blockstates, and language files;
- prefer Fabric events, using Mixins or access wideners only when the public API cannot implement the requested behavior;
- keep networking and state changes server-authoritative, with packet validation, permissions, rate limits, persistent-state schemas, migrations, and restart tests;
- add configuration, entities, rendering, GeckoLib animation, audio, commands, multiplayer systems, or mod-owned worldgen only when requested;
- verify source with static checks, Eclipse JDT LS, Gradle, GameTest, dedicated-server loading, JAR inspection, runtime checks, SBOM, and provenance before release.

The full method matrix is in [docs/MOD_DEVELOPMENT_METHODS.md](docs/MOD_DEVELOPMENT_METHODS.md).

## Google Colab

1. Click the Colab badge and choose a GPU runtime.
2. Enter `PROMPT`, then run the cells in order.
3. Use the optional revision cell to change the plan in plain language.
4. Run **Build this plan**, then download the result.

The notebook does not require an engine ZIP. On every setup run it clones or fast-forwards GitHub `main` and prints the exact commit it used.

If a Colab tab was already open before an engine/setup update, reopen the notebook from the badge and restart the runtime once before running all cells. The pulled setup then verifies the Qwen CUDA kernels before downloading model weights and refuses the memory-heavy fallback.

- New mod: leave `PATCH_EXISTING=False`. Nothing is uploaded.
- Modify an existing mod: set `PATCH_EXISTING=True`, then upload one source/release ZIP that you own or may modify. It must contain source code and a Gradle project.
- A JAR by itself can be inspected, but it is not presented as editable source.

Google Drive storage is enabled by default, so rerunning the same `RUN_NAME` resumes completed work instead of failing because the folder already exists.

## Local or remote models

`MODEL_PROFILE="t4_local"` is the safe Colab T4 default and runs the 4B planner in 4-bit. Choose `t4_quality` only when you want the optional 9B planner and the runtime passes its stricter VRAM preflight. Set it to `remote_quality` and fill in the HTTPS API address and model fields to use OpenAI-compatible remote endpoints; the notebook asks for the API key without saving it in the notebook.

For local Python:

```python
from minecraft_mod_ai import (
    CompleteModAISession,
    resolve_mod_development_methods,
)

methods = resolve_mod_development_methods(
    "Make a seasonal farming and cooking mod."
)
print(methods["method_ids"])

session = CompleteModAISession(output_root="mmm-output")
plan = session.plan("Make a seasonal farming and cooking mod.")
print(plan.message)
plan = session.revise("Remove combat and add a winter greenhouse.")
result = session.build(plan, source_only=True)
print(result.release_zip)
```

## Codex plugin

The optional plugin bundle is in
[`plugins/mmm-minecraft-mod-ai`](plugins/mmm-minecraft-mod-ai). It packages the conversational entry skill and the stage-specific M.M.M MCP configuration. Its `mmm-generation` server exposes the mod-only generation surface and does not expose standalone map or external Builder tools. Colab and Python usage do not require installing this plugin.

## Scale

There is no fixed product-wide cap on feature count, module count, or total mod scope. Large plans become paged modules, bounded artifact shards, and resumable checkpointed work instead of one oversized prompt or file.

This does not mean infinite hardware. Minecraft and Java formats, GPU and RAM, disk space, model APIs, and Colab session quotas still impose real limits. M.M.M keeps those as per-task safety and resource boundaries and continues the project through additional shards and sessions.

## License

[MIT](LICENSE)
