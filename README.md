# M.M.M Make Mincraft Mode

[한국어](README_KO.md)

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

M.M.M turns a natural-language request into a Minecraft Java on the host-selected executable target mod plan, lets the user revise and approve that plan, then creates or patches the mod project. It does **not** produce standalone maps, world saves, world ZIPs, schematics, Litematica files, or external Builder block-delta jobs.

World generation is an ordinary mod capability. Structures, biomes, dimensions, ores, configured features, and placed features are generated only when the requested mod needs them, and remain inside the Fabric source project and data resources.

## Mod development methods

Every request starts from the same production baseline and adds only the capabilities the mod requires:

- lock Minecraft, Fabric Loader, Fabric API, Yarn, Loom, Gradle, and Java versions;
- separate common/server-safe code from client rendering, screens, keybinds, and model registration;
- generate typed registries and data resources for items, blocks, recipes, loot, tags, models, blockstates, and language files;
- prefer Fabric events, using Mixins or access wideners only when the public API cannot implement the requested behavior;
- keep networking and state changes server-authoritative with validation, permissions, rate limits, persistent-state schemas, migrations, and restart tests;
- add configuration, entities, rendering, GeckoLib animation, audio, commands, multiplayer systems, or mod-owned worldgen only when requested;
- verify source with static checks, Eclipse JDT LS, Gradle, GameTest, dedicated-server loading, JAR inspection, runtime checks, SBOM, and provenance before release.

The full method matrix is in [docs/MOD_DEVELOPMENT_METHODS.md](docs/MOD_DEVELOPMENT_METHODS.md).

## Google Colab

The repository has one canonical notebook: [`M.M.M_Make_Mincraft_Mode_Colab.ipynb`](M.M.M_Make_Mincraft_Mode_Colab.ipynb).

Choose `RUN_MODE` in the first cell:

- **Full** — create a new plan, revise it in conversation until you explicitly approve it, then build.
- **Plan** — create and revise a plan, approve it, and save the plan without building.
- **Revise** — upload exactly one existing source/release ZIP that you own or may modify, create a revision plan, approve it, then patch the project.
- **Execute** — load a saved plan, review or revise the full plan, and build only after explicit approval.

The notebook does not require an engine ZIP. The setup cell clones or fast-forwards the official GitHub `main`, verifies that the checkout exactly matches `origin/main`, and prints the commit actually used. If a Colab tab predates an engine/setup change, reopen the notebook and restart the runtime before running the cells again.

The checked-in notebook currently exposes these model profiles:

- `Qwen3.5-9B_6GB`
- `Qwen3.6-35B_23GB`
- `Qwen3.6-27B_18GB`
- `Qwen3.6-27B_14GB`
- `mini_mod`
- `fast_test`

The optional local CUDA llama-server cell uses the same resolved planner configuration. Google Drive storage is enabled by default, and resumable runs reuse completed work instead of rebuilding it unnecessarily.

`PERFORMANCE_MODE` defaults to `Auto`. On a cold or cache-invalid run, the engine measures the live CPU, system RAM, and GPU budget, probes one, two, and—when feasible—four shared llama-server slots, and keeps the best deterministic candidate that clears the minimum-gain gate; an exactly matching cached decision is reused. `Latency` favors one request at a time; `Throughput` favors concurrent independent planning and implementation pages. All slots share one resident model, and each batch is merged and validated deterministically. Large projects remain paginated instead of using an unbounded single model response.

## Local Python

```python
from minecraft_mod_ai import CompleteModAISession, resolve_mod_development_methods

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

The optional plugin bundle is in [`plugins/mmm-minecraft-mod-ai`](plugins/mmm-minecraft-mod-ai). It packages the conversational entry skill and stage-specific M.M.M MCP configuration. Its `mmm-generation` server exposes the mod-only generation surface and does not expose standalone map or external Builder tools. Colab and Python usage do not require the plugin.

## Scale

There is no fixed product-wide cap on feature count, module count, or total mod scope. Large plans are split into bounded, resumable work instead of one oversized prompt or file. Minecraft and Java formats, GPU/RAM, disk, model runtimes, and session quotas remain real execution limits and are treated as per-task resource boundaries.

## License

[MIT](LICENSE)
