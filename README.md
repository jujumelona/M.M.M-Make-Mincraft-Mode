# M.M.M Make Mincraft Mode

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


## Integrity registration and execution evidence

`minecraft_mod_ai.integrity_bootstrap.bootstrap_integrity()` registers all 346
canonical leaf callables, their 692 input/output schemas, executable validators,
and the actual YAML templates as one authority. Missing files, conflicting
registrations and changed source fail closed. There are no `LENIENT` identities.
Registration is separate from production admission: packaged leaf bindings remain
`not_reviewed` until a matching candidate has real compile and GameTest evidence.
A generic registered callable does not prove all 346 Minecraft behaviors.

The generator consumes an explicit HOST render mold and bounded, schema-constrained
slots. A candidate is generated in isolation, tested with Gradle, then admitted
through `integrity_catalog.admit_evidenced_leaves`. Production dispatch checks the
same input contract, generated content, implementation, validators, schemas,
classpath, side and content-addressed evidence before materialization or reuse.

```bash
python tools/integrity_pipeline.py registration --minecraft 1.20.1 --output registration.json
python tools/integrity_pipeline.py inspect --minecraft 1.20.1 --loader fabric --namespace named --java 17 --jar minecraft.jar --mappings mappings.tiny --source-namespace official --output api.json
python tools/integrity_pipeline.py candidate --help
python tools/verify_integrity_minecraft.py --output .mmm/integrity-validation
```

The inspection command parses JVM class files and Tiny v2 names/descriptors;
API epochs are derived from declarations and separated by loader. Java validation
uses the installed JDK compiler for syntax and classpath symbol attribution.
Use a JDK matching the requested target. The reproducible evidence probe requires
JDK 17 and network access, and runs Fabric 1.20.1 item registration through Gradle
and a real GameTest. Its result is scoped to that item fixture and target.
Evidence retains raw source, actual Gradle classpath, compiled classes, logs and
XML test reports under the selected output directory. CI uploads these records.
Missing evidence or zero executed tests never authorizes production.

### Resource image contracts

The asset producer stores a `mmm/resource-asset-generation-plan-v3` manifest and
re-resolves it against HOST facts, content owners, visual semantics and the image
registry before generation. Older image plans must be rebuilt and approved.
`VisualSpec` accepts only role, silhouette, materials, motifs and palette;
`asset/item_sprite` contributes framing while the compiler consumes registry
settings. Preferred and fallback generation resolutions belong to
`config/model_registry.yaml`, independently of final Minecraft geometry.

HOST snapshots may carry `resource_asset_bindings`, keyed by exact resource
subject. These bindings provide `render_kind`, `geometry: {width, height}`,
optional `animation: {frame_count, frametime}`, and explicit crop cardinality.
Entity bindings require `uv_schema: {id, lora_compatible, regions}` with named
`box: [x, y, width, height]` regions. GUI bindings require `gui.generated_regions`
(background/frame/decorative_border/ornament) and `gui.protected_regions` with
deterministic RGBA pixels. GUI controls, text and exact icons remain code/template
responsibilities; the image compositor restores protected pixels after decoration.

Entity and GUI `model_binding` must contain a consumer `path`, `sha256` and exact
`texture_reference` such as `demo:textures/entity/guardian.png`. Production checks
the existing consumer bytes and reference literal before inference. This is a
resource binding check, not proof that a renderer or screen executes correctly.
Missing layouts or consumers fail closed; no generic entity/GUI atlas is invented.
Animated UV/GUI composition is currently rejected.

PNG validation checks native geometry, palette, alpha, region coverage, tile edges,
protected pixels, animation sidecars and the complete generated model reference
set. Candidate sources are composed by region or animation frame using nearest-grid
conversion. A registry fallback is attempted only after a memory allocation failure.
Static assets reject stale animation sidecars. Pixel fixtures and stub image backends
exercise these contracts in `tests/test_resource_contract_pipeline.py`; real model
quality, Gradle, GameTest and Minecraft runtime validation remain separate gates.

### Research evidence and small-model validation

Catalog queries retain authored topic anchors; the full original task and sibling
requirements stay in the query context, rather than becoming long keyword searches
or thousands of unrelated singleton hits. Retrieved bodies are deduplicated by source
identity and content hash without a candidate top-N cutoff. Lexical matches rank review
work and never authorize research completion.

Each acceptance obligation is assessed against bounded source windows using the active
model input budget. A separate entailment call checks a proposed exact quote; the host
validates source, body hash, window, requirement and obligation bindings. These remain
model-based semantic observations, not infallible proofs. Completed observations are
stored under `.mmm/semantic-research-cache` and resumed only for matching policy,
model configuration and source bindings. Corrective retrieval requires new topic-bound
queries after semantic evidence fails. The detailed planner receives only admitted
obligation proofs; full bodies and unsuccessful candidates remain in the research state.

The design draws on [CRAG's retrieval evaluation](https://arxiv.org/abs/2401.15884),
[ALCE's citation quality evaluation](https://arxiv.org/abs/2305.14627), and the input-position
limitations studied in [Lost in the Middle](https://arxiv.org/abs/2307.03172). It is an
adaptation for this pipeline, not a reproduction of those papers' benchmark results.

Run host contract and context-size regressions with:

```bash
python -m pytest -q tests/test_planning_semantic_research.py tests/test_planning_candidate_evidence.py tests/test_planning_mod_discovery.py
```

On the actual Qwen/Colab runtime, evaluate positive, negative, partial, synonym and
source-instruction cases using the configured model (this command loads that model):

```bash
python -m minecraft_mod_ai.research_semantic_eval --profile Qwen3.5-9B_6GB
```

The evaluator rejects mock adapters and writes `.mmm/research-semantic-eval.json`.
Unit tests and bounded-input tests do not establish real-model accuracy or complete
planner/Minecraft runtime readiness; those require the real evaluation and full run.
