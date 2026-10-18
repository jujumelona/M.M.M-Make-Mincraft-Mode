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

- **Full** — write and save the plan, then prepare and build the implementation.
- **Plan** — write and save the plan, then stop without preparing production contracts.
- **Revise** — upload an existing source/release ZIP, write a revision plan, then patch the project.
- **Execute** — load a saved plan and build its implementation.

The Planner retrieves local task templates, decomposes the original request into
distinct features and source-search queries, and saves that feature ledger alongside
the supporting design in an `mmm/authored-plan-v2` document. Model responses use small,
bounded JSON pages. The ledger preserves the whole request even when one worksheet
concern has only four records. Irrelevant concerns are skipped, and state authoring
receives the preceding behavior design. Legacy plans remain loadable.
Dense source pages continue after 32 features without expanding each model response.

Runtime worksheet records have one persisted execution owner before Typed IR is
authored. Custom gameplay hooks belong to their feature implementation; retained
UI transactions keep their exact action identities. This prevents the limited
Typed dispatcher and generated feature callbacks from both applying one effect.
After dependencies are generated, the coder receives their actual Java declarations
and bounded source excerpts, alongside assigned state and atomic mutation records.
Independent ownership calls use the measured native model slots and retain input
order. Execution reads pure record/source views without importing planning models.

Production generates feature code directly from the selected Minecraft/Fabric
API and the current project. The planning and execution path does not retrieve
external mod candidates, transplant donor source or install source-reuse proofs.
Compiler diagnostics retain the failed Java source for repair. Real build,
GameTest and runtime evidence remain independent acceptance gates.

Successful model calls emit a compact completion line; failures and periodic progress
remain visible. Set `MMM_LLAMA_LIFECYCLE_VERBOSE=1` for detailed transport lifecycle logs.

The notebook does not require an engine ZIP. The setup cell clones or fast-forwards the official GitHub `main`, verifies that the checkout exactly matches `origin/main`, and prints the commit actually used. If a Colab tab predates an engine/setup change, reopen the notebook and restart the runtime before running the cells again.

The checked-in notebook currently exposes these model profiles:

- `MiMo-V2.6-9B_6GB`
- `Qwen3.8-27B_18GB`
- `fast_test`

The optional local CUDA llama-server cell uses the same resolved planner configuration. Google Drive storage is enabled by default, and resumable runs reuse completed work instead of rebuilding it unnecessarily.

`PERFORMANCE_MODE` defaults to `Auto`. On a cold or cache-invalid run, the engine measures the live CPU, system RAM, and GPU budget, probes one, two, and—when feasible—four shared llama-server slots, and keeps the best deterministic candidate that clears the minimum-gain gate; an exactly matching cached decision is reused. `Latency` favors one request at a time; `Throughput` favors concurrent independent planning and implementation pages. All slots share one resident model, and each batch is merged and validated deterministically. Large projects remain paginated instead of using an unbounded single model response.

## Local Python

Production compiles declaration-only concerns such as
`persistence.stored_state` directly from canonical records. New authored plans
persist a Typed PlanIR when preparing production. Supported Typed PlanIR operations are
validated and lowered to `AuthoredProgram.java` by a deterministic host compiler;
state operations use the canonical structured-state compiler and
`AuthoredStateModel.java`. These production paths do not invoke the coder, Java
structure assembly, paging, completion decisions, or model repair, and the
orchestrator does not load the model router for Typed PlanIR generation.

Unsupported Typed IR operations are rejected at the planning boundary and checked
again at production. Original gameplay callbacks have their own persisted source
implementation owners and receive actual prerequisite APIs. Legacy saved plans without Typed PlanIR retain the
compatibility path so old artifacts remain readable.

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

### Visual resources: no FLUX or standalone texture packs

M.M.M no longer creates standalone texture/resource-pack ZIPs or generates
AI textures using FLUX, diffusers, rembg/BiRefNet or other image backends.
New complete proposals do not schedule image asset jobs, and old proposals with
AI texture assets are rejected until migrated. Colab's normal installation no
longer includes the `image` dependency extra or its image-model preflight.

Fabric mod resource **documents** remain supported: item/block models, blockstates,
recipes, loot tables and their references are required for gameplay. The
extended-content generator uses existing vanilla Minecraft textures as a **generic
temporary appearance** where custom image files were previously generated. It
must not claim original artwork or custom visual quality. Existing valid
mod-owned PNG resources are preserved. Missing entity UV textures or GUI artwork
that cannot be represented by vanilla references must be reported as unsupported
rather than silently invented.

The retired image code remains in some legacy modules for compatibility and
test migration, but it is not an active production backend. The
`generate-textures` skill is retired in both canonical and packaged plugin
skills, and the `asset-generation` capability is not buildable.

### Plugin versus mod dependencies

`capability_plugins.py` is an internal M.M.M capability manifest, not a
generator of separately distributed Bukkit/Paper plugins. The deliverable is
a **Fabric mod JAR** built for the exact version and loader the user selects.
Any required compatible existing mods are separate dependencies only when
verified by the host; missing behavior is implemented in the Fabric mod itself.
A standalone Paper/Bukkit plugin would require a different supported target and
is not produced by this pipeline.

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

On the actual MiMo/Colab runtime, evaluate positive, negative, partial, synonym and
source-instruction cases using the configured model (this command loads that model):

```bash
python -m minecraft_mod_ai.research_semantic_eval --profile MiMo-V2.6-9B_6GB
```

The evaluator rejects mock adapters and writes `.mmm/research-semantic-eval.json`.
Unit tests and bounded-input tests do not establish real-model accuracy or complete
planner/Minecraft runtime readiness; those require the real evaluation and full run.
