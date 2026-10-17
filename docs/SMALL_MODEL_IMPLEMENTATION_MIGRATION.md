# M.M.M small-model-first implementation engine: executable migration plan

Status: **IN PROGRESS** (not production-complete). Started 2026-10-18 (KST).
Owner: repository implementation, not an open-ended research task.
Baseline commit: `054b9d3e37b33011a6ec184a688f2371b117f271`.
Target: **the user's selected Minecraft version and loader**; NEVER hardcode 26.2.
Source discovery: **CurseForge only**. GitHub may be used to maintain THIS codebase, NEVER as a runtime mod-candidate search provider.

## User-visible acceptance contract

Given a request such as *modular spaceships, trading, mining and hostile alien entities*, the pipeline must:
1. Extract a **bounded list of distinct gameplay capabilities**, preserving each requested behavior and acceptance criterion, deduplicating synonyms; one feature receives one owning implementation unit.
2. Consider only a **small CurseForge candidate shortlist per capability** (not whole catalogs). Reject incompatible version/loader, missing rights or missing evidence. Record why each candidate is adopted or rejected.
3. Choose **one** implementation route per feature:
   - `dependency`: an available, compatible distributable mod with an authorized dependency declaration and integration contract.
   - `source_reuse`: a downloadable, licensed, hash-pinned source snapshot with proven target compilation and explicit source provenance.
   - `fresh`: new implementation of only the uncovered behavior.
   Metadata alone is NEVER installed code and cannot claim reuse.
4. Generate a host-owned compile-ready **feature scaffold** (lifecycle, registries, types, ownership, imports, state and events) from the actual chosen version+loader classpath. Explicitly refuse unsupported scaffolds instead of inventing classes.
5. Ask the small model for **complete Java method/class units**, not hundreds of field/symbol/operation micro-decisions. Supply only relevant verified APIs, exact signatures and short adjacent source.
6. Run fast local syntax/type checks and **real Gradle compileJava** on source units or tight batches. Return bounded, structured compiler diagnostics to the model for at most **two local edits per feature**; on repeated failure switch route or report a hard failure with source and logs. Never claim a successful runtime behavior from AST admission alone.
7. Integrate all feature units, compile package JAR, then only appropriate GameTest/playtest/acceptance tests, and emit a report/ZIP containing both passed and failed capabilities. No false PASS on missing, stub or no-op behavior.
8. Be resilient to Colab limits: reuse model instance, project/classpath and grounded snippets; avoid unbounded agent loops, entire-bytecode inventory ranking, repeated Gradle distribution downloads, and redundant searches. Persist hash-bound feature checkpoints so a restart resumes completed code and diagnostics.

## Existing root causes (verified in code and 2026-10-17 production log)

- `construction_author.py`: the default stateful author requests one typed operation at a time. `construction_operation_binding.py` then asks the model to select API and synthesize unconstructed inputs.
- `construction_api_admission.py` currently uses action/name stems; `Registry.register` vs `RecipeType.register` is not disambiguated by semantic nouns, registry ownership and input construction.
- `construction_direct_writer.py` already exists but rejects every stateful behavior, runtime ownership and retained typed plan: the complex features get forced through micro-node authoring.
- `typed_plan_authoring.py` authoring loop makes one model call per semantic AST decision; `single_record_template.py` recently batches up to three primitive record fields, but this is not whole-feature planning.
- `grounded_source_reuse.py` forces every capability to `fresh` and returns zero verified adoption: catalog findings are not implementation assets.
- Observed 2026-10-17 log at `054b9d3`: 583 inference requests, ~440 planning / 143 coding, accepted operations without proof of behavior, zero verified donors and no completed JAR.
- Pre-existing main CI at baseline: Cross-stage ABI passed, while CI, Partial CI, Production Kill Gates and Full Debug Gate failed. Many tests still assert former GitHub/Modrinth runtime discovery; those tests must be updated to CurseForge-only intended policy rather than restoring forbidden functionality.

## Ordered engineering tasks (never mark DONE without executable evidence)

| ID | Deliverable | Stage | Completion evidence |
|---|---|---|---|
| M0 | This authoritative migration checklist, baseline and acceptance fixtures | DOCUMENTED | committed doc with status and invariants |
| M1 | `FeatureWorkOrder`: bounded, stable per-feature work item with required effects, state/effects, target, dependencies, source route and a compile acceptance contract | IMPLEMENTED, CI NOT YET VERIFIED | `feature_work_order.py`, `construction_generation.py`, `tests/test_feature_work_order.py` |
| M2 | Verified route selector: CurseForge compatibility + licensing/permissions + byte/hash proof, `dependency`/`source_reuse`/`fresh` and explicit reasons | PARTIAL | Explicit dependency file ID / authorized metadata / local SHA1 / Fabric mod manifest + Gradle local dependency lane; automatic selection, external license rights, distributable dependency, source donor proof remain unimplemented |
| M3 | Version/loader-aware scaffold factory grounded in inspected classpath for registries, lifecycle, state holders, events, server/client sides | PARTIAL | Host-verified Fabric ModInitializer/entrypoint scaffold and tests; no compile-tested gameplay/entity/item/block/side templates yet |
| M4 | Unified **feature-level** Java writer: host supplies signatures and state/event ownership, small AI writes complete methods; no default micro-node writer | PARTIAL | Whole-declaration author and checkpoint for independent canonical methods AND initialized fields; stateful and donor/port implementations still legacy; real compile E2E NOT proven |
| M5 | Bounded model context: at most relevant shortlist of verified callable APIs + nearby source; reject wrong registries and missing constructor chains BEFORE model calls | PARTIAL | Wrong-domain registrar filter plus bounded exact-type argument producer evidence; full constructor-chain constructibility checker and target compile proof remain TODO |
| M6 | Host compiler feedback: real compileJava per feature/batch, diagnostics normalized by file/line/code, local 0-2 edits, rollback/replan on persistent error | PARTIAL, INTEGRATED | One direct-owned javac-error file, up to 2 method-local corrections and rollback; production failed-build hook reruns validation/JDT/build. Latest CI and real Colab repair not yet verified |
| M7 | Replace per-field/AST planner recursion with one small semantic feature planning call; host handles cardinality, dedup, scheduling | TODO | call-count regression on multi-capability scenario and coverage equality |
| M8 | Persist code, compile outputs, source evidence and work order with hashes; resume without regenerating finished feature | PARTIAL | Exact compiler-repaired source replay bound to original source SHA and four project configuration hashes, plus unit fixtures. Full work graph resume and real interrupted run unverified |
| M9 | Integrate dependency/source/fresh results, resolve dependency versions, normalize provenance/licenses, integrate runtime behavior and build final JAR | PARTIAL | Explicit proven external Fabric dependency wired into build.gradle and fabric.mod.json, revalidated against current CurseForge metadata. Source donor materialization, portable build and combined runtime acceptance still unverified |
| M10 | Acceptance suite: playtest/GameTest, per-feature pass/fail coverage, no-op rejection, residual work audit, output ZIP/report | TODO | representative space+entity+trade scenario produces verifiable evidence |
| M11 | Reconcile all pre-existing failures and enforce CI/Partial CI/Kill Gate on every changed route; remove obsolete GitHub/Modrinth runtime-search tests | TODO | every required Actions workflow green on same SHA |
| M12 | Colab performance budget: bounded inference, no redundant catalog/API sweeps, no wasted repeat preflights, no unnecessary checks inside micro-steps | PARTIAL | Shortlisted query and API reads, whole-method writer and max two edits. No representative Colab end-to-end call-count benchmark or runtime reduction proven |

### Routing invariants

- Work orders are immutable across an execution attempt. A failed feature produces a focused replan with recorded diagnostics, not a new global catalog sweep.
- An API is usable only if it is accessible, exact-type compatible, has the correct **domain owner/registry** and all required receivers/arguments are constructible from verified source/scaffolds. Similar names are not evidence.
- A dependency is NOT source reuse. CurseForge file IDs, version compatibility, license conditions and explicit user-controlled packaging govern dependencies; code transplant needs independently inspectable authorized sources. Never silently download GitHub sources as a runtime search fallback.
- The small model's main job is implementing **real gameplay**, not signing off on schema/graph evidence. Mechanical checks are host-owned.
- A compile passes syntax/API proof but not gameplay: accepted behavior still requires appropriate integration and runtime acceptance.
- The pipeline reports `not_implemented` or `failed` for uncovered behavior rather than emitting a misleading empty successful mod.

### Implementation strategy

Do these in order: M0 → M1 → M4/M3 small stateless fixture → M6 → M4 stateful fixture → M5 → M2 → M7 → M8 → M9/M10 → M11/M12. Stage IDs can be partially complete, but the **status must state exactly which tests/CI gates passed**, not aspirational promises.

### Operational baseline / test triage

Main at `054b9d3` had failed CI, Partial CI, Production Kill Gates and Debug Gate. Partial CI includes tests for the now-removed runtime GitHub/Modrinth discovery (legacy expectations); do NOT restore those runtime search providers merely to satisfy old tests. Investigate failures per test and migrate/remove obsolete expectations with replacement CurseForge-only assertions.

### Update log

- 2026-10-18: M0: exhaustive architecture/acceptance plan committed. Implementation and verification remain staged; no assertion of production completion.

- 2026-10-18: M1 implemented: immutable `FeatureWorkOrder` carrying every behavior, target coordinates, state and acceptance contract. Work order is injected at canonical feature-entrypoint composition and reflected in direct-writer prompts; tests added (CI not yet green).
- 2026-10-18: M4 **partial**: added `construction_member_direct_author.py`; independent method declarations use one complete Java method call with one optional syntax correction and hash-bound checkpoint; legacy operation IR remains for stateful, retained typed effects and donor/port obligations. `construction_generation.py` routes these eligible methods directly. Method presence is not compiler/gameplay proof. Real Gradle and restart fixtures remain required.
- 2026-10-18: M5 **partial**: root registry API admission checks owner domain (entity, item, block, recipe, etc.) as well as method stem; excludes `RecipeType.register`, `RuleTestType.register`, `EntityRendererRegistry.register` when registering entity types. Full argument constructibility contract still open.

- 2026-10-18: M4 partial expansion: host-fixed initialized Java fields now use the same bounded complete-declaration writer as methods, with a regression fixture. Canonical non-entrypoint leaves such as `minecraft/entity/registry` now receive a feature work order. This reduces model micro-operations but does not yet establish actual compilation or runtime correctness.
- 2026-10-18: Status note: Cross-stage Contract ABI passed for intermediate SHAs; the baseline CI/Partial CI/Kill Gates were already failing before this migration, and latest-sha CI is not yet green. Preserve the full failure inventory and fix actual code/tests, not merely the gate declaration.

- 2026-10-18: M5 partial expansion: `construction_direct_writer._evidence` now exposes at most six exact inspected input types and at most two real constructor/static provider signatures per type from the target bytecode index. Semantically unrelated constants (e.g. a vanilla entity type other than the requested alien) are not offered merely because their Java return type matches. Added `tests/test_construction_direct_input_producers.py`. This is not a proof that all required argument chains are constructible; M5 remains PARTIAL.

- 2026-10-18: M6 PARTIAL INTEGRATED: Added `feature_compile_repair.py`, `focused_method_compile_repair.py`, `production_compiler_repair.py`. The post-build hook admits only real owned javac diagnostics for direct-authored source and only one Java file, permits 0-2 targeted changes, requires full source validation / JDT / Gradle build to run again, and never claims GameTest was fixed by a compile. Newly added unit tests are NOT a substitute for a passing Colab production run.
- 2026-10-18: M8 PARTIAL: `compiled_feature_patch.py` persists repaired whole-source bytes bound to original SHA and selected project's build.gradle, settings.gradle, gradle.properties and fabric.mod.json SHA. Direct and typed renderers replay only if all identities match. Cross-restart replay and version invalidation unit fixtures added. Gameplay remains unverified.
- 2026-10-18: M2/M9 PARTIAL: `curseforge_dependency_adoption.py`, `curseforge_dependency_project.py`, and exact-file provider re-verification via `authored_reuse_bridge.py`. Opt-in local JAR dependency ONLY. User license acknowledgement is not automatic license verification; JAR not redistributed or guaranteed available in a portable release. Full candidate selection and source transplant remain separate tasks.
- 2026-10-18: M3 PARTIAL: `verified_feature_scaffold.py` generates an explicitly UNIMPLEMENTED host-owned Fabric initialization point only when matching inspected classpath symbols are present. No invented registration API or false feature PASS.
- 2026-10-18: Evidence rule: The 2026-10-18 current SHA has not demonstrated a completed Minecraft mod under the small model. Treat all published implementation stages as provisional until the required CI AND real construction compile/GameTest receipts pass.
