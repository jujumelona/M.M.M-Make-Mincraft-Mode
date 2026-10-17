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
| M1 | `FeatureWorkOrder`: bounded, stable per-feature work item with required effects, state/effects, target, dependencies, source route and a compile acceptance contract | TODO | isolated unit tests, version changes invalidate cache |
| M2 | Verified route selector: CurseForge compatibility + licensing/permissions + byte/hash proof, `dependency`/`source_reuse`/`fresh` and explicit reasons | TODO | fixture for each mode, no false donor proof; no forbidden runtime provider |
| M3 | Version/loader-aware scaffold factory grounded in inspected classpath for registries, lifecycle, state holders, events, server/client sides | TODO | compile real minimal entity/item/block/event fixture against chosen target |
| M4 | Unified **feature-level** Java writer: host supplies signatures and state/event ownership, small AI writes complete methods; no default micro-node writer | TODO | stateful and stateless real feature fixtures; old IR reserved as explicit legacy option |
| M5 | Bounded model context: at most relevant shortlist of verified callable APIs + nearby source; reject wrong registries and missing constructor chains BEFORE model calls | TODO | `Registry.register` accepted for EntityType only when object construction feasible; `RecipeType.register` rejected |
| M6 | Host compiler feedback: real compileJava per feature/batch, diagnostics normalized by file/line/code, local 0-2 edits, rollback/replan on persistent error | TODO | controlled failing Java fixture repaired then compiled, 3rd identical attempt impossible |
| M7 | Replace per-field/AST planner recursion with one small semantic feature planning call; host handles cardinality, dedup, scheduling | TODO | call-count regression on multi-capability scenario and coverage equality |
| M8 | Persist code, compile outputs, source evidence and work order with hashes; resume without regenerating finished feature | TODO | restart/resume and invalidation tests |
| M9 | Integrate dependency/source/fresh results, resolve dependency versions, normalize provenance/licenses, integrate runtime behavior and build final JAR | TODO | real Gradle JAR + generated resources + runtime launch smoke |
| M10 | Acceptance suite: playtest/GameTest, per-feature pass/fail coverage, no-op rejection, residual work audit, output ZIP/report | TODO | representative space+entity+trade scenario produces verifiable evidence |
| M11 | Reconcile all pre-existing failures and enforce CI/Partial CI/Kill Gate on every changed route; remove obsolete GitHub/Modrinth runtime-search tests | TODO | every required Actions workflow green on same SHA |
| M12 | Colab performance budget: bounded inference, no redundant catalog/API sweeps, no wasted repeat preflights, no unnecessary checks inside micro-steps | TODO | logged call count, elapsed time and representative end-to-end completion |

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
