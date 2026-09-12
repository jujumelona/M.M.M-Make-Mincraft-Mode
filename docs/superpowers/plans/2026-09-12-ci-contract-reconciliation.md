# CI Contract Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the current `main` runtime/contracts with the 55 failing broad-shard tests without weakening fail-closed evidence, resource-safety, SSOT, or deterministic execution guarantees.

**Architecture:** Treat each failure as either a production contract/composition defect or a stale test fixture. Preserve current authoritative runtime owners (resolved version context, registry-backed resource policy, canonical generation semantics, host-owned template cardinality) and update tests only when they assert superseded APIs or layouts. Production changes must be covered by the regression that currently fails.

**Tech Stack:** Python 3.11, pytest, GitHub Actions, Minecraft/Fabric deterministic generation contracts.

**Spec:** Current `main` runtime contracts and CI run `34685043351` at baseline commit `1273cec39ec738d80db59f3b3f5dc475b557fdac`.

## Global Constraints

- Work only on `main`; do not create a branch.
- Keep fail-closed leaf/version evidence requirements intact.
- Keep registry/resource ceilings authoritative; do not restore retired fixed backend constants.
- Keep host-owned cardinality and deterministic template orchestration intact.
- Do not restore deleted duplicate template/workflow authorities.
- A completion claim requires the relevant dedicated jobs and all three broad pytest shards to pass on one current HEAD.

---

### Task 1: Reconcile artifact-expansion fixtures with reviewed leaf evidence

**Files:**
- Modify: `tests/test_complete_orchestrator_artifact_jobs.py`
- Modify if required by a real defect: `minecraft_mod_ai/artifact_expansion.py`
- Reference: `minecraft_mod_ai/resolved_version_context.py`

**Interfaces:**
- Consumes: `ResolvedVersionContext.require_leaf_binding()` and canonical artifact leaf IDs.
- Produces: tests that supply reviewed/executable leaf evidence instead of bypassing `UNSUPPORTED_LEAF`.

- [ ] Add/use a test helper that derives the normal platform context and marks only the artifact leaves exercised by the fixture as reviewed/executable using the existing evidence API.
- [ ] Keep the existing failure behavior for an unreviewed leaf unchanged.
- [ ] Update artifact job IDs only when current `artifact_expansion.py` proves the canonical suffix changed.
- [ ] Run the three failing tests from `tests/test_complete_orchestrator_artifact_jobs.py` and verify PASS.
- [ ] Commit the fixture reconciliation on `main`.

### Task 2: Restore runtime parallelism test isolation and serial-stage admission

**Files:**
- Modify: `tests/test_llama_server_autotune.py` and/or the runtime installer that owns `_install_dynamic_runtime_parallelism` if composition is defective.
- Modify: `minecraft_mod_ai/scheduler_parallel_safety_contract.py` only if its installed claim wrapper violates same-stage serial admission.
- Test: `tests/test_scheduler_parallel_safety_contract.py`

**Interfaces:**
- Consumes: registry/resource-ceiling runtime parallelism and scheduler claim API.
- Produces: deterministic unit isolation for VRAM policy and an installed scheduler that cannot claim a second node in a serial CPU stage while one is running.

- [ ] Reproduce the VRAM test with the installed runtime wrapper and identify the authoritative callable that must be restored or isolated.
- [ ] Make the test isolate that owner rather than weakening the expected safe-width calculation.
- [ ] Reproduce scheduler second-claim behavior and trace wrapper/install order.
- [ ] If production composition bypasses serial-stage filtering, fix the installed claim path while preserving cross-stage parallelism.
- [ ] Run both failing tests and the dedicated parallel/runtime contract suites.
- [ ] Commit on `main`.

### Task 3: Reconcile canonical generation and immutable production-contract fixtures

**Files:**
- Modify as warranted: `minecraft_mod_ai/scalable_generator.py`, `minecraft_mod_ai/minecraft_generation_contract.py`, or the semantic lowering path that builds `ProductionModule`.
- Modify stale fixtures: `tests/test_complete_production.py`, `tests/test_python_api.py`, `tests/test_durable_work_graph.py`, `tests/test_mcp_complete_plan_refs.py`, `tests/test_production_contract.py`.

**Interfaces:**
- Consumes: canonical `ProductionModule` semantic fields and finalized asset catalog.
- Produces: generated item/block modules with all required authored semantics and production contracts compiled from the final asset set.

- [ ] Trace missing `display_name`, `main_color`, `attack_speed`, and `hardness` to the lowering boundary; map existing authored facts forward, never invent fallback gameplay values.
- [ ] Ensure tests compiling immutable production contracts derive/finalize semantic assets before contract compilation.
- [ ] Keep validation rejecting missing required generation semantics.
- [ ] Run all five affected test modules.
- [ ] Commit on `main`.

### Task 4: Align tests with canonical artifact/template/resource APIs

**Files:**
- Modify: `tests/test_item_vertical_slice_pipeline.py`
- Modify: `tests/test_template_runtime_ssot.py`
- Modify: `tests/test_resource_asset_clean_contract.py`
- Modify: `tests/test_resource_asset_preflight_contract.py`
- Modify: `tests/test_resource_leaf_pipeline.py`
- Reference: current template manifests, `complete_spec.AssetRequest`, resource preflight owner, and version-context APIs.

**Interfaces:**
- Consumes: canonical leaf IDs, manifest/SSOT template layout, semantic `AssetRequest`, current reference-closure signature.
- Produces: tests against the current single authorities without recreating deleted duplicate files/constants.

- [ ] Replace retired `model_basic`/legacy resource IDs with IDs emitted by current expansion.
- [ ] Rewrite template SSOT assertions around the surviving manifest authority instead of deleted `research_template_pipeline.py`, `research/workflow.yaml`, or removed capture files.
- [ ] Construct `AssetRequest` with semantic fields and call reference closure with its current required context.
- [ ] Verify the live orchestrator preflight marker/installation against the actual owner; fix production installation only if it is genuinely absent.
- [ ] Supply required `ResolvedVersionContext` to real resource-leaf materialization tests.
- [ ] Run the five affected test modules and commit on `main`.

### Task 5: Reconcile RAG, grounding, mutation-target, and proposal-deserialization contracts

**Files:**
- Modify: `tests/test_predesign_performance_contract.py`
- Modify: `tests/test_reference_source_research_strategy.py`
- Modify: `tests/test_repository_grounding.py`
- Modify: `tests/test_fresh_mutation_target_grounding.py`
- Modify: `tests/test_target_context_hardening.py`
- Modify: `tests/test_proposal_deserialization_contract.py`
- Modify: `tests/test_state_provenance_contract.py`
- Modify production only where installed-wrapper identity or fresh-target authority is actually broken.

**Interfaces:**
- Consumes: current bounded query-row policy, host-reserved mutation target authority, strict proposal schema/deserializer.
- Produces: tests exercising current policy names/signatures and production wrappers that retain their authority markers after composition.

- [ ] Replace removed worker-count internals with assertions on the public bounded execution behavior.
- [ ] Update source-strategy expectations to the current bounded-query-row policy while preserving catalog-first/fallback semantics.
- [ ] Trace repository-grounding and target-context wrapper installation; repair only lost production markers/authority, not expected wording.
- [ ] Build deserialization fixtures from complete required proposal payloads, then mutate the one target field so each test reaches the intended validation error.
- [ ] Run all affected tests and commit on `main`.

### Task 6: Reconcile native resource, prefill cache, model registry, and platform optimizer tests

**Files:**
- Modify: `tests/test_llama_prefill_calibration_cache.py`
- Modify: `tests/test_llama_server_native_resource_fit.py`
- Modify: `tests/test_native_runtime_resource_parallelism.py`
- Modify: `tests/test_runtime_hardening_regressions.py`
- Modify: `tests/test_platform_dynamic_coder_contract.py`
- Modify production only for demonstrated cache-key/generation or path-resolution defects.

**Interfaces:**
- Consumes: shared prefill cache key `(server identity, wire shape/process generation)`, registry-owned resource sizing, canonical package config resolver, explicit-version optimizer contract.
- Produces: isolated tests that do not leak global cache state or expect retired raw launcher arguments.

- [ ] Reset shared prefill cache/in-flight state per test and verify shape/process-generation key separation still causes exactly one calibration per key.
- [ ] Assert current resource-policy outputs rather than pre-safety raw `--gpu-layers`/`--ubatch-size` values.
- [ ] Stub native launcher where the test is about bookkeeping so it never invokes a real `llama-server` binary.
- [ ] Replace repository-root config assumptions with the canonical packaged config resolver.
- [ ] Verify explicit-version hint behavior against current optimizer authority; repair production only if an explicit hint incorrectly hard-pins the adapter.
- [ ] Run all affected tests and commit on `main`.

### Task 7: Reconcile design-graph, work-graph, packaging, Gradle, and worksheet atomicity tests

**Files:**
- Modify: `tests/test_content_design_graph_expansion.py`
- Modify: `tests/test_four_axes_contract.py`
- Modify: `tests/test_durable_work_graph.py`
- Modify: `tests/test_generation_lane_parallelism_contract.py`
- Modify: `tests/test_gradle_runner_hot_path_efficiency.py`
- Modify: `tests/test_plugin_package.py`
- Modify: `tests/test_project_context_pagination.py`
- Modify: `tests/test_proposal_store_integrity_efficiency.py`
- Modify: `tests/test_worksheet_atomic_chunker.py`
- Modify production only where a current invariant is actually violated.

**Interfaces:**
- Consumes: host-owned design cardinality/relation scheduling, current work-graph asset generation nodes, current packaged Skill catalog, atomic worksheet chunker.
- Produces: tests coupled to observable contracts instead of deleted source-loop text, retired constants, or old package layouts.

- [ ] Replace direct model-call count/source-text assertions with observable host-owned cardinality and relation closure assertions.
- [ ] Ensure work plans contain generation nodes for semantic assets before references are validated.
- [ ] Align generation-lane and Gradle hot-path tests with current lock/wrapper authority after confirming the intended invariant.
- [ ] Read the packaged Skill catalog via its current path/manifest and assert required skills are present.
- [ ] Parse project-context pages through their current envelope instead of assuming a retired raw JSON body.
- [ ] Reconcile proposal-store collection counts with reachable advertised sections.
- [ ] For worksheet chunking, assert every emitted chunk individually satisfies the atomicity contract; do not require unrelated fields to collapse into one chunk.
- [ ] Run all affected tests and commit on `main`.

### Task 8: Full verification on one HEAD

**Files:**
- No feature files expected; only fixes discovered by verification.

**Interfaces:**
- Consumes: all preceding commits.
- Produces: one `main` HEAD with green dedicated jobs and all broad shards.

- [ ] Refetch `main` and verify no concurrent commit invalidated the tested tree.
- [ ] Run/observe CI for the exact HEAD.
- [ ] Require `Remaining tests 1/3`, `2/3`, and `3/3` to pass.
- [ ] Require planner/fixed-template, runtime composition, MCP evidence, coder, registry/Fabric evidence, parallel safety, static/packaging, coverage, and environment gates to pass.
- [ ] If any gate fails, download its current artifact and return to the relevant task; do not claim completion.
- [ ] Report the final HEAD and workflow evidence only after all required gates are green.
