# HOST-Owned Template Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every deterministic Minecraft/Fabric target fact into one evidence-backed HOST authority, make supported-target completeness fail closed, migrate consumers to HOST-only lowering, and remove proven dead or superseded template responsibilities.

**Architecture:** `ResolvedVersionContext.host_facts` is the only authority for target-specific toolchain, mappings, dependencies, API symbols, schemas, artifact rules, and canonical-leaf bindings. Product support is enforced by audits that cross-check `SUPPORTED_MINECRAFT_VERSIONS` and `REQUIRED_CANONICAL_LEAVES`; templates keep only semantic/model-owned values, while target-specific executable lowering is selected by HOST leaf bindings. Migration proceeds leaf-family by leaf-family, starting with item registration because the 1.21.2 API boundary is already independently established.

**Tech Stack:** Python, pytest, YAML task templates, JSON HOST catalog/evidence, GitHub Actions, Fabric/Minecraft Java source generation.

**Spec:** `docs/superpowers/specs/2026-09-12-host-owned-template-authority-design.md`

## Global Constraints

- Work directly on `main`; do not create feature branches or worktrees.
- Never delete a production template or field without producer/consumer/reachability proof and a verified replacement where applicable.
- Never extrapolate target-specific Minecraft/Fabric facts across releases without evidence.
- Supported targets fail closed when any required HOST fact or canonical leaf is missing, unsupported, unreviewed, or unverified.
- Model-owned templates contain semantic/content decisions only; HOST owns target-specific compatibility, API spelling, dependencies, schemas, paths, and executable lowering choices.
- Before every write, re-read the current `main` HEAD and the target file SHA because concurrent changes may land.
- Every implementation slice is test-first and independently committed.

---

### Task 1: Lock Product Support Into HOST Coverage and Readiness

**Files:**
- Modify: `minecraft_mod_ai/host_version_catalog.py`
- Modify: `minecraft_mod_ai/product_support_matrix.py` only if contract defects are found
- Test: locate existing HOST catalog/support-matrix tests and extend the canonical test file rather than creating duplicate suites

**Interfaces:**
- Consumes: `SUPPORTED_MINECRAFT_VERSIONS`, `REQUIRED_CANONICAL_LEAVES`, `validate_support_matrix(bundles)`
- Produces: `coverage_audit()` and `production_readiness_audit()` that reject any missing/unadmitted product-promised target/leaf

- [ ] **Step 1: Write failing tests** proving a missing supported target and a required leaf in `not_reviewed` both fail `coverage_audit()`.
- [ ] **Step 2: Run the focused tests** and verify they fail for the missing enforcement path.
- [ ] **Step 3: Implement minimal enforcement** by invoking the support matrix from the canonical HOST audit before general template admission checks.
- [ ] **Step 4: Run focused HOST/support-matrix tests** and verify PASS.
- [ ] **Step 5: Commit** the support-contract slice.

### Task 2: Define Item Registration API Epochs as HOST Facts

**Files:**
- Modify: `minecraft_mod_ai/populate_version_artifact_rules.py`
- Modify: `minecraft_mod_ai/data/host_version_catalog.json` through the repository's canonical catalog-generation/publish path, not by ad-hoc hand editing if a generator exists
- Modify: `minecraft_mod_ai/data/official_version_evidence.json` only when evidence records are missing
- Test: extend existing artifact-rule / resolved-context / template-lowering tests

**Interfaces:**
- Consumes: exact Minecraft target and mappings regime
- Produces: deterministic item-registration epoch facts and API symbols for three families:
  - pre-1.21.2: direct Mojang-mapped `ResourceLocation` registration without required `setId`
  - 1.21.2 through 1.21.10: `ResourceKey` + `ResourceLocation` + `setId`
  - 1.21.11 and later: `ResourceKey` + `Identifier` + `setId`

- [ ] **Step 1: Write failing boundary tests** for 1.21.1 versus 1.21.2 and for 1.21.10 versus 1.21.11.
- [ ] **Step 2: Assert mapping-family consistency**: official-Mojang targets must not emit Yarn package owners or factories such as `net.minecraft.registry.*` / `Identifier.of` when the target contract expects Mojang names.
- [ ] **Step 3: Run focused tests** and capture the exact failure produced by the current mixed-era facts.
- [ ] **Step 4: Implement explicit epoch resolution** in the canonical HOST artifact-rule generator using target comparison only to select a reviewed epoch row; keep the row contents fixed and evidence-backed.
- [ ] **Step 5: Regenerate/publish catalog data via the existing canonical process.**
- [ ] **Step 6: Run the focused boundary and catalog-integrity tests.**
- [ ] **Step 7: Commit** the item epoch HOST slice.

### Task 3: Split Item Executable Lowering by HOST-Selected Implementation

**Files:**
- Modify/create focused templates under `minecraft_mod_ai/templates/fabric/item/` according to the current canonical leaf/template naming scheme
- Modify: canonical leaf-binding construction in `minecraft_mod_ai/populate_version_artifact_rules.py`
- Modify: `minecraft_mod_ai/task_template_catalog.py` only if template admission requires a catalog entry
- Test: extend current item leaf/rendering tests

**Interfaces:**
- Consumes: HOST item-registration epoch from Task 2
- Produces: `minecraft/item/registry` binding to exactly one admitted implementation for the target

- [ ] **Step 1: Write failing rendering tests** that render each supported item epoch and assert exact imports/factory/registration shape.
- [ ] **Step 2: Create or refactor focused implementation templates** so no one template contains mutually incompatible API eras.
- [ ] **Step 3: Bind `minecraft/item/registry` deterministically in HOST**; the model must not select the API epoch or template.
- [ ] **Step 4: Run item lowering/template-contract tests.**
- [ ] **Step 5: Prove old mixed templates are unreferenced** by leaf bindings, template catalog, runtime lookup, and tests.
- [ ] **Step 6: Delete only the superseded mixed template(s)** after the proof in Step 5.
- [ ] **Step 7: Re-run template catalog/reachability/integrity tests.**
- [ ] **Step 8: Commit** the item lowering migration.

### Task 4: Make Deterministic Template Context Projection Complete

**Files:**
- Modify: `minecraft_mod_ai/version_template_context.py`
- Modify: direct consumers found by code search that independently infer target facts
- Test: extend context-projection and override-rejection tests

**Interfaces:**
- Consumes: `ResolvedVersionContext` deterministic domains
- Produces: one immutable projection of capabilities, dependencies/repositories, API symbols, schemas, artifact rules, leaf bindings, replacements, compatibility, and target paths required by templates/consumers

- [ ] **Step 1: Write failing tests** for every HOST deterministic domain that a production consumer needs but the projection omits.
- [ ] **Step 2: Write override-rejection tests** proving model/template input cannot replace HOST-owned deterministic values.
- [ ] **Step 3: Extend the canonical projection minimally** and remove consumer-local fallback/inference paths encountered by the tests.
- [ ] **Step 4: Run projection, resolver, and consumer integration tests.**
- [ ] **Step 5: Commit** the projection migration.

### Task 5: Exhaustive Template Producer/Consumer/Reachability Audit

**Files:**
- Modify: existing template-contract validation module(s)
- Modify: production templates under `minecraft_mod_ai/templates/` only where the audit proves authority duplication, missing producer/consumer, or dead responsibility
- Test: existing template-contract validation suite

**Interfaces:**
- Consumes: production template catalog, renderer placeholders, schema fields, runtime entrypoints
- Produces: machine-checkable failures for duplicate authority, producerless consumed fields, consumerless produced fields unless diagnostic, target-fact requests from model templates, and unreachable non-standalone templates

- [ ] **Step 1: Add failing contract fixtures/tests** for each invalid graph condition.
- [ ] **Step 2: Implement graph extraction from the existing template metadata/contracts rather than inventing a second catalog.**
- [ ] **Step 3: Run the audit across the complete production template catalog.**
- [ ] **Step 4: For each real failure, classify the field/template as HOST-owned, model-owned, derived, or invalid/redundant.**
- [ ] **Step 5: Migrate HOST-owned fields to context injection; compute derived fields in code; retain model-owned semantic fields.**
- [ ] **Step 6: Delete invalid/redundant fields/templates only after runtime reachability and replacement proof.**
- [ ] **Step 7: Re-run full template-contract validation.**
- [ ] **Step 8: Commit** in small independent template-family slices rather than one bulk deletion commit.

### Task 6: Expand Evidence-Backed HOST Rows Across Every Supported Canonical Leaf

**Files:**
- Modify: `minecraft_mod_ai/populate_version_artifact_rules.py` or smaller canonical helper modules factored from it when responsibility is too large
- Modify: HOST catalog/evidence data through canonical publish tooling
- Modify: leaf-specific implementation templates only when an API epoch genuinely differs
- Test: leaf-family target-lowering tests

**Interfaces:**
- Consumes: supported-target list and canonical required leaves
- Produces: complete evidence-backed target rows for item, block/block-item, entity/entity-type/spawn-egg, block entity, GUI/menu/screen, networking payloads, worldgen/data registries, recipe/loot/tag/resource/data/assets, commands/events/components, and every other actually supported canonical leaf

- [ ] **Step 1: Enumerate current required canonical leaves from code at the current HEAD.**
- [ ] **Step 2: For one leaf family at a time, write a completeness test that fails if any supported target lacks required symbols/schema/binding.**
- [ ] **Step 3: Research exact target boundaries from primary sources or existing pinned official evidence.**
- [ ] **Step 4: Add reviewed HOST rows/epochs only for facts supported by evidence; leave unknown combinations fail-closed.**
- [ ] **Step 5: Add/adjust executable implementation templates only where reviewed API differences require them.**
- [ ] **Step 6: Regenerate catalog and run focused leaf-family tests plus HOST completeness.**
- [ ] **Step 7: Commit each independently green leaf-family migration.**

### Task 7: Remove Independent Target Guessing and Legacy Deterministic Fallbacks

**Files:**
- Modify: all consumers located by searches for direct Minecraft-version conditionals, API-name guesses, hardcoded dependency coordinates, schema/path guesses, and latest-target fallback outside HOST construction
- Test: consumer-specific regression tests plus repository-wide deterministic-authority contract tests

**Interfaces:**
- Consumes: HOST context projection from Task 4
- Produces: deterministic consumers that fail with a HOST-unavailable error rather than infer a nearby target

- [ ] **Step 1: Add repository contract tests/search assertions** for forbidden deterministic fallback patterns where mechanically enforceable.
- [ ] **Step 2: Replace each proven consumer-local deterministic authority with the resolved HOST fact.**
- [ ] **Step 3: Ensure unsupported/unreviewed target facts raise the canonical fail-closed error.**
- [ ] **Step 4: Run consumer integration and negative-path tests.**
- [ ] **Step 5: Commit** by consumer subsystem.

### Task 8: End-to-End Supported-Target Verification

**Files:**
- Modify: CI workflow only if a required existing gate is not actually invoked
- Test: all relevant repository tests and GitHub Actions workflows

**Interfaces:**
- Consumes: final `main` HEAD
- Produces: evidence that every product-promised target and canonical leaf survives catalog integrity, template contracts, lowering/rendering, generation/integration, and available compile/GameTest/runtime gates

- [ ] **Step 1: Run focused local-equivalent test commands available through repository CI for HOST, templates, resolver/context, lowering, and generation integration.**
- [ ] **Step 2: Inspect exact final-HEAD GitHub Actions workflow runs.**
- [ ] **Step 3: For every failure, fetch job steps/logs, identify root cause, add a reproducing test where appropriate, fix, and recommit.**
- [ ] **Step 4: Repeat until relevant workflows for the exact final HEAD conclude successfully.**
- [ ] **Step 5: Run `production_readiness_audit()` and record PASS for the supported scope.**
- [ ] **Step 6: Verify no deletion left an unresolved template/catalog/runtime reference.**
- [ ] **Step 7: Report exact final HEAD, relevant successful workflows, and any explicitly unsupported targets/leaves; do not claim completion if any required gate remains red.**
