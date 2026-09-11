# Host-Owned Relation Cardinality Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove model-authored relation arrays and make relation cardinality, iteration, endpoint ownership, validation, and termination host-owned without truncating authored relations to the model array limit.

**Architecture:** For each ordered entity pair, a scalar `design/content_relation_count` record determines how many explicit relations the requirement needs. The host then requests exactly one `relation_type` per iteration, injects the already-fixed source/target IDs itself, rejects duplicates or impossible cardinality, and never asks the model to emit an unbounded array. The legacy `design/relation_set` array contract is removed.

**Tech Stack:** Python 3.11, YAML task templates, JSON Schema Draft 2020-12, pytest, GitHub Actions.

**Spec:** Existing `model_output_atomicity_contract.py`, `design_record_runtime.py`, and content-graph fail-closed behavior.

## Global Constraints

- Work only on `main`; do not create a branch.
- Do not add arbitrary `maxItems` truncation to authored relations.
- Model-authored structured outputs must satisfy the global atomicity contract.
- Source and target entity IDs remain host-owned and must never be re-selected by the model.
- Unknown, duplicate, or semantically impossible relation cardinality must fail closed.
- Verify affected tests first, then the full CI shards.

---

### Task 1: Add the regression test

**Files:**
- Create: `tests/test_relation_cardinality_runtime.py`

**Interfaces:**
- Consumes: `run_record_template(router, "design/content_relation", context=...)`
- Produces: a regression proving more than four explicit relations for one ordered pair are emitted without an array-shaped model response.

- [ ] **Step 1: Write the failing test**

Create a deterministic router that accepts `submit_one_design_content_relation_count` and `submit_one_design_content_relation`, but deliberately has no `submit_one_design_relation_set` path. Assert that five explicit relation types for one pair are returned in source order and that every structured call is scalar/atomic.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_relation_cardinality_runtime.py`

Expected: FAIL on the current runtime because it still calls `design/relation_set` / `submit_one_design_relation_set` and its schema contains an unbounded `relations` array.

### Task 2: Replace relation-set arrays with host-owned repetition

**Files:**
- Create: `minecraft_mod_ai/templates/design/content_relation_count.yaml`
- Modify: `minecraft_mod_ai/templates/design/content_relation.yaml`
- Modify: `minecraft_mod_ai/design_record_runtime.py`
- Modify: `minecraft_mod_ai/template_contract_validation.py`
- Delete: `minecraft_mod_ai/templates/design/relation_set.yaml`

**Interfaces:**
- Consumes: scalar `{"count": integer >= 0}` for one fixed ordered pair.
- Produces: host-injected `{relation_type, source_id, target_id}` records.

- [ ] **Step 1: Define scalar count contract**

`content_relation_count.yaml` returns only `count`. Rules state that the fixed source/target pair is host supplied, zero means no explicit relation, authored cardinality must be preserved, and the model owns neither iteration nor completion.

- [ ] **Step 2: Make one-relation output truly atomic**

`content_relation.yaml` returns only one `relation_type`, using a closed enum for all supported ordinary relations and `key_A` through `key_Z` plus `key_0` through `key_9`. Endpoints are removed from model output because they are already host facts.

- [ ] **Step 3: Implement host-owned iteration**

For each ordered entity pair: request count once, reject counts above the finite semantic maximum (all ordinary relation kinds plus one key relation), request exactly `count` one-relation records with host-supplied ordinal context, reject duplicates and multiple key relations, then inject source/target IDs.

- [ ] **Step 4: Remove the obsolete array authority**

Delete `relation_set.yaml`, remove it from runtime consumer roots, and register `content_relation_count` instead.

### Task 3: Align existing deterministic tests

**Files:**
- Modify: `tests/test_atomic_design_pipeline.py`

**Interfaces:**
- Consumes: the new count + one-relation tool calls.
- Produces: deterministic fixtures that exercise the same production protocol instead of an obsolete `relations[:4]` truncation.

- [ ] **Step 1: Update the GraphRouter fixture**

Return pair-local scalar counts from `submit_one_design_content_relation_count`. For `submit_one_design_content_relation`, select the requested pair-local relation by host-owned `record_index` and return only `relation_type`.

- [ ] **Step 2: Add the count template to atomicity coverage**

Validate `design/content_relation_count` alongside the other atomic design record templates.

### Task 4: Verification

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest -q tests/test_relation_cardinality_runtime.py tests/test_atomic_design_pipeline.py`

Expected: PASS.

- [ ] **Step 2: Run template/runtime contract tests**

Run: `python -m pytest -q tests/test_fixed_template_generation_boundary.py tests/test_template_runtime_ssot.py tests/test_planner_template_schema.py`

Expected: PASS.

- [ ] **Step 3: Run static checks**

Run: `python -m ruff check minecraft_mod_ai/design_record_runtime.py minecraft_mod_ai/template_contract_validation.py tests/test_relation_cardinality_runtime.py tests/test_atomic_design_pipeline.py --select F,E7,E9`

Expected: PASS.

- [ ] **Step 4: Verify GitHub Actions**

Require the latest `main` CI run to complete with the affected planner/static jobs and all deterministic pytest shards green before claiming completion.
