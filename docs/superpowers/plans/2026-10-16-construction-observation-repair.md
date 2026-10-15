# Construction Observation and Repair Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with static review.

**Goal:** Give the small model a durable action/observation/local-repair loop.

**Architecture:** Share one JavaOperations validator contract between partial
admission and final validation. Keep typed operations, host-owned selection,
source reuse, and exact checkpoint identity. Repair one node through an atomic
revision and preserve unrelated work.

**Tech Stack:** Python, native JSON schemas, existing typed Java IR/checkpoints.

**Spec:** `docs/superpowers/specs/2026-10-16-construction-observation-repair.md`

## Global Constraints

- No tests or model inference in this session, as explicitly requested.
- No new token, node, task, or elapsed-time limits.
- Do not remove RAG, donor integration, or final Java/runtime verification.
- Commit each implemented change and publish to main after static review.

## Review Focus

- A forward input reveals an invalid previously saved parent.
- Repair must not discard independent roots or shared values.
- Crash between graph and cursor saves must retain child purposes and repair state.
- A validator must not rescan the entire target inventory on every node.
- Completion errors, cancellation, and unrelated host errors must remain distinct.

### Task 1: Shared partial validation

- [ ] Refactor `java_operation_ir.py` contract preparation from node validation.
- [ ] Add partial validation with explicit holes, cycle checks, and located errors.
- [ ] Retain final roots/return/control/exception validation unchanged.
- [ ] Inspect AST and diff statically, then commit.

### Task 2: Durable observations and revisions

- [ ] Add `construction_feedback.py` for exact diagnostics and node replacement.
- [ ] Extend `construction_progress.py` with graph CAS and atomic author snapshots.
- [ ] Extend state payload/export/restore for one bounded observation/repair target.
- [ ] Inspect callers and persistence boundaries, then commit.

### Task 3: Wire action/observation/local repair

- [ ] Validate candidate nodes before normal append and handle located diagnostics.
- [ ] Feed the rejected node and actual diagnostic into same-slot reconstruction.
- [ ] Reopen provisional parents through the atomic revision callback.
- [ ] Expose validated/deferred counts without claiming gameplay completion.
- [ ] Run static AST/name/diff checks only, obtain static review, and commit.
- [ ] Publish main with expected-head verification and byte-identical tree check.
