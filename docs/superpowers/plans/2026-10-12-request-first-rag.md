# Request-first RAG Implementation Plan

> Execute inline with superpowers:executing-plans; work on main.

**Goal:** Connect original feature decomposition, source RAG and residual implementation.

**Architecture:** Persist a feature ledger across planning and production; retain
the existing content graph, source adaptation and target compiler.

**Tech Stack:** Python, ProjectRAGIndex, structured local model calls, pytest.

**Spec:** ../specs/2026-10-12-request-first-rag-design.md

## Global constraints

- Preserve user scope without game-specific required state or currency.
- Keep legacy plans readable and introduce no new approval/certification gate.
- Search references do not prove transplanted or target-compatible code.
- Keep finite model output pages; commit on main.

## Review focus

- Korean compound requests with more than four independent features.
- Resume and saved plans without a ledger.
- Optional RAG failure must not prevent authoring.
- Older mod source remains discoverable for adaptation.
- Wrapper recovery remains checksum verified.

## Task 1: Request feature preservation

Files: request_scope.py, authored_plan.py, planner_checkpoint.py,
complete_planner.py, authored_content_contract.py, authored_reuse_bridge.py.

- [x] Add failing feature, search, save/load and RAG-context tests.
- [x] Implement author_request_scope(router, prompt, *, budget=None) -> dict.
- [x] Carry the ledger into worksheet context, content discovery and source search.
- [x] Run authored plan/content/reuse tests and commit.

## Task 2: State continuity and active worksheet pages

Files: authored_structured_design.py, worksheet_atomic_chunker.py,
execution_contract_policy.py.

- [x] Reproduce lost prerequisites and unnecessary concern calls.
- [x] Decide section concern counts once and author only active pages.
- [x] Carry behavior into state calls and allow complete compact text.
- [x] Run worksheet/planner/state tests and commit.

## Task 3: Donor discovery and wrapper preparation

Files: reuse_discovery.py, verified_scaffold_registry.py.

- [x] Reproduce exact-version search filtering and missing wrapper download.
- [x] Keep source search target-neutral and leave compatibility downstream.
- [x] Fetch exact release wrapper using official checksum.
- [x] Run reuse/wrapper tests and commit.

## Task 4: Readable production progress

Files: llama_completion_liveness_contract.py and logging tests.

- [x] Reproduce redundant lifecycle output.
- [x] Retain completion/failures/watchdogs and make transport details opt-in.
- [x] Run lifecycle tests and commit.

## Task 5: Integration verification and publication

- [x] Run the project suite and relevant CI audits; fix demonstrated regressions.
- [x] Review against the supplied scenario and original feature scope.
- [ ] Push main, check CI, and report actual verification limits.


## Review follow-ups

- [x] Keep failed deterministic donor migrations as licensed, hash-checked source references.
- [x] Download immutable source archives and validate each selected Git blob hash.
- [x] Persist runtime record owners before Typed authoring; retain exact UI action guards.
- [x] Read actual generated prerequisite APIs after dependency completion.
- [x] Invalidate resumed content when its generator, dispatcher or ownership implementation changes.
- [x] Pass actual template/native owner paths and registry constants into implementation tasks.
- [x] Record unavailable optional source references and continue fresh implementation.
- [x] Inspect/prove ranked source batches lazily while retaining the whole candidate frontier.
- [x] Continue feature decomposition after 32 records without increasing a model output page.
- [x] Detect repeated feature identities and use bounded recovery without adding a feature-count limit.

Validation: 320 production/source contracts passed with six JVM/environment skips;
focused follow-up regressions are tracked in test_request_runtime_ownership.py and
test_production_checkpoint_fingerprint.py. A public Fabric example commit archive
was downloaded and 12 source/configuration files verified against Git blob hashes.
A new full local-model gameplay JAR has not been produced in this environment.
