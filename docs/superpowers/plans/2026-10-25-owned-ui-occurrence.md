# Owned UI occurrence implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation and one final independent review.

**Goal:** Construct a requested player action before its gameplay listener needs its ABI.

**Architecture:** Reuse Typed Action forms, codecs and server command transport in a separate generated namespace. Publish a real, host-authored event whose arguments are the server actor and validated player inputs. Preserve existing inspected callbacks and Mixin sources.

**Tech Stack:** Python, generated Java, Fabric/Minecraft 26.2, pytest, javac.

**Spec:** `docs/superpowers/specs/2026-10-25-owned-ui-occurrence-design.md`

## Global constraints

- Work on `main` only, as previously requested.
- No prompt-specific gameplay, invented passive-event activation, or whole-file model fallback.
- Preserve the original behavior contract and server authority.
- Source/manifest receipts are necessary wiring evidence, never runtime certification.

## Review focus

- Passive world events must not become user commands.
- Invalid or forged inputs must not reach the gameplay listener.
- A declined listener must not report successful gameplay.
- Failed body authoring must not leave startup registrations behind.
- Replayed receipts must reject changed files and missing startup entries.

### Task 1: Concrete Typed Action occurrence producer

**Files:** Create `minecraft_mod_ai/source_owned_ui_event.py`; test `tests/test_source_owned_ui_event.py`.

**Interface:** `plan_typed_ui_occurrence(...) -> dict`, `author_typed_ui_occurrence(...) -> dict`, `materialize_typed_ui_occurrence(...) -> dict`, `read_verified_typed_ui_events(...) -> list[dict]`.

- [x] Write and run failing tests for actual server action dispatch, invalid inputs, declined effects, passive triggers, idempotent installation and receipt replay.
- [x] Implement deterministic event/adapter generation using existing Typed Action renderers and bounded input metadata authoring.
- [x] Run producer tests and compile/run its generated Java against a controlled external ABI fixture.

### Task 2: Bind the new producer through the production host

**Files:** Modify `first_pass_java_source.py`, `construction_file_author.py`, `feature_registration_proof.py`, `feature_source_work_graph.py`, `host_callback_dispatch.py`; test `tests/test_source_owned_ui_host.py`.

**Interface:** Use Task 1's planned callback without installing files until bodies exist; replay Task 1's byte-checked receipts during full-source registration proof.

- [x] Reproduce the absent source-site failure through `compose_first_pass_java` with an explicit player action.
- [x] Build its occurrence producer after inspected hooks/sites decline, retain original behavior predicates/effects, and record planned source work separately from verified work.
- [x] Run integration tests, related callback/source suites and the project test suite; record any environment-dependent failures.
- [x] Review, commit and push the concrete fix to `main`; verify the remote ref.

Final evidence: 192 related tests passed, 1 exact-SDK skip; full suite 4,647
passed, 3 remaining failures and 8 skips. See `docs/production_log_21_repair.rst`
for exact remaining failures and verification limits.
