# Worker 07 — coder localization / mutation-target hardening

## Status

**COMPLETE** — Worker 07 scope is finalized on `main`.

**READY_FOR_WORKER_13: YES**

Latest-main source-edit pinning commit:

- `5a26cf7ff0ae5205f108f14c79a72406adfbe86c` — `fix(worker07): pin source edits to localized targets`

Canonical implementation commit:

- `a2b295a54af95d9f8520fd12f13b924ae7d022ee` — `refactor: make coder target hardening canonical`

Initial hardening checkpoint:

- `e653e976e15a1ba702dbc08f5c88cc37429011b2` — `fix: harden coder target localization`

Branch policy was respected throughout: direct `main`, no worker branch, no force push. Parallel-worker changes were preserved by re-syncing/rebasing against current `origin/main` before the final push.

## Scope

Worker 07 owns repository-aware coder targeting, `apply_source_edit` / source-mutation localization safety, file-symbol-body target identity, and repair-loop target stability. Planner semantics and Minecraft API/domain selection remain outside this worker's ownership.

## Root causes fixed

1. Fresh-target handling and target-identity hardening had been installed as late runtime monkey-patches from `implementation_kind_boundary_contract.py` instead of being owned by the coder execution loop.
2. A nominal `fresh` task could be treated as mutation-ready even when contradictory reuse/component/source evidence was present.
3. `TargetMutationContext.merge` could mix file A's symbol/body/ranges with file B's path and could preserve `is_new_file=True` after exact evidence proved the target already existed.
4. A host-reserved fresh path already present in bounded initial exact-source evidence could bypass existing-source body localization.
5. During end-to-end coder-loop verification, `deterministic_minecraft_content_contract._role_dynamic_tools` was found calling the removed `skills_for_model_role()` API, causing coder tool execution to fail before localization. It now consumes the current single role-policy snapshot.
6. Unified coder-loop tests used `MagicMock` model configs without an explicit `max_input_tokens`, which became invalid after the newer llama context-safety contract. The fixture now explicitly represents the unset override as `0`.
7. The final execution-boundary audit found that a model-supplied `apply_source_edit` path could reach runtime without being compared against the already-localized target. That final drift path is now rejected before runtime execution.

## Final architecture

### `minecraft_mod_ai/progress_aware_tool_loop.py`

Coder localization is now owned canonically here, with no Worker-07 late rebinding:

- fresh/reuse contradictions fail closed to repository localization;
- host-reserved fresh anchors are resolved without guessing a repository file;
- the already-bounded initial exact-source payload is reused only on an exact reserved-path match;
- unrelated source cannot hijack a reserved fresh target;
- cross-file context changes replace the old localization context instead of carrying stale symbol/body/ranges forward;
- exact existing-file evidence can clear a prospective `is_new_file` state and restore body grounding;
- same-file file -> symbol -> body localization remains cumulative;
- `apply_source_edit` requires a `READY` localized target immediately before runtime execution;
- the concrete edit payload path must match the pinned target path;
- create operations cannot recreate a localized existing target;
- ACT prefers the canonical `apply_source_edit` surface whenever it is available, while preserving legacy mutators only as a compatibility fallback when it is absent;
- VERIFY -> ACT repair keeps the same `state.mutation_context`, so repair edits are checked against the same target pin.

### `minecraft_mod_ai/implementation_kind_boundary_contract.py`

The module is back to its single responsibility: implementation-kind routing. Worker 07's target-localization monkey-patches and duplicate fresh-target helpers were deleted.

### `minecraft_mod_ai/deterministic_minecraft_content_contract.py`

Dynamic coder tools now read `capability_module._role_policy_snapshot(stage, model_role).skills`, removing the stale call to the deleted `skills_for_model_role()` API.

## Clean-code / performance result

- No new repository walk.
- No new RAG request.
- No new model call.
- No duplicate fresh-target parser.
- No Worker-07 late `TargetMutationContext.merge` monkey-patch.
- No Worker-07 late `_extract_mutation_context_from_payload` monkey-patch.
- Existing bounded source evidence and existing host state are reused.
- Original canonicalization reduced the reviewed behavioral mutation surface from **438 to 436** (`delta = -2`) and `implementation_kind_boundary_contract.py` from **7 to 5**.
- Latest-main source-edit target pinning added **no** runtime monkey-patch growth: **427 -> 427** on the re-certification baseline.
- Temporary Worker-07 re-certification workflow, trigger, and patcher were removed before the final code push.

## Verification

The original canonicalization gate ran against the latest `main` available to that job, then rebased once more immediately before push.

Original pre-push gate:

```text
py_compile: PASS
ruff F/E7/E9: All checks passed
focused/integration pytest: 62 passed
static repository audit: PASS — 380 package Python files and 16 workflows checked
vulture --min-confidence 100: PASS
runtime mutation comparison: 438 -> 436 (delta -2)
implementation-kind boundary mutations: 7 -> 5
```

Original post-rebase gate:

```text
py_compile: PASS
focused Worker-07 / unified-loop / implementation-boundary / deterministic-content pytest: 44 passed
push to main: PASS
```

## Latest-main source-edit target re-certification

The final re-certification job synchronized the then-current `main`, applied the execution-boundary pinning patch, ran the full Worker-07 focused/integration gate, fetched parallel-worker changes again, successfully rebased onto `main@b33f72ffbfb1aa78de26569ad3833cb465d70e3a`, repeated the same gate after that rebase, and pushed the resulting Worker-07 code commit directly to `main`.

Final re-certification run:

- GitHub Actions run: `33358754134`
- focused/integration pytest: **PASS — 70 passed, 0 failed**
- targeted `py_compile`: **PASS**
- targeted Ruff `F,E7,E9`: **PASS**
- `debug_repo_audit.py`: **PASS — 375 package Python files checked**
- `vulture --min-confidence 100`: **PASS**
- behavioral runtime mutation surface: **427 -> 427** — no growth
- rebase onto newest parallel `main`: **PASS**
- identical post-rebase focused/integration suite: **PASS — 70 passed**
- post-rebase static audit: **PASS — 375 package Python files and 14 workflows checked**
- direct push to `main`: **PASS**

Final Worker-07 code SHA:

- `5a26cf7ff0ae5205f108f14c79a72406adfbe86c`

## Regression coverage

Worker-07 coverage includes:

- exact reserved target already present => existing target, not blind creation;
- incidental source cannot hijack a legitimate fresh target;
- legitimate new target remains creatable;
- contradictory task/binding reuse evidence blocks fresh bypass;
- different-file localization cannot inherit stale symbol/body state;
- path-only retargeting resets previous body/symbol state;
- repository evidence converts a prospective new target into an existing target;
- same-file hierarchical localization remains cumulative;
- existing target without a concrete body cannot mutate;
- model-supplied cross-file/generated path drift is rejected before runtime;
- equivalent `./path` normalization does not cause false drift rejection;
- create operations on an existing localized file are rejected;
- reserved new-target creation remains allowed;
- ACT prefers `apply_source_edit` when multiple mutation surfaces are exposed;
- legacy mutation fallback remains available only when `apply_source_edit` is absent;
- VERIFY -> ACT repair is checked against the same pinned target context;
- no Worker-07 late monkey-patch owner remains;
- coder tool routing uses the current role-policy API.

## Completion checklist

- [x] repository-aware target localization hardened
- [x] blind new-file bypass blocked when exact existing evidence exists
- [x] fresh/reuse contradiction fails closed
- [x] file/symbol/body identity contamination blocked
- [x] legitimate fresh creation preserved
- [x] duplicate target parser removed
- [x] Worker-07 late runtime monkey-patches removed
- [x] stale coder dynamic-skill API repaired
- [x] concrete `apply_source_edit` path pinned to localized target before runtime
- [x] cross-file/generated target drift blocked
- [x] existing-target recreation blocked
- [x] missing-body existing mutation blocked
- [x] repair-loop target pin preserved
- [x] canonical model-facing source editor preferred
- [x] no added repository/RAG/model-call bottleneck
- [x] syntax and focused lint checks pass
- [x] focused and integration regression suites pass
- [x] post-rebase focused and integration suites pass
- [x] static repository audit passes
- [x] high-confidence dead-code audit passes
- [x] latest hardening adds zero runtime monkey-patch growth
- [x] rebase-after-parallel-change verification passes
- [x] final hardening pushed to `main`
- [x] temporary re-certification workflow/trigger/patcher removed
- [x] progress evidence corrected and ready for Worker 13 handoff

No unresolved Worker-07-owned blocker remains.
