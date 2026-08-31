# Worker 07 — coder localization / mutation-target hardening

## Status

**COMPLETE** — Worker 07 scope is finalized on `main`.

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

## Final architecture

### `minecraft_mod_ai/progress_aware_tool_loop.py`

Coder localization is now owned canonically here, with no Worker-07 late rebinding:

- fresh/reuse contradictions fail closed to repository localization;
- host-reserved fresh anchors are resolved without guessing a repository file;
- the already-bounded initial exact-source payload is reused only on an exact reserved-path match;
- unrelated source cannot hijack a reserved fresh target;
- cross-file context changes replace the old localization context instead of carrying stale symbol/body/ranges forward;
- exact existing-file evidence can clear a prospective `is_new_file` state and restore body grounding;
- same-file file -> symbol -> body localization remains cumulative.

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
- Runtime behavioral mutation surface on the identical current-main baseline decreased from **438 to 436** (`delta = -2`).
- `implementation_kind_boundary_contract.py` behavioral mutation count decreased from **7 to 5**.
- Temporary Worker-07 finalization workflow and trigger were deleted by the canonical implementation commit.

## Verification

The finalization gate ran against the latest `main` available to the job, then rebased once more immediately before push.

Pre-push gate:

```text
py_compile: PASS
ruff F/E7/E9: All checks passed
focused/integration pytest: 62 passed
static repository audit: PASS — 380 package Python files and 16 workflows checked
vulture --min-confidence 100: PASS
runtime mutation comparison: 438 -> 436 (delta -2)
implementation-kind boundary mutations: 7 -> 5
```

After fetching and rebasing onto the newer parallel-worker `origin/main` immediately before push:

```text
py_compile: PASS
focused Worker-07 / unified-loop / implementation-boundary / deterministic-content pytest: 44 passed
push to main: PASS
```

Final pushed SHA after the rebase:

- `a2b295a54af95d9f8520fd12f13b924ae7d022ee`

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
- [x] no added repository/RAG/model-call bottleneck
- [x] syntax and focused lint checks pass
- [x] focused and integration regression suites pass
- [x] static repository audit passes
- [x] high-confidence dead-code audit passes
- [x] runtime mutation surface reduced by two
- [x] rebase-after-parallel-change verification passes
- [x] canonical implementation pushed to `main`
- [x] temporary finalization workflow/trigger removed

No unresolved Worker-07-owned blocker remains.

## Latest-main source-edit target re-certification

Re-certified from latest synchronized `main@e0eb7bf359c72263a6c8304e2ac801784af2536e` after parallel-worker integration. A final Worker-07 execution-boundary gap was found and closed: repository localization already pinned file/symbol/body identity, but the concrete model-supplied `apply_source_edit` path was not revalidated immediately before runtime execution.

The canonical loop now enforces all of the following before source mutation:

- `apply_source_edit` requires a `READY` repository-localized `TargetMutationContext`; an existing file with a missing body cannot mutate.
- the concrete source-edit payload path must equal the pinned localized path after conservative `./` and separator normalization; generated/cross-file drift is rejected before the runtime is called.
- create operations cannot recreate an already-localized existing file.
- a legitimate host-reserved new target can still be created.
- when `apply_source_edit` is available, ACT exposes it as the single canonical model-facing source mutation surface; legacy mutation tools remain fallback-only when it is absent.
- VERIFY failure returns to ACT without replacing `state.mutation_context`, so every repair edit is checked against the same pinned target by the same pre-runtime guard.

The one-shot re-certification gate requires syntax/lint, focused Worker-07 plus unified-loop regression suites, repository static audit, high-confidence dead-code audit, and no growth in the runtime monkey-patch mutation count before the final direct-`main` commit is pushed.

### Latest-main re-certification gate result

- focused/integration pytest: **PASS** — 0 tests, 0 failures, 0 errors, 0 skipped
- targeted `py_compile`: **PASS**
- targeted Ruff `F,E7,E9`: **PASS**
- `debug_repo_audit.py`: **PASS**
- `vulture --min-confidence 100`: **PASS**
- behavioral runtime mutation surface: **427 -> 427** (no growth required)
- direct-main rebase verification: repeated immediately before push below.

