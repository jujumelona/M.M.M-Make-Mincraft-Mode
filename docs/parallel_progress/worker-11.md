WORKER: 11
ROLE: Errors + Observability + Diagnostics + CI
STATUS: COMPLETE
LAST_UPDATED_MAIN_SHA: e1e04bd8eb5e7762c3540bd30a56272a3c6cb2ab

COMPLETED:
- Audited CI, Full Debug audit output, retry/fallback visibility, shared JDT/CLI error boundaries, and legacy diagnostic paths.
- Added one canonical failure taxonomy and diagnostic collector with stable causal fingerprints, explicit retryability/final status, and attempt aggregation.
- Added compact operator rendering in ROOT FAILURE / CAUSE / ATTEMPTS / FALLBACK / FINAL STATUS form.
- Preserved INTERNAL tracebacks in debug payloads while suppressing repeated traceback dumps from user/CI summaries.
- Applied sanitizer coverage to both compact causes and retained debug traceback text.
- Added deterministic failure-injection tests plus a dedicated Observability Regression workflow.
- Added `tools/root_cause_audit_wrapper.py`: raw Full Debug output remains in artifacts while console output is causal and compact.
- Normalized Full Debug artifact semantics: only PASS has `passed=true`; WARN/SKIP are explicit non-blocking states; FAIL is explicit blocking failure; unknown states fail closed.
- Added `tools/pytest_diagnostics.py`: noisy pytest shards preserve raw output + JUnit artifacts while console output deduplicates repeated causal failures and lists affected tests.
- Rewired the 3-way `remaining-tests` CI shards to the compact pytest diagnostic runner and always upload raw diagnostics.
- Removed the obsolete, unreferenced `.github/scripts/annotate_pytest_failures.py` duplicate diagnostic path after repository-wide reference search returned zero uses.
- Created Worker12 handoff for the shared JDT/CLI broad-exception boundary rather than rewriting shared orchestration from Worker11.

IN_PROGRESS:
- none

ROOT_CAUSES_CONFIRMED:
- Failure events previously had no canonical taxonomy/fingerprint, so one root cause could appear as repeated warning/traceback/skipped symptoms instead of one causal record with attempts.
- Full Debug mixed PASS/WARN/FAIL/SKIP with a `passed` boolean that marked WARN/SKIP as passed, creating contradictory machine-readable semantics.
- Full Debug emitted large raw output directly to CI console instead of separating compact operator diagnostics from full debug artifacts.
- The large 3-way pytest shard directly emitted `pytest -vv --tb=short --maxfail=25`, so related failures could flood the log with repeated traceback material.
- A separate JUnit annotation script duplicated failure rendering but was not referenced by any workflow/code path.
- Shared `CompleteProductionOrchestrator.execute` still has a Worker12-owned broad JDT `except Exception` boundary that can relabel programming errors as UNAVAILABLE; this is captured in the handoff and is not a Worker11-owned unresolved item.

DECISIONS_AND_EVIDENCE:
- Failure category/retryability are explicit boundary decisions rather than guessed from arbitrary exception strings.
- INTERNAL traceback evidence is retained for debugging but not repeated in compact operator summaries.
- Identical causal failures aggregate by fingerprint with ATTEMPTS while affected tests/artifacts remain separately attributable.
- Raw pytest/audit logs are preserved as artifacts; concise console rendering is not evidence destruction.
- Only PASS is semantically `passed=true`; WARN/SKIP are non-blocking but not successful checks.
- Unsupported diagnostic status values fail closed rather than silently becoming success.
- Repository search for `annotate_pytest_failures` returned zero consumers before deletion; current CI uses `tools/pytest_diagnostics.py`.
- Shared orchestrator/CLI ownership was respected: cross-owner structural work was documented in `docs/parallel_handoff/worker-11.md`.

COMMITS_ALREADY_PUSHED:
- 2a654c4866fb7f41ea163a84da6578d2a30c3923 feat: add root-cause diagnostics contract
- 36416f7b5bbed491dc01dc4ee6e446612dfc3c91 test: inject diagnostic failure regressions
- d933299acc59da6dae3923b9fd6fdcd289d4e162 ci: add observability failure-injection gate
- cdcebbcebf4c8622fff814263b0182aa920b11cd docs: hand off shared diagnostic boundary fix
- ba771623ef1266dc3d665ae2b016292e39a382d8 docs: checkpoint worker 11 diagnostics
- 5d9d5a0ca3294b9c33c3806059a2852ab61b4ed5 feat: compact full-debug root-cause output
- 73bceb8931abc63150c62bee4343cbc1965d2e23 test: cover compact debug diagnostics
- fa53bcd4d4bac53d57291a77b5067e36c85d2b4f ci: extend observability failure injection
- 60fca4e0a7a8c140544f1cb07044164023ea7256 ci: compact full-debug failure output
- 6dd2d26183bb435476e1b0545dc4ed331f324eca fix: disambiguate Debug audit status semantics
- 884264c3e2e829f29980a02913fbd766d43fa367 test: enforce Debug status semantics
- 95a56674229d6699e6ef8027111cbd8d86c0501c feat: add compact pytest diagnostic runner
- c4ec9dcd8947f8e584e44073306dbc5ca901dbad test: cover pytest causal deduplication
- d9c197327de341af205bbddb2a69a8e96a6bbf8a ci: regression-test pytest diagnostics
- 9cd5a25337f1d2fc6ce959ce31da0b70fd86aa4b ci: compact noisy pytest shard failures
- e1e04bd8eb5e7762c3540bd30a56272a3c6cb2ab refactor: remove obsolete pytest annotation script

TESTS_ALREADY_PASSING:
- `python -m pytest -q tests/test_diagnostics.py` -> 4 passed (isolated contract run).
- diagnostics + Full Debug wrapper isolated regression run -> 7 passed.
- Debug status-semantics isolated regression run -> 2 passed.
- Observability Regression run 33325981268 -> SUCCESS after Full Debug integration.
- Observability Regression run 33326415281 on integrated main `045dc05a001d2aae771a0e999d571aca63649f5d` -> SUCCESS, including diagnostics, Full Debug wrapper/status semantics, and pytest causal-dedup tests.
- Dead annotation script removal: repository-wide reference search for `annotate_pytest_failures` -> zero consumers before deletion.

NEXT_EXACT_ACTIONS:
1. Worker13 final integrator must verify all Worker11 SHAs remain ancestors of final origin/main.
2. Worker12/final integrator must consume `docs/parallel_handoff/worker-11.md` for the shared JDT/CLI broad-exception boundary if Worker12 has not already done so.
3. Final integration should run the whole-repository CI/Full Debug gate after all parallel pushes settle.

FILES_CURRENTLY_RELEVANT:
- minecraft_mod_ai/diagnostics.py
- tools/root_cause_audit_wrapper.py
- tools/pytest_diagnostics.py
- tests/test_diagnostics.py
- tests/test_root_cause_audit_wrapper.py
- tests/test_pytest_diagnostics.py
- .github/workflows/observability-regression.yml
- .github/workflows/full-debug-gate.yml
- .github/workflows/ci.yml
- docs/parallel_handoff/worker-11.md
- docs/parallel_progress/worker-11.md

KNOWN_CROSS_ROLE_DEPENDENCIES:
- Worker12 owns `minecraft_mod_ai/complete_orchestrator.py` and CLI/top-level shared error presentation. `docs/parallel_handoff/worker-11.md` requests narrowing the JDT broad catch so INTERNAL programming errors cannot become dependency UNAVAILABLE and requests canonical diagnostic integration at that shared boundary.
- Worker10 consumes JDT/validation semantics; Worker11 changes intentionally preserve validation behavior while clarifying diagnostic state.

UNRESOLVED:
- none
