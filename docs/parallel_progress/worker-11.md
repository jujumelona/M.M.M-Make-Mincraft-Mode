WORKER: 11
ROLE: Errors + Observability + Diagnostics + CI
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: cdcebbcebf4c8622fff814263b0182aa920b11cd

COMPLETED:
- Audited the current CI, Full Debug audit path, CLI error boundary, shared orchestrator JDT boundary, and JDT client exception surface.
- Confirmed the JDT shared boundary catches every Exception and converts unexpected programming defects to UNAVAILABLE.
- Added a reusable failure taxonomy and diagnostic collector with root-cause fingerprint deduplication.
- Added compact operator rendering in the required ROOT FAILURE / CAUSE / ATTEMPTS / FALLBACK / FINAL STATUS shape without user-visible traceback spam.
- Preserved INTERNAL tracebacks in debug payloads while suppressing them from the compact user summary.
- Added sanitizer coverage so secrets are removed from both compact causes and stored debug traceback text.
- Added deterministic failure-injection regression tests and a dedicated CI workflow.
- Created a worker-12 handoff for the shared JDT/CLI boundary rather than rewriting shared orchestration from worker 11.

IN_PROGRESS:
- Integrate the new diagnostic contract into worker-11-owned audit reporting and concise Debug console output.
- Verify GitHub Actions for the diagnostics checkpoint and repair any worker-11 regression.
- Audit remaining broad fallback/error boundaries and separate operational failures from programming errors without crossing ownership boundaries.

ROOT_CAUSES_CONFIRMED:
- `CompleteProductionOrchestrator.execute` wraps JDT diagnostics in `except Exception` and returns `status=UNAVAILABLE`, masking programming errors as dependency availability failures.
- Existing failure information had no canonical cross-boundary taxonomy/fingerprint, so repeated symptoms could not be reliably collapsed to one root cause with an attempt count.
- `tools/full_project_audit.py` models PASS/WARN/FAIL/SKIP but `Check.passed` currently treats every non-FAIL status, including WARN and SKIP, as passed; its console also emits the entire report rather than a compact causal summary.
- The main CI suite had no dedicated failure-injection contract proving duplicate failure collapse, INTERNAL traceback retention, and compact rendering.

DECISIONS_AND_EVIDENCE:
- Failure category and retryability are explicit boundary decisions, not inferred from arbitrary exception message text.
- INTERNAL tracebacks are retained for debugging but are not repeated in user-facing summaries.
- Identical causal failures are fingerprinted from stage/operation/category/type/message/artifact and aggregated by attempt count.
- A failure terminal status is explicit (FAILED/UNAVAILABLE/DEGRADED/RECOVERED); unresolved failures never use an ambiguous PASS label.
- Shared orchestrator/CLI changes are handed to worker 12 rather than rewritten by worker 11.

COMMITS_ALREADY_PUSHED:
- 2a654c4866fb7f41ea163a84da6578d2a30c3923 feat: add root-cause diagnostics contract
- 36416f7b5bbed491dc01dc4ee6e446612dfc3c91 test: inject diagnostic failure regressions
- d933299acc59da6dae3923b9fd6fdcd289d4e162 ci: add observability failure-injection gate
- cdcebbcebf4c8622fff814263b0182aa920b11cd docs: hand off shared diagnostic boundary fix

TESTS_ALREADY_PASSING:
- Local isolated contract run: `python -m pytest -q tests/test_diagnostics.py` -> 4 passed.
- Local syntax check: `python -m py_compile minecraft_mod_ai/diagnostics.py tests/test_diagnostics.py` -> PASS.

NEXT_EXACT_ACTIONS:
1. Inspect GitHub Actions for the pushed observability checkpoint and repair any worker-11 regression.
2. Refactor worker-11-owned `tools/full_project_audit.py` to use canonical diagnostics, explicit PASS/WARN/SKIP semantics, and compact causal console output while preserving the full artifact report.
3. Add regression tests for the Full Debug report/console behavior.
4. Re-scan fallback boundaries; perform only worker-11-owned fixes and append shared-core requests to the existing worker-11 handoff.
5. Re-fetch latest main, revalidate, then mark worker-11 COMPLETE only when unresolved worker-11 work is empty.

FILES_CURRENTLY_RELEVANT:
- minecraft_mod_ai/diagnostics.py
- tools/full_project_audit.py
- .github/workflows/ci.yml
- .github/workflows/full-debug-gate.yml
- .github/workflows/observability-regression.yml
- minecraft_mod_ai/complete_orchestrator.py
- minecraft_mod_ai/java_lsp.py
- minecraft_mod_ai/cli.py
- tests/test_diagnostics.py

KNOWN_CROSS_ROLE_DEPENDENCIES:
- Shared `complete_orchestrator.py` and CLI/top-level presentation belong to worker 12; handoff created for minimal integration.
- Validation/JDT semantics are also consumed by worker 10; worker 11 changes must preserve validator contracts while making failure category visible.

UNRESOLVED:
- New diagnostics contract is not yet wired into Full Debug audit output.
- Shared JDT/CLI boundaries still need worker-12 integration so programming errors cannot be mislabeled as UNAVAILABLE.
- GitHub Actions verification for this checkpoint is pending.
