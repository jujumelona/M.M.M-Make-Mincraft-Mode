WORKER: 11
ROLE: Errors + Observability + Diagnostics + CI
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: 4e8713b495f056449f3be71c26cb65ee84ee2d71

COMPLETED_BASELINE:
- Canonical failure taxonomy, causal fingerprinting, retry/final status, compact rendering, traceback retention, and sanitization.
- Full Debug root-cause wrapper with raw artifact preservation and compact console output.
- Explicit PASS/WARN/SKIP/FAIL artifact semantics and fail-closed unknown-status handling.
- Compact pytest diagnostic runner with raw log/JUnit artifacts and causal deduplication.
- Dedicated Observability Regression workflow and CI integration.
- Removed the unreferenced duplicate `annotate_pytest_failures.py` path.
- Worker12 handoff exists for the shared JDT/CLI broad-exception boundary.

CURRENT_HARDENING_PASS:
- Reopened after the user required clean-code/dead-code/bottleneck/optimization verification beyond the first completion boundary.
- Hardened `minecraft_mod_ai/diagnostics.py`: bounded root rendering, bounded fallback rendering, newline/oversize compaction, explicit deduplication keys, and latest-attempt terminal state.
- Added regression coverage for bounded rendering, explicit retry deduplication, sanitization, and invalid render limits.
- Hardened `tools/root_cause_audit_wrapper.py`: remove stale report before each run, stream raw child output to artifact, validate report/check/summary/index consistency, atomically rewrite normalized report, and reject process/report exit disagreement.
- Hardened `tools/pytest_diagnostics.py`: remove stale JUnit before each run, stream pytest output to artifact, iterparse JUnit, reject missing/empty/contradictory JUnit evidence, reject output-path collisions, and bound affected-test rendering.
- Added failure-injection tests for stale artifacts, process/report contradictions, malformed evidence, namespace-aware JUnit parsing, and zero-evidence success.

COMMITS_THIS_PASS:
- be1c0f25bbf74d81eb16a7d0029764c589673833 refactor: harden compact diagnostics
- be2a5a6bab0c99a681022b8bf519ea5096499538 test: cover bounded diagnostic rendering
- 9732bb8fa3d28f0acb189d5dcb6894c6db91242d fix: fail closed on stale audit evidence
- baad4afa6d09d3e238427d0174d43f211710d598 fix: make pytest diagnostics fail closed
- 496e74f304b2b2f1ddd2e2247aa66142b1c8c56e test: cover stale audit and exit mismatch
- 7d85ddb94e92ea1916004211ae232d3e2dc260dc test: cover fail-closed pytest evidence

ROOT_CAUSES_STILL_BEING CLOSED:
- `tools/full_project_audit.py` still captures command stdout through PIPE and duplicates full text into `LOGS`, causing avoidable memory amplification on large pytest/build output.
- `tools/full_project_audit.py::Check.passed` still reports WARN/SKIP as passed before wrapper normalization, leaving two semantic authorities.
- Worker11 wrappers still need explicit directory-creation failure handling so filesystem failure never falls through as an unstructured traceback.
- `remaining-tests` CI sharding still uses a pre-filter positional index, so dedicated-test additions/removals can perturb unrelated shard assignment.

IN_PROGRESS:
- Eliminate raw filesystem-traceback surfaces in diagnostic wrappers.
- Remove Full Debug command-output memory duplication while preserving complete raw evidence and bounded failure detail.
- Make Full Debug PASS semantics single-source and internally consistent.
- Stabilize remaining-tests shard selection and regression-test the selection rule.
- Re-run targeted tests, Observability Regression, ancestry checks, and latest-main verification before returning to COMPLETE.

KNOWN_CROSS_ROLE_DEPENDENCY:
- `minecraft_mod_ai/complete_orchestrator.py` still contains the Worker12-owned JDT `except Exception -> UNAVAILABLE` boundary. It remains documented in `docs/parallel_handoff/worker-11.md`; Worker11 will not overwrite shared orchestrator ownership.

UNRESOLVED:
- The four Worker11 hardening items listed above are intentionally open until code + regression evidence are green.
