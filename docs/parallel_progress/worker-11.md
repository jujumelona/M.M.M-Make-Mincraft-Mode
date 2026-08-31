WORKER: 11
ROLE: Errors + Observability + Diagnostics + CI
STATUS: COMPLETE
LAST_UPDATED_MAIN_SHA: 9a95bbaa74e73e212ee050cee8508ecf0055b0ed

COMPLETED_BASELINE:
- Canonical failure taxonomy, causal fingerprinting, retry/final status, compact rendering, traceback retention, and sanitization.
- Full Debug root-cause wrapper with raw artifact preservation and compact console output.
- Explicit PASS/WARN/SKIP/FAIL artifact semantics and fail-closed unknown-status handling.
- Compact pytest diagnostic runner with raw log/JUnit artifacts and causal deduplication.
- Dedicated Observability Regression workflow and CI integration.
- Removed the unreferenced duplicate `annotate_pytest_failures.py` path.
- Worker12 handoff exists for the shared JDT/CLI broad-exception boundary.

FINAL_HARDENING_COMPLETED:
- `minecraft_mod_ai/diagnostics.py`: bounded root/fallback rendering, oversize/newline compaction, explicit deduplication keys, latest-attempt terminal state, traceback retained only as debug evidence, and sanitizer support at the canonical exception boundary.
- `tools/full_project_audit.py`: subprocess output is drained through bounded temporary-file streaming rather than an unbounded PIPE/full-memory copy; only a bounded redacted tail is retained in check detail while full redacted evidence is kept in the audit log.
- `tools/full_project_audit.py::Check.passed`: PASS is the only successful state; WARN/SKIP are explicitly non-blocking rather than semantically passed.
- `tools/root_cause_audit_wrapper.py`: stale report/log removal, atomic normalized report rewrite, report/check/summary/index consistency validation, process/report exit agreement, explicit filesystem-failure handling, a 45-minute hard audit timeout, bounded temporary raw-output capture, and redacted persisted runner output.
- Full Debug console boundary: failed report details, category/name fields, internal exception messages, fallback text, and runner events are passed through the same streaming secret sanitizer before rendering.
- `tools/pytest_diagnostics.py`: stale output removal, bounded raw pytest capture, positive timeout enforcement, namespace-aware streaming JUnit parsing, missing/empty/contradictory evidence rejection, bounded affected-test rendering, and sanitized console/raw evidence.
- JUnit redaction is context-safe: persisted XML uses `&lt;redacted&gt;`, remains valid XML, and parses back to the logical `<redacted>` diagnostic marker.
- `tools/audit_stream_redactor.py`: one streaming redaction engine is used across audit/pytest boundaries; supports exact environment secrets, quoted JSON values, structured values, header-like values, arbitrary chunk boundaries, caller-provided context-safe replacement markers, and fully redacts unquoted multiword sensitive values through the next structural delimiter/line boundary.
- `tools/ci_test_shard.py`: remaining-test shard ownership is stable SHA-256 based rather than pre-filter positional indexing, preventing unrelated dedicated-test changes from reshuffling shards.
- Observability gate now compile-checks, targeted-ruffs, and failure-injects diagnostics, streaming redaction, Full Debug wrapper, pytest diagnostics, filesystem failures, full-project observability, and stable sharding.

FAILURE-INJECTION / REGRESSION COVERAGE:
- repeated identical failures collapse to one causal root with an ATTEMPTS count;
- INTERNAL programming failures retain debug traceback but compact user rendering excludes traceback spam;
- sanitizer removes exact and labelled secrets from message and traceback paths;
- WARN/SKIP cannot be mislabeled PASS and unknown status fails closed;
- stale Full Debug report/JUnit/log evidence cannot be reused;
- audit process/report contradictions fail closed;
- missing/empty/contradictory JUnit evidence fails closed;
- Full Debug timeout is canonical TRANSIENT and preserves only redacted output;
- pytest non-positive timeout is rejected before launch;
- raw pytest log and JUnit artifacts cannot retain injected exact/labelled secrets;
- XML-safe redaction remains valid through JUnit parsing;
- custom replacement markers survive chunk widths down to one character;
- unquoted multiword `token/password/secret/api-key` values are fully redacted instead of leaking the tail after whitespace;
- stable CI shard assignment is deterministic under input reordering and dedicated-test filtering changes.

COMMITS_FROM_FINAL_HARDENING:
- e1fc8b48276128a56bb0003d669d7ff89ba2cae4 `fix: bound and redact full debug runner`
- 0d71da4b26804928b35c692e27a07aaace9b288b `test: cover full debug timeout and redaction`
- ef4faf551c9ce8e22e7028a6217731f2af143429 `fix: support context-safe redaction markers`
- 541736c200553003bc44e163d7e1465ee388195c `fix: preserve valid redacted JUnit XML`
- 1d517492ae3b8233cc299bf37255b05483e64b0b `test: enforce valid redacted JUnit artifacts`
- f1db76d4125264150b8efc06fc187fe870149dfd `fix: sanitize full debug console boundary`
- 072d41de0a8c05eb006f29126c4a178a3f5b842a `test: enforce full debug console redaction`
- b647424966988a072bbd38b9218468af4fb98903 `test: cover context-safe redaction markers`
- 0848cd7a5cd88f8ce0799c067e451cd165ac6c91 `ci: gate context-safe redaction regression`
- e913fc24c5a65733e01a4c70bf2b0e578849b7b8 `fix: redact unquoted multiword secret values`
- 9a95bbaa74e73e212ee050cee8508ecf0055b0ed `test: cover unquoted multiword secret values`

FINAL_VALIDATION:
- Origin main at validation time: `9a95bbaa74e73e212ee050cee8508ecf0055b0ed`.
- Observability Regression run `33359027134` for that exact SHA: COMPLETED / SUCCESS.
- The dedicated gate's compile/lint stage and all Worker11 failure-injection suites completed successfully.
- Main-only workflow was preserved; no Worker11 branch or force push was used.
- General CI run `33359027142` fails before the test fan-out at the repository-wide runtime-mutation budget (`behavioral=427`, reviewed budget `413`). That gate covers runtime code outside Worker11's tools/tests surface and is not a Worker11 observability regression; it remains a repository-level integration item for the owning worker/final integrator.

KNOWN_CROSS_ROLE_DEPENDENCY:
- `minecraft_mod_ai/complete_orchestrator.py` contains the Worker12-owned JDT `except Exception -> UNAVAILABLE` boundary. It is already documented in `docs/parallel_handoff/worker-11.md`; Worker11 intentionally did not overwrite shared-orchestrator ownership. Worker12/Worker13 must ensure programming exceptions cannot be mislabeled dependency unavailability at final integration.

UNRESOLVED:
- none within Worker11 ownership.

HANDOFF_TO_WORKER_13:
- Worker11 observability/diagnostics/CI area is ready for integration.
- Preserve the canonical failure taxonomy/fingerprint contract and the single StreamingRedactor behavior when resolving cross-worker conflicts.
- Treat the repository-wide runtime-mutation budget failure and the Worker12 JDT broad-catch handoff as cross-role integration checks, not reasons to reopen Worker11's completed observability implementation unless a final integration test exposes a regression in this owned surface.
