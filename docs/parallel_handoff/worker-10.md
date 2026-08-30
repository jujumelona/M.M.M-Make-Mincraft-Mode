REQUESTING_WORKER: 10
TARGET_OWNER: 12

WHY:
- Validation execution currently has two authorities for the same JDT diagnostic interpretation.
- `validation_execution_contract.py` still defines a stale list-shaped `_diagnostic_errors` implementation.
- Runtime bootstrap then installs `validation_diagnostic_contract.diagnostic_errors` over that symbol to understand the real URI -> diagnostics JDT-LS receipt shape.
- Worker 10 hardened the installed adapter so JDT `UNAVAILABLE`, explicit errors, and malformed receipts fail closed, but deleting/reordering the bootstrap integration belongs to worker 12 shared-core ownership.

REQUIRED_INTERFACE_CHANGE:
- Consolidate JDT diagnostic interpretation to one explicit authority while preserving worker-10 fail-closed semantics.
- Preferred end state: the real URI->diagnostics flattening plus JDT-availability validation lives directly in the validation execution owner; the extra runtime monkey-patch is removed.
- If `validation_diagnostic_contract.py` remains the sole owner instead, remove the stale duplicate implementation and make that ownership explicit rather than relying on shadowing order.
- Preserve severity policy: LSP severity 1 blocks; severity 2 warnings remain visible but do not by themselves defer Gradle.

FILES_INVOLVED:
- minecraft_mod_ai/runtime_bootstrap.py
- minecraft_mod_ai/validation_execution_contract.py
- minecraft_mod_ai/validation_diagnostic_contract.py

TEST_THAT_CURRENTLY_FAILS:
- No functional test is intentionally left failing after worker 10.
- Preserve `tests/test_worker10_validation_fail_closed.py` and existing validation execution tests while consolidating the authority.
