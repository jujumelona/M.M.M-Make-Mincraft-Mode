WORKER: 12
STATUS: COMPLETE
BASE_SHA: 5abfbd5492eca9fe941225146c2f7de6969b40f8
COMMIT_SHA: 80a82b9c4ebe335521d2c563f4fb6af499b331ca
ROOT_CAUSES_FIXED:
- RAG receipt/evidence trust could be composed across sibling results; receipt fallback evidence is scoped to the same result subtree.
- JDT diagnostic normalization accepted malformed nested diagnostic groups; malformed payloads fail closed.
- Agentic repair verification duplicated JDT invocation compatibility and retried programming TypeError; it uses the shared validation diagnostic execution boundary.
- Agentic repair verification imported diagnostic_errors from the retired repair diagnostic adapter; execution and interpretation share validation_diagnostic_contract as the single authority.
- Agentic candidate generation caught BaseException in both the base repair-search wrapper and the parallel shared-integration wrapper, so KeyboardInterrupt/SystemExit could be swallowed and retried; both layers now catch Exception only and process-control exceptions propagate immediately.
- The parallel repair-search integration assumed every wrapped test/double exposed router before it was needed; missing router now degrades through the existing no-config path instead of failing before the underlying cancellation/error contract can run.
- Trajectory fallback deduplication only inspected the most recent 512 JSONL records, allowing an older identical trajectory_id to be appended again; fallback dedupe now checks the full durable log before append.
- Agentic repair-experience reads scanned the full JSONL file even though only the recent window is ranked; recent-row reads are bounded from the file tail and duplicate writes no longer build an unbounded in-memory id set.
- Trajectory state/cache/index leaf paths could follow symlinks; component and leaf containment plus no-follow JSONL append protections are enforced.
- Colab local-to-remote profile switching could leave a managed llama-server and inactive-profile state alive; inactive profile state is quiesced during profile changes.
- Colab local-profile fingerprints and receipts consumed remote-only URL/model inputs, so malformed or stale remote configuration could perturb/fail a local run; local profiles now ignore remote-only fingerprint inputs and persist an empty remote receipt.
- Temporary Worker-12 restart-audit script, trigger, and workflow were removed after the permanent source patch passed focused tests, rebased latest main, re-ran the tests, and pushed successfully.
FILES_CHANGED:
- minecraft_mod_ai/agent_security_contract.py
- minecraft_mod_ai/validation_diagnostic_contract.py
- minecraft_mod_ai/agentic_optimization_contract.py
- minecraft_mod_ai/agentic_search_efficiency_contract.py
- minecraft_mod_ai/trajectory_memory.py
- tools/colab_runtime_setup.py
- tests/test_agent_security_contract.py
- tests/test_worker12_shared_core.py
- docs/parallel_progress/worker-12.md
TESTS_ADDED_OR_VERIFIED:
- RAG sibling receipt/evidence isolation and terminal-receipt non-rescue regression tests.
- Nested malformed JDT diagnostics fail-closed regression tests.
- Agentic verifier programming-TypeError single-call/non-retry coverage against the canonical implementation.
- Agentic candidate KeyboardInterrupt propagation/non-retry coverage through the installed shared search wrappers.
- Trajectory state/cache leaf-symlink rejection and full-log fallback dedupe coverage.
- Bounded recent repair-experience JSONL reader coverage.
- Colab profile-switch stale managed-server shutdown coverage.
- Colab local-profile fingerprint/receipt independence from remote-only malformed or stale configuration.
- Worker-12 focused regression suite passed before commit and passed again after rebasing latest main immediately before push.
- Worker-12 diff/compile/ruff/Colab-notebook static gates passed.
CROSS_OWNER_FILES_CHANGED:
- .github/workflows/worker12-restart-audit.yml deleted as cleanup of temporary Worker-12 one-shot scaffolding only; no Worker-11 CI policy/workflow behavior was added or modified.
HANDOFF_CREATED: none
KNOWN_REMAINING: none in Worker-12 scope
PUSH_VERIFIED_ON_ORIGIN_MAIN: true
