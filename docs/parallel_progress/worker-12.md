WORKER: 12
STATUS: COMPLETE
BASE_SHA: 210185f6e939148bf9e47050a80a8b6a5ae4b4ec
COMMIT_SHA: a6190e82e4a65583f34a5bbdb028ebd50d201407
ROOT_CAUSES_FIXED:
- RAG receipt/evidence trust could be composed across sibling results; receipt fallback evidence is now scoped to the same result subtree.
- JDT diagnostic normalization accepted malformed nested diagnostic groups; malformed payloads now fail closed.
- Agentic repair verification duplicated JDT invocation compatibility and retried programming TypeError; it now uses the shared validation diagnostic execution boundary.
- Agentic repair verification still imported diagnostic_errors from the retired repair diagnostic adapter; execution and interpretation now share validation_diagnostic_contract as the single authority.
- Trajectory state/cache/index leaf paths could follow symlinks; component and leaf containment plus no-follow JSONL append protections are enforced.
- Colab local-to-remote profile switching could leave a managed llama-server and inactive-profile state alive; inactive profile state is now quiesced during profile changes.
- Legacy repair-experience.jsonl hardening was identified as obsolete duplicate state because unified trajectory v3 is the authoritative repair-memory store; the duplicate hardening path was removed rather than revived.
- Temporary Worker-12 patch scripts, workflow, trigger, and checkpoint markers were removed after permanent source verification.
FILES_CHANGED:
- minecraft_mod_ai/agent_security_contract.py
- minecraft_mod_ai/validation_diagnostic_contract.py
- minecraft_mod_ai/agentic_optimization_contract.py
- minecraft_mod_ai/trajectory_memory.py
- tools/colab_runtime_setup.py
- tests/test_agent_security_contract.py
- tests/test_worker12_shared_core.py
- docs/parallel_progress/worker-12.md
TESTS_ADDED:
- RAG sibling receipt/evidence isolation and terminal-receipt non-rescue regression tests.
- Nested malformed JDT diagnostics fail-closed regression tests.
- Agentic verifier programming-TypeError single-call/non-retry regression coverage against the canonical implementation.
- Colab profile-switch stale managed-server shutdown regression coverage.
- Trajectory state/cache leaf-symlink rejection regression coverage.
- Worker-12 dedicated regression suite passed before source commit and passed again after rebasing the latest main immediately before push.
- Worker-12 diff/compile/ruff/vulture/Colab-notebook static gates passed.
CROSS_OWNER_FILES_CHANGED: none; shared integration-glue files only within Worker-12 scope
HANDOFF_CREATED: none
KNOWN_REMAINING: none in Worker-12 scope
PUSH_VERIFIED_ON_ORIGIN_MAIN: true
