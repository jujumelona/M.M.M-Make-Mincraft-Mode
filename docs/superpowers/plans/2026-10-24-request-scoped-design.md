# Request-scoped design implementation plan

> Execute inline with the executing-plans and test-driven-development skills.

**Goal:** Preserve each requested behavior through bounded design authoring.

**Architecture:** Schedule one existing worksheet DAG per source requirement and
merge validated records. Carry shared state declarations and checkpoint completed
units. Keep the current UI source-effect gate.

**Tech stack:** Python, JSON Schema, existing typed state DSL and planner checkpoints.

**Spec:** `docs/superpowers/specs/2026-10-24-request-scoped-design.md`

## Global constraints

- Work on main, as previously requested; no new branch or worktree.
- No prompt-specific space-mod logic or weakened acceptance checks.
- Decoder limits stay bounded; only host accumulation may grow.
- Do not claim live model or gameplay success from fixture tests.

## Review focus

- More than four features must retain distinct behaviors and state updates.
- Shared variable reuse must retain its type/default/owner contract.
- A conflicting later variable must not overwrite an accepted earlier definition.
- An interrupted run must not reuse a feature under different shared-state input.
- Legacy callers and saved plans must retain their existing interface/validation.

## Task 1: Scope and merge design units

Files: `complete_planner.py`, `authored_structured_design.py`, new
`request_structured_design.py`, and `tests/test_request_scoped_design.py`.

- [x] Write and run regressions for whole-request collapse and accumulated records.
- [x] Route source requirements into independent existing worksheet DAGs.
- [x] Merge records and propagate validated shared state without overwriting it.
- [x] Verify existing worksheet, state, UI and request-first tests.

## Task 2: Resume accepted units and verify

Files: `planner_checkpoint.py`, request-scoped authoring, regression tests.

- [x] Write and run an interrupted-authoring resume regression.
- [x] Cache accepted units under requirement and shared-state input identity.
- [x] Run focused tests, the project suite, lint, compilation and architecture audit; record remaining verification failures below.
- [x] Obtain a fresh code review and resolve material findings; prepare the verified change for main.

## Verified work and review

- Whole-request collapse reproduced before implementation: six source actions
  became one entry condition. Canonical accumulation also rejected six updates.
- Focused verification: 112 passed, 1 skipped before enabling the JDK compiler.
- Actual Java rule/state execution regression: 55 passed with the JDK on PATH.
- Original log replay still rejects the unrelated purchase UI.
- Fresh review found a cross-requirement trigger collision. Three reproductions
  failed before source provenance was retained, then passed with source-aware
  UI identity, input derivation, preflight and lowering. Known runtime ownership
  is now determined from accepted host provenance.
- Ruling: continue on main in this fresh checkout, following the existing user
  branch policy; no additional branch or worktree.
- Ruling: certify orchestration and deterministic lowering with fixtures and
  actual Java execution only. Live MiMo/Colab and full gameplay are not certified.

## Final verification (2026-10-24)

- Focused request/state/UI/runtime-ownership suite with JDK: **113 passed**.
- Actual Java host-rule/state execution regressions: **55 passed**.
- Full project suite: **4565 passed, 5 failed, 8 skipped** in 863 seconds.
- Ruff critical errors, Python compilation, whitespace and architecture regression
  checks pass. Both new regression files are included in the 32-group CI matrix.

The five full-suite failures were inspected individually:

1. `test_debug_fixture_runs_real_build_and_packaging_without_live_model` and
   `test_debug_fixture_releases_when_only_jdt_infrastructure_is_unavailable` in
   `tests/test_deterministic_debug_full_pipeline_e2e.py`: the JDT owner cannot
   download Maven/Gradle dependencies in this restricted environment. The concrete
   failure is `Network is unreachable`, before code generation.
2. `test_bridge_schema_binding_isolated_by_server_scope_and_owner_reused` in
   `tests/test_external_mcp_role_scope.py`: also reproduced on the unchanged
   `9c5bd9a` source by running the bound-unavailable, bridge-safety, registry-YAML
   and role-scope test files together (11 passed, 1 failed). It passes in isolation.
3. `test_persistent_client_has_bounded_idle_read_timeout` in
   `tests/test_llama_server_streaming.py` and
   `test_planner_role_sse_transport_finishes_healthy_8192_budget_stream_after_300_seconds`
   in `tests/test_planner_stream_liveness.py`: missing optional SOCKS support in
   the host proxy environment. After installing that environment dependency,
   these exact two tests passed together. No repository change was needed.

The remaining network-dependent integration tests and pre-existing MCP order
failure are not reported as passing. Live MiMo planning and a complete Minecraft
playthrough were not run. Re-sync Colab and author a new plan; the old incomplete
purchase/launch design remains correctly rejected by the production effect gate.
