# Host callback compiler implementation plan

Goal: remove callback syntax/control flow and unresolved state storage from the small coder's responsibilities.
Spec: docs/host-callback-compiler-design.rst
Execution: native in the existing clean main checkout, as authorized by the user.

Global constraints: preserve original behavior requirements; inspected Fabric ABI is authoritative; no return-only effects; no hidden legacy fallback; no prompt-specific code; commit and push each completed change.

1. Add expression/operation lowering in `host_callback_body.py` and result selection in `host_callback_dispatch.py`. First add failing parser and compiled runtime tests; then switch all verified callbacks in `first_pass_java_source.py` to host-owned predicate/effects/result dispatch. Update transport contract tests to exercise the new interface. Run the related suite, commit, push.
2. Add upstream state binding in `construction_behavior_state.py`. First test missing-state declaration, world observations, aliases, preserved state types, repeat runs and invalid bindings. Integrate before `compile_production_state_section` in `authored_production.py`, preserving original source/provenance. Run state/planner/production suites, commit, push.
3. Run project checks, inspect the exact production completion payloads against the new compiler boundary, and independently review the implementation. Fix verified regressions with tests, record external-model/runtime limits, commit and push remaining verified fixes.

Review focus: conditional no-op callbacks; result fields whose concrete type is a subtype; returns hidden in lambdas; state aliases that change ownership; state reconciliation repeated after checkpoint restore.

Completed validation
--------------------

Task 1: complete. New operation/value compiler replaces Java statement generation on verified callbacks. Real javac/java probes exercise void and non-void dispatch.
Task 2: complete. State bindings resolve before host state/code jobs; canonical declarations, world observations, provenance and per-concept checkpoints are covered.
Task 3: complete. Compiler repair keeps callback ownership. Independent review found and reproduced three Important defects: statement-bearing switch expressions, discarded prerequisite API declarations, and handled results on empty effect execution paths. Regression tests failed before fixes; expressions now reject hidden statements, prerequisites coexist with classpath APIs, and host execution receipts cover false branches and empty loops. Local value assignments cannot satisfy an effect obligation or overwrite the receipt.

The external legacy statement-author changes made concurrently on main were preserved. Callback production now uses the new compiler. No extra branches were created.

External-model/Minecraft-runtime certification is not claimed. The broad pytest run encountered Gradle/JDT dependency-download failures under restricted networking; the deterministic full-pipeline tests fail before model/gameplay work for that reason. Related host/compiler/state tests and compiled runtime probes run locally without that dependency bootstrap.

Final focused verification: 135 tests passed across the callback compiler, real javac/java probes, compiler repair, state binding/checkpoints, planning, state runtime, first-pass source generation and feature job compilation. Ruff passed for all new modules and regression tests.

Known blocked tests (isolated replay, before model/gameplay work):
- tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_runs_real_build_and_packaging_without_live_model
- tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_releases_when_only_jdt_infrastructure_is_unavailable
Both encounter unavailable Maven/Gradle dependencies in the JDT bootstrap under restricted networking. The broader run was not green. tests/test_external_mcp_role_scope.py passed when replayed independently.

State-store reconciliation regression (2026-10-21)
-----------------------------------------------

The next Colab run reached state binding but failed because adding a canonical variable left the previously derived state-store schema version and migration destinations stale. The state-binding transformation must return the new state authority and its derived storage contract atomically. Validate the old contract against the old authority before regenerating it; preserve namespace, transfer policy, migration operations, other platform modules, and original design text. Do not weaken the production drift gate.

Follow-up checks: reproduce through reconciliation and production lowering; materialize state/persistence/network sources from the same authority; reject pre-existing drift; verify aliases/world observations leave the typed plan untouched; replay captured 67-reference bindings and checkpoint restore. Add failing regressions first, implement the host-only rebind, run related checks, review, commit and push main.

Follow-up outcome: the captured 67-reference inputs reproduced the exact schema-version drift before the fix and now reach production lowering. The host validates the untouched old state-store contract, then changes only its schema version and canonical migration destinations in the same returned plan as the new scalar. Namespace, transfer settings, other modules and original source remain unchanged. Invalid settings stay invalid.

Thirteen new regressions and the related state/Typed PlanIR suites pass (101 tests). Generated state, persistence and network sources use the same updated variables. Independent review identified a missing-migrations field that could otherwise be silently filled; its regression failed before adding old platform-module validation and passes afterward. The final review found no remaining Critical or Important defects. Ruff and diff whitespace checks pass. This replay does not certify the external model's semantic choices or Minecraft gameplay.
