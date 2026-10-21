# Host callback compiler implementation plan

Goal: remove callback syntax/control flow and unresolved state storage from the small coder's responsibilities.
Spec: docs/host-callback-compiler-design.rst
Execution: native in the existing clean main checkout, as authorized by the user.

Global constraints: preserve original behavior requirements; inspected Fabric ABI is authoritative; no return-only effects; no hidden legacy fallback; no prompt-specific code; commit and push each completed change.

1. Add expression/operation lowering in `host_callback_body.py` and result selection in `host_callback_dispatch.py`. First add failing parser and compiled runtime tests; then switch all verified callbacks in `first_pass_java_source.py` to host-owned predicate/effects/result dispatch. Update transport contract tests to exercise the new interface. Run the related suite, commit, push.
2. Add upstream state binding in `construction_behavior_state.py`. First test missing-state declaration, world observations, aliases, preserved state types, repeat runs and invalid bindings. Integrate before `compile_production_state_section` in `authored_production.py`, preserving original source/provenance. Run state/planner/production suites, commit, push.
3. Run project checks, inspect the exact production completion payloads against the new compiler boundary, and independently review the implementation. Fix verified regressions with tests, record external-model/runtime limits, commit and push remaining verified fixes.

Review focus: conditional no-op callbacks; result fields whose concrete type is a subtype; returns hidden in lambdas; state aliases that change ownership; state reconciliation repeated after checkpoint restore.
