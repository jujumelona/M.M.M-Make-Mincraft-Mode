# Incremental Java generation progress

> For agentic workers: execute this repair inline with test-driven development.

**Goal:** stop declaration-only growth in the logged `m1` work item and produce implemented, callable helpers.

**Evidence:** commit `01bae7a826c6`; the supplied log contains 1,250 consecutive coder responses on the same `m1`, cycling through `method`, `parameters_count`, and `kind`. `author_block` defers all declared bodies until its parent finishes, and `new_method` always allocates a new identity.

**Architecture:** finish newly created child work before returning to the parent. Reuse declarations with the same owner, behavior, purpose, and callable contract. Publish the exact declaration result and its implementation status to the next operation request. After reuse, omit another declaration of that operation at the same position until executable work progresses.

**Constraints:** retain host-owned Java syntax, expression-only source generation, exact callback contracts, compiler repair, and distinct-purpose helpers. Do not infer success from model response completion or use a total operation cap.

## Task 1: implement declaration progress

Files: `incremental_java_author.py`, `incremental_java_context.py`, new `incremental_java_progress.py`, `incremental_java_declarations.py`, and `tests/test_incremental_java_progress.py`.

- [x] Reproduce deferred helper implementation and repeated declaration growth with failing tests.
- [x] Implement child scheduling, identity reuse, and explicit operation feedback.
- [x] Verify separate helpers remain distinct and nested authoring restores the parent scope and draft stack.
- [x] Compile and execute a generated callback with real Java; expected claim count is two.
- [x] Run existing incremental Java and direct-feature integration regressions.

## Task 2: verify and deliver

- [x] Review cancellation, compiler-created helpers, active self-delegation, and more than one behavior registrar.
- [x] Run the full test suite and required Python compile/lint checks; document environmental failures separately.
- [ ] Commit and make the repair available in a reviewable branch/PR, with the actual validation limits stated.

## Verification record

- Related source generation, callback, state, and compiler-repair tests: **190 passed**, zero failures/errors/skips, including real Java compilation and callback execution.
- Independent review found active self-delegation, type-layout operation suppression, and erased parameter roles; all three received failing reproductions and passing fixes. Callback feedback also distinguishes an already-connected listener from an ordinary helper.
- Critical Ruff, Python compileall, dead-code audit, and architecture regression audit pass.
- Full-suite run: 4,993 tests, 19 failures, 45 skips. After correcting the local compiler command path, adapting two scripted production fixtures to immediate listener implementation, and rerunning on stable sources, 16 of the 17 rerun failures pass; the remaining GameTest compilation also passes when using the installed JDK 21. The additional callback-feedback case is included in the 190 related passes.
- `test_emitted_server_gametest_java_compiles_with_real_javac` passes with the installed Temurin JDK 21 and its required `--release 21`.
- `test_debug_fixture_runs_real_build_and_packaging_without_live_model` and `test_debug_fixture_releases_when_only_jdt_infrastructure_is_unavailable` cannot resolve Maven/Gradle dependencies because this environment blocks their network access.
- `test_bridge_schema_binding_isolated_by_server_scope_and_owner_reused` failed in the full suite and passed in isolated reruns. No MCP code was changed; its apparent ordering sensitivity is not claimed fixed.
- No live MiMo9B or Minecraft gameplay success is claimed. Exact declaration reuse does not prove termination for a model that continually requests distinct new work.
