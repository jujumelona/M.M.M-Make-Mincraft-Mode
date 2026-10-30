Small coder work ownership repair
=================================

Scope and evidence
------------------

The supplied production log runs main d37621834b41. The failure is
INCREMENTAL_JAVA_NO_SEMANTIC_PROGRESS after 96 unproductive choices.
The trace shows a zero-input helper with only local boolean copies and
repeated lookup pages, followed by repeated calls to a sibling registrar.
The saved feature work order was absent from the operation prompt. Logical
world-state names also projected the existing storage API into an empty list.

Repair
------

* Preserve the immutable feature work order, its acceptance contract and all
  additional pages. Keep the current behavior's aligned intent/effect visible.
* Page exact materialized state declarations even when descriptive world names
  do not match storage IDs. Separate declaration discovery from canonical
  read/write permission. Never infer aliases or redirect writes to other state.
* Use compact stable host page tokens that fit the actual lookup schema.
* Reserve lifecycle registrars and initializer entrypoints for the host,
  including interface upcasts, subclasses and method references.
* Acquire actual observation inputs before native-world implementation when
  an indexed observation API exists. Preserve seeded source-owned UI observers.
* Do not finish world work with local copies alone. Account for calls executed
  in initializer/assignment/condition expressions and ignore deferred closures.
* Preserve nested rejection returns and supporting ABI hooks, including an
  intentionally empty hook alongside a working hook.

This changes host work scheduling and evidence delivery. It contains no
space-combat, colonization, trading, or other request-specific implementation
fallback. Compiler, state provenance, native mutation and runtime acceptance
remain mandatory and independent of this necessary structural progress check.

Verification
------------

The new regressions were observed failing before their fixes. Validation
includes actual Java compilation/execution of generated listeners and a
supporting adapter with one implemented result and one empty void hook.
The independent review's four important findings received reproductions and
fixes; declaration-discovery permission metadata was also clarified.

Affected authoring/repair/source-UI/state tests: 342 passed, 1 skipped.
The skipped test requires MMM_ACTION_COMPILE_CLASSPATH for the actual SDK25
classpath. Adjacent source generation, prerequisite and feature integration
checks: 61 passed. An additional first-pass/pipeline selection: 47 passed
(partly overlapping the other selections).

Ruff's repository CI selection (F,E7,E9), git diff --check, and the repository
code-quality regression audit pass. No new complexity, import-cycle, duplicate
body or serial expensive-loop regression is reported.

Verification limits
-------------------

A live MiMo-V2.6-Distill-Qwen-9B sampling run and a Minecraft 26.2 gameplay
playthrough were not executed. Scripted schema replay does not establish
that every future model response will implement an arbitrary feature.
The integrity probe could not download Gradle 8.7: Java reported
java.net.SocketException: Network is unreachable. No Minecraft integration
success is claimed from that probe.

The unmodified d37621834b41 checkout also reproduces
``test_construction_file_author.py::test_model_to_real_javac_to_diagnostic_repair_to_checkpoint``
with StopIteration in this environment. Full-suite failures are examined
separately from the affected incremental authoring regressions.

The first full-suite parallel run stopped in test infrastructure: setuptools
84.1.0's build_py._Warning could not be recovered by the pytest-xdist worker,
followed by an xdist worker-controller KeyError. That run is not a full-suite
pass. The compatibility-environment rerun also stopped on this warning transport:
14 failed, 5139 passed, 8 skipped. It is incomplete and is not a full-suite pass.
Its assertion failures concern missing JDK compiler launchers, unsupported
Java --release 21 with JDK17, and missing socksio for the environment proxy.
All 13 ordinary test failures were rerun against unmodified main
 d37621834b41 under the identical environment: all 13 fail there too.
The packaging-worker crash is test infrastructure, separate from those 13
assertion failures. No full-suite or live-model certification is claimed.

Baseline-reproduced failed tests
-------------------------------

* ``tests/test_construction_file_author.py::test_model_to_real_javac_to_diagnostic_repair_to_checkpoint``
* ``tests/test_construction_project_compile.py::test_sibling_source_included_and_repository_unchanged``
* ``tests/test_construction_project_compile.py::test_real_compiler_diagnostic_returns_sibling_context``
* ``tests/test_construction_project_compile.py::test_same_package_siblings_discovered_without_source_context``
* ``tests/test_construction_project_compile.py::test_imported_java_dependency_discovered``
* ``tests/test_construction_project_compile.py::test_old_generated_target_does_not_shadow_candidate``
* ``tests/test_construction_project_compile.py::test_unrelated_broken_sibling_not_compiled``
* ``tests/test_construction_project_compile.py::test_client_source_can_link_unbuilt_main_project_source``
* ``tests/test_construction_project_compile.py::test_wildcard_package_import_resolves_real_java_source``
* ``tests/test_construction_project_compile.py::test_comments_must_not_import_broken_java_sources``
* ``tests/test_planner_stream_liveness.py::test_planner_role_sse_transport_finishes_healthy_8192_budget_stream_after_300_seconds``
* ``tests/test_managed_content_gametest.py::test_emitted_server_gametest_java_compiles_with_real_javac``
* ``tests/test_llama_server_streaming.py::test_persistent_client_has_bounded_idle_read_timeout``

Infrastructure failure: ``test_distribution_package_contract.py::test_wheel_contains_all_python_packages_and_runtime_resources`` lost its worker during warning serialization.
