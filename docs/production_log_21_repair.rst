Production log 21: construct an original UI occurrence before its listener
=======================================================================

Observed failure
----------------

The attached run used ``7e356d79a5047f83a2efb4088fb5f12a4a6878ed``.
Its Gradle classpath preflight passed. Generation then failed in
``generate-content-00000002`` for ``behavior_51abfccc946fe0413520``:
``SOURCE_HOOK_PRODUCER_REQUIRED / no_relevant_inspected_source_site``.

The immutable behavior trigger was: "Player reaches a planet or space location
and initiates colonization." The model correctly declined an arrival callback
as the complete source of that interaction. The source-owned route then
required a pre-existing custom method; it had no job that built the missing
input source before binding the listener.

Implementation
--------------

* Preserve real inspected Fabric callbacks and precise existing Mixin sites.
* For an explicitly requested player selection/confirmation/initiation, build a
  concrete Typed Action form, bounded input codec, server transport, custom
  event and startup owners before authoring gameplay predicate/effect methods.
* The server supplies the actor. Input names, arity, types, ranges and declared
  choices are checked before dispatch. An absent/declining listener returns
  failure. Passive, automatic and negated occurrences cannot gain a new manual
  trigger merely from an action word elsewhere in the text.
* Defer publishing the generated producer until listener authoring succeeds.
  Re-render receipts and check source bytes and both startup entries on replay.
  Compiler staging includes the producer's concrete server sources.
* Replace unchanged prior output for the same behavior when its contract or
  input metadata is reauthored; preserve user-modified output by rejecting the
  replacement before changing files. Share the existing Mixin manifest lock.
* Bind every generated event receipt to its behavior ID. A different action's
  correctly compiled event cannot certify this behavior's registration.
* Resolve callback class access through actual inspected owner records when
  symbol-table keys are opaque; preserve access/side restrictions and avoid
  copying the complete target index per callback.

Independent review
------------------

One independent read-only review identified passive-clause admission,
reauthoring/receipt duplication and cross-behavior registration proof issues.
Each was reproduced with a failing regression, then corrected and rerun. A
separate concurrency regression covers the shared manifest-writer lock.

Verification
------------

* Final related suite: 192 passed, 1 skipped (exact Minecraft/Fabric SDK unavailable).
* The generated Java codec/event harness compiled and executed using a real
  JDK, checking a changed state, rejected inputs and declined effects against
  a controlled external Fabric/Minecraft ABI fixture.
* Repository CI Ruff rules (F,E7,E9): passed.
* Python compilation: passed.
* Wheel build: passed; the packaged producer exactly matches the source.
* Completed full suite: 4,647 passed, 3 failed, 8 skipped. The final shared
  project-lock adjustment was subsequently covered by the 192-test related
  run (including both UI and Mixin writers) and a 70-test concurrency/API run.
  A preceding broad run overlapped with source edits and triggered the
  intended integrity-snapshot guard; that superseded run is not certification.

Remaining full-suite failures
-----------------------------

* ``tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_runs_real_build_and_packaging_without_live_model``
* ``tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_releases_when_only_jdt_infrastructure_is_unavailable``

Both fail while the real JDT owner build attempts to resolve locked dependencies
from Maven/Gradle repositories: outbound network access is unavailable. No
generation result from these tests is claimed.

* ``tests/test_external_mcp_role_scope.py::test_bridge_schema_binding_isolated_by_server_scope_and_owner_reused``

The full process observes an extra provider-a schema-selection call. Its targeted
module passes separately; the shared-process role-scope ordering issue remains
unresolved and is not presented as a green full suite.

Limits
------

This fixes construction and verification of an explicit user-input occurrence.
It does not certify the saved design's gameplay semantics, remove overlapping
requested feature scopes, or prove that free-text world observations are
implemented. The uploaded saved plan also contains inconsistent state/action
records. No automatic rewrite of that approved plan was made.

The new Java execution harness uses controlled external ABI fixtures; exact
Minecraft 26.2 client/initializer compilation, actual MiMo generation, a loaded
Minecraft mod and live gameplay remain unverified in this environment. The
host continues to require its ordinary target build and runtime acceptance.
