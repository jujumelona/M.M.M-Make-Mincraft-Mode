Runtime source repair: validation and limits
===========================================

Source change: f749553ef05722fbfe914b99e968bff8d7c67b34

Observed production failures
----------------------------

The attached run used 071e908724439b747c9350df3260531404f5efc7.
There were 829 completion journals (591 planner, 238 coder).
The actual source handoff admitted the generated entity as materialized_source,
but observation lowering accepted only materialized_prerequisite_declaration.
A second defect accepted getMaxSectionY as selected_section on one matching word.
The coder then accumulated unrelated JNA callbacks, world flags and reflection
queries without constructing the required predicate.

Changes
-------

* One path/hash-based materialized-type evidence contract is consumed by source
  resolution and typed IR. Existing canonical source members take precedence
  over the declaration fallback without adding duplicate aliases.
* The host builds complete inspected receiver paths before choosing an event.
  All required observations must be available together. Partial word matches
  do not certify a source. Source/destination-world ambiguity remains explicit.
* The model selects complete paths when necessary. The host emits their typed
  calls, shares prefixes, guards absent intermediate receivers, and pins the
  resulting value IDs in later prompts.
* Predicates may inspect objects derived from pinned sources using checked
  observation APIs. Unrelated callback receivers, static JNA helpers and
  reflection discovery cannot supply missing source evidence. Effect authoring
  retains its existing separate phase.
* Exact source plans and unverified-runtime flags remain in the saved rule IR.
  Existing checkpoints depend on the changed transitive implementation hash.

Verified evidence
-----------------

82 relevant tests passed with JDK 25, including actual javac/java execution of
entity presence and repeated callbacks, multi-hop observations with absent
receivers, production-shaped source indexing, event intersections, object-valued
source queries and source-ID visibility beyond the recent-value window.
Repository-wide F/E7/E9 checks, compileall and git diff --check passed.

An independent review identified object-valued query access and multiple-world
receiver gaps. Both were reproduced as failing tests and fixed before the
source change was published. The GitHub impacted-tests, changed-Python, real Gradle/GameTest and final
partial-CI gate all passed for f749553ef05722fbfe914b99e968bff8d7c67b34.

Scope and remaining obligations
-------------------------------

This repairs source resolution and bounded observer construction. A named
custom gameplay concept still requires a real producer in the generated
program. The attached selected_section input has no proved producer; accepting
world section height would conceal that missing implementation. The new path
reports that obligation without spending hundreds of unrelated API decisions.
The currently unbound selection/build behavior remains an implementation
obligation; this change does not synthesize that behavior.

The current source planner follows public zero-argument observation members
for at most three receiver hops. Parameter-dependent producers and deeper
custom observers need an explicit implementation rather than a guessed API.

No real MiMo 9B full generation or original-request Minecraft playtest was run
here. The executable Java probes use explicit test implementations of the
inspected ABI. They do not certify all gameplay effects or the whole mod.

Whole-suite and baseline results
-------------------------------

Full pytest: 4,389 passed, 4 failed, 7 skipped (74 warnings), 939.90 seconds.
The complete suite is not green.

Failed cases:

* ``tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_runs_real_build_and_packaging_without_live_model``
* ``tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_releases_when_only_jdt_infrastructure_is_unavailable``
* ``tests/test_external_mcp_role_scope.py::test_bridge_schema_binding_isolated_by_server_scope_and_owner_reused``
* ``tests/test_retrieval_cpu_budget_live_owner.py::test_production_tool_boundary_cannot_enable_dense_without_opt_in``

Both deterministic-debug failures reproduced on unchanged 071e90872443.
Their JDT owner could not resolve locked Gradle/Maven dependencies in this
execution environment. The same production integration tests passed in GitHub
CI for f749553e. That CI fixture targets Minecraft 1.21.8 with JDK 21; it is not
an original-request Minecraft 26.2 playtest.

The MCP scope-cache assertion also reproduced on unchanged 071e90872443 when
its six preceding external-MCP files ran together (19 passed, 1 failed).
An isolated run and an all-module-collection/one-test run passed. The assertion
observes an extra schema-description call for the same reviewed provider;
it does not show selection of a forbidden provider. Its order dependency
remains outside this source-plan change.

The RAG CPU dense-opt-in case failed only in the full run. An unchanged-baseline
prefix of 67 relevant resource/RAG tests passed; the current code's isolated
case also passed. Its full-suite order dependency remains unresolved. It is
not labelled fixed or proven pre-existing by those partial baseline runs.

CI evidence:
https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode/actions/runs/39760656152

No claim is made that every repository failure, absent gameplay producer, or
original model-driven generation is resolved by this change.
