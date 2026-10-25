Native callback catalogue repair
===============================

The production run based on ``5fb118c8`` stopped with
``CALLBACK_RUNTIME_MUTATION_TARGET_UNRESOLVED: Player health``. The body author
supported scalar value slots, but its upstream callback API sampler discarded
methods whose parameters were absent from the callback itself. The native
selector also stopped after a fixed candidate prefix. A valid inspected
mutator could therefore disappear before the model selected an operation.

Native effect authoring now has a complete, host-owned catalogue separate
from the relevance-ranked callback prompt. Accessible inspected instance
methods retain their exact signatures, scalar value slots and verified
one-hop receiver alternatives, including paths back to callback root types.
Receiver paths always start from original event inputs. Equivalent callback
frames reuse the bound catalogue within one source construction.

The model selects checked IDs from bounded pages; unavailable advances to
the next page until the catalogue is exhausted. Exact duplicate invocations
are emitted once while retaining a receipt for every explicitly selected
state obligation. Separate behaviors can legitimately use the same API.

Compiler repair uses the same catalogue and replaces the repaired behavior's
receipts together with its body. Final source verification still requires
each receipted invocation directly inside its own guarded effect helper.
The construction implementation fingerprint includes the new catalogue
module, invalidating earlier source checkpoints automatically.

``tests/test_native_callback_catalog.py`` exercises real index discovery,
the complete first-pass source boundary, Java compilation/execution, paging
beyond the former candidate limits, receiver identity, shared obligations
and compiler repair. Fixture Java execution establishes those contracts;
real-model selection, Minecraft gameplay and the full production index's
time/RAM still require their production runtime checks.

Local verification record
-------------------------

The related generation/repair suite passed 95 tests, and full Python
compilation plus critical Ruff checks passed. The broader parallel run
passed 4,699 tests and skipped eight. Its separate MCP cache test
``tests/test_external_mcp_role_scope.py::test_bridge_schema_binding_isolated_by_server_scope_and_owner_reused``
failed in that batch; a fresh run of the complete five-test file passed.

Two external-build tests were attempted separately and blocked by Maven
dependency downloads reporting ``Network is unreachable``:

* ``tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_runs_real_build_and_packaging_without_live_model``
* ``tests/test_deterministic_debug_full_pipeline_e2e.py::test_debug_fixture_releases_when_only_jdt_infrastructure_is_unavailable``

These observations establish the callback repair contracts and preserve
the outstanding full-runtime checks explicitly.

Catalogue discovery, receiver validation, model-page presentation and repair
publication have separate helpers. Runtime repair bindings retain receipt
ownership while preserving existing repair signatures. The architecture
regression audit against the original source reports zero violations; the
95 related tests also pass after this responsibility split.

Official Minecraft 26.2 verification
----------------------------------

JDK 25 compiled all 111 discovered common Fabric callbacks, their proof
state owner and the entity/attribute registry: 114 Java sources, exit 0.
The compiler-only probe uses an explicit owned state write, so it also
covers callbacks whose inputs expose no gameplay mutator.

The actual Minecraft/Fabric index contains 164,380 symbols. For the exact
combat callback in the failed run, the old prompt sample omitted both
``setHealth(float)`` and ``ItemStack.setDamageValue(int)``; the complete
catalogue retains both. Argument combinations are represented as independent
checked value slots. On this index, method choices fell from 168,064 to
6,825 while retaining all operand alternatives. Method menus contain compact
argument summaries; participant IDs are selected on separate bounded pages.

The final related regression set passes 112 tests, including arguments beyond
the first page. Real-model generation and in-game behavior remain separate
runtime checks.
