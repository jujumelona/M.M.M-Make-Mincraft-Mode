# Worker 13 — Final Integrator

WORKER: 13
ROLE: Final integration across Workers 01-12
STATUS: IN_PROGRESS
BASE_MAIN_SHA: fe3da4848e9270baa23ca99dc09cad971a163dab
LAST_UPDATED_MAIN_SHA: 356732f0eee102f48084c03efd74c562ad06b631

COMPLETED:
- Verified Workers 01-12 are finalized on origin/main; Worker 03 is COMPLETE, has no unresolved owned work, and is READY_FOR_WORKER_13.
- Read the Worker 13 protocol and current cross-role handoff set.
- Audited latest full CI run 33383088159 and separated dedicated green integration gates from the three failing remaining-test shards.
- Confirmed static/packaging audit, runtime composition, planner host template contract, Minecraft MCP evidence, parallel repair safety, conversational UI contract, and observability regression are green on the pre-integration main.
- Fixed the first concrete integration break: final-architecture regression collection still imported diagnostic_errors/flatten_diagnostics from the retired repair diagnostic adapter instead of the canonical validation_diagnostic_contract authority.

ROOT_CAUSES_CONFIRMED:
- Repository-wide remaining tests still contain stale expectations for architectures intentionally removed/hardened by Workers 01-12 (static target matrix/NeoForge advertisement, permissive design fallback/coercion, permissive JDT quiet settlement, weaker reuse-proof promotion, legacy deterministic template assumptions, and old runtime/tool-loop expectations).
- Some failures are genuine integration references to retired symbols/modules and must be migrated to the canonical owner rather than restored through compatibility shims.

DECISIONS_AND_EVIDENCE:
- Do not restore retired duplicate authorities merely to satisfy old tests. Update integration callers/tests to authoritative contracts and preserve fail-closed hardening.
- Treat all three remaining-test shards as Worker 13 reconciliation work; no worker-owned blocker remains.

COMMITS_ALREADY_PUSHED:
- 356732f0eee102f48084c03efd74c562ad06b631 test: follow canonical validation diagnostics authority

TESTS_ALREADY_PASSING:
- CI 33383088159: Static and packaging audit PASS
- CI 33383088159: Runtime composition PASS
- CI 33383088159: Planner host template contract PASS
- CI 33383088159: Minecraft MCP evidence PASS
- CI 33383088159: Parallel repair safety PASS
- CI 33383088159: Conversational UI contract PASS
- Observability Regression 33383088021 PASS

NEXT_EXACT_ACTIONS:
1. Inspect the new CI from 356732f0... so shard 3 can fully collect and expose remaining final-architecture mismatches.
2. Reconcile stale cross-worker tests with the authoritative post-worker contracts without weakening production fail-closed behavior.
3. Fix genuine integration references/imports, then iterate repository-wide CI until all remaining-test shards pass.
4. Validate request→requirements→research→design→target→reuse→tasks→coder→compile/repair→validation→jar/runtime and all handoffs.
5. Mark Worker 13 COMPLETE only after final main CI is green and push verification is confirmed.

FILES_CURRENTLY_RELEVANT:
- tests/test_final_architecture_contract.py
- tests/test_fresh_mutation_target_grounding.py
- tests/test_pre_design_rag_root_cause.py
- tests/test_canonical_capability_ontology_and_reuse_proof.py
- tests/test_runner_parallel_validation_contract.py
- tests/test_agentic_research_game_design.py
- tests/test_game_design_router.py
- tests/test_java_lsp_scaling.py
- tests/test_java_lsp_diagnostic_quiet.py
- tests/test_system_pack_generator.py
- tests/test_content_catalog_scaling.py
- docs/parallel_handoff/worker-08.md
- docs/parallel_handoff/worker-10.md
- docs/parallel_handoff/worker-11.md

KNOWN_CROSS_ROLE_DEPENDENCIES:
- Worker 08 handoff is an integration input: deterministic/specialized capabilities must remain provider-evidence gated.
- Worker 10/11 JDT handoffs are integration inputs; Worker 12 already consolidated major diagnostic ownership, and Worker 13 must ensure stale references/tests follow it.

UNRESOLVED:
- Full remaining-test shards are not green yet.
- Full end-to-end final integration validation not yet complete.
