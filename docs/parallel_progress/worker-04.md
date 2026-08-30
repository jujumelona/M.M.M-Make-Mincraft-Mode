WORKER: 04
ROLE: Agent + MCP + Tool Routing
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: ab6744b91ccd05d3e0650e1fdb5287a5054ff5e4

COMPLETED:
- Audited agent role routing, capability filtering, model tool execution phase enforcement, tool-result reinjection, and forced-tool control flow.
- Confirmed tool results are appended as matching assistant tool_calls + tool observations before the next model turn.
- Confirmed the execution loop rejects model calls outside the exact current phase tool surface before runtime execution.
- Fixed explicit unreviewed model roles failing open to the entire raw tool schema surface.
- Made explicit unreviewed roles fail closed for Skill contracts and external MCP capability routing while preserving model_role="" catalog/introspection behavior.
- Added regression coverage for unknown-role tools, Skills, reviewed MCP servers, capabilities, and access maps.
- Separated the host-owned fresh-evidence invariant from model-owned semantic RAG route selection when more than one reviewed retrieval route is eligible. The host now uses generic tool_choice="required" instead of choosing search_code_rag over search_project_rag in that case.
- Preserved exact host forcing when only one reviewed RAG route remains, because there is no semantic choice left for the model.
- Added an integration regression in which the model selects search_project_rag while both reviewed RAG routes are exposed, and the resulting observation is reinjected before final reasoning.
- Made external MCP capability-manifest degradation explicit while remaining fail closed: capability/access maps stay empty, status becomes UNAVAILABLE/MANIFEST_BUILD_FAILED, and raw exception text is not included in model context.
- Added regression coverage proving manifest failure state is visible without leaking the provider error message/token-like text.
- Verified the worker-04 checkpoint CI run 33325365785 had no worker-04 failures; its remaining failures were in worker-02 RAG and worker-03 game-design tests.
- Verified worker-04 commits through affe72de805b37440704052f9438d465ffa856a4 remain ancestors of the moving origin/main while other workers continue pushing.

IN_PROGRESS:
- Obtain one non-cancelled latest-main CI execution containing the new required-RAG and MCP-manifest regressions; rapid parallel main pushes currently trigger the workflow's cancel-in-progress policy before the test shards run.

ROOT_CAUSES_CONFIRMED:
- filter_tool_schemas_for_role treated an explicit role with no reviewed route as a reason to return the complete raw tool surface. A misspelled/new/unconfigured role therefore gained rather than lost tool authority.
- _request_contracts treated missing role skill assignment as generic stage introspection even when a non-empty execution model role was supplied, leaking stage Skills into unreviewed-role metadata.
- External capability context used `not role_routes` as an allow-all condition, so an explicit unreviewed role could receive provider capability metadata despite having no reviewed server route.
- Required-evidence recovery mixed deterministic host policy with semantic model routing: after prose/no evidence the host named search_code_rag first and encoded that preference into an exact function tool_choice even when search_project_rag was equally reviewed and eligible.
- External MCP capability-manifest construction converted any manifest exception into an indistinguishable empty map, hiding the difference between an intentionally empty reviewed surface and degraded capability discovery.

DECISIONS_AND_EVIDENCE:
- Keep deterministic authorization, phase enforcement, evidence sufficiency, mutation requirements, and fail-closed behavior host-owned; model output cannot authorize a tool that was not on the reviewed surface.
- Keep semantic selection among multiple already-authorized read-only evidence routes model-owned. A required tool call is an execution invariant, not a reason for the host to decide which evidence source best matches the information need.
- Use generic native tool_choice="required" when multiple reviewed RAG choices exist; require exactly one reviewed RAG call on that bounded turn and disable parallel calls for the invariant check.
- Preserve generic model_role="" context for catalog/reachability diagnostics; only explicit execution roles fail closed when absent from agent_roles.yaml.
- Current agent_roles.yaml declares silent_fallback: forbidden and explicitly maps planner/researcher/coder/coder_safe/visual_critic/image_generator roles to reviewed routes.
- The reviewed ground-production-with-live-evidence Skill explicitly allows both search_project_rag and search_code_rag in generation/quality, so host preference between them is not an authorization decision.
- Current MCP tool-choice semantics support a generic required mode distinct from naming a particular tool, matching the invariant/semantic-routing separation.
- Manifest failures expose only a stable category and exception type; raw exception messages are omitted to avoid leaking provider details while preserving observability.

COMMITS_ALREADY_PUSHED:
- 787c7fe663009fa7e807d9777875296bfda2f2bc fix: fail closed for unreviewed agent roles
- c7087732754c357a632e3157e4e31c964508fe0e test: deny tools for unknown agent roles
- 41fa59046d220ea67b009938b06ec9220971b737 fix: expose external MCP manifest degradation
- 88d16f58fa096089a454e98e4913c59496df30a8 test: surface fail-closed MCP manifest degradation
- 41b3279e092facb9d8c6c0a092a51838093db855 refactor: separate required evidence from RAG route choice
- affe72de805b37440704052f9438d465ffa856a4 test: let model choose among required RAG routes

TESTS_ALREADY_PASSING:
- GitHub Actions CI run 33325365785: Static and packaging audit PASS.
- GitHub Actions CI run 33325365785: Minecraft MCP evidence PASS.
- GitHub Actions CI run 33325365785: Planner host template contract PASS.
- GitHub Actions CI run 33325365785: Parallel repair safety PASS.
- Agent/MCP/tool-routing tests in that checkpoint passed; the observed shard failures were cross-owner worker-02/worker-03 failures.
- New CI runs after affe72d have repeatedly been CANCELLED, not failed, because newer parallel main pushes replace them under concurrency.cancel-in-progress=true.

NEXT_EXACT_ACTIONS:
1. Re-fetch origin/main and locate the first non-cancelled CI run whose head contains affe72de805b37440704052f9438d465ffa856a4.
2. Inspect Static and packaging audit plus the deterministic pytest shard containing test_agent_required_rag_semantic_choice.py and test_agent_capability_manifest_degradation.py.
3. If any failure is attributable to worker 04, fix it on latest main with a coherent tested checkpoint; do not touch unrelated worker failures.
4. Re-verify all worker-04 commit SHAs are ancestors of latest origin/main.
5. When the new worker-04 regressions have a real PASS and no worker-04 unresolved items remain, set STATUS: COMPLETE and push the final progress checkpoint.

FILES_CURRENTLY_RELEVANT:
- minecraft_mod_ai/agent_capability_context.py
- minecraft_mod_ai/agent_roles.py
- minecraft_mod_ai/agent_tool_runtime.py
- minecraft_mod_ai/external_mcp_router.py
- minecraft_mod_ai/model_router.py
- minecraft_mod_ai/progress_aware_tool_loop.py
- tests/test_agent_capability_context.py
- tests/test_agent_capability_manifest_degradation.py
- tests/test_agent_required_rag_semantic_choice.py
- tests/test_agent_tool_calling.py
- tests/test_agent_routing_contract.py

KNOWN_CROSS_ROLE_DEPENDENCIES:
- progress_aware_tool_loop.py participates in coder localization/repair behavior owned in part by worker 07; the worker-04 change preserves host localization stages and changes only the multiple-reviewed-RAG semantic-choice case.
- Any large top-level orchestration change belongs to worker 12 and must be handed off instead of rewritten here.
- Repository-wide CI may remain red for worker-02/worker-03 or other concurrently changing areas; worker 04 must distinguish those from its own regressions.

UNRESOLVED:
- Need one non-cancelled CI execution containing commits 41fa590, 88d16f5, 41b3279, and affe72d so the newly added worker-04 regressions actually execute on GitHub Actions.
