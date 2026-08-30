WORKER: 04
ROLE: Agent + MCP + Tool Routing
STATUS: COMPLETE
LAST_UPDATED_MAIN_SHA: 396b0838244f5d730fe815c3df8c61a09e8b09a2

COMPLETED:
- Audited agent role routing, capability filtering, model tool execution phase enforcement, tool-result reinjection, forced-tool control flow, and external MCP capability discovery.
- Confirmed tool results are appended as matching assistant tool_calls + tool observations before the next model turn.
- Confirmed the execution loop rejects model calls outside the exact current phase tool surface before runtime execution.
- Fixed explicit unreviewed model roles failing open to the entire raw tool schema surface.
- Made explicit unreviewed roles fail closed for Skill contracts and external MCP capability routing while preserving model_role="" catalog/introspection behavior.
- Added regression coverage for unknown-role tools, Skills, reviewed MCP servers, capabilities, and access maps.
- Separated the host-owned fresh-evidence invariant from model-owned semantic RAG route selection when more than one reviewed retrieval route is eligible. The host now uses generic tool_choice="required" instead of choosing search_code_rag over search_project_rag in that case.
- Preserved exact host forcing when only one reviewed RAG route remains, because there is no semantic choice left for the model.
- Added an integration regression in which the model selects search_project_rag while both reviewed RAG routes are exposed, and the resulting observation is reinjected before final reasoning.
- Made external MCP capability-manifest degradation explicit while remaining fail closed: capability/access maps stay empty, status becomes UNAVAILABLE/MANIFEST_BUILD_FAILED, and raw exception text is not included in model context.
- Added regression coverage proving manifest failure state is visible without leaking provider error text/token-like content.
- During final validation found a Worker-04 regression in required-RAG exhaustion: an already-attempted weak RAG source could re-enter the semantic-choice candidate set.
- Fixed required-RAG exhaustion so attempted weak RAG sources are excluded from later candidate selection; this preserves semantic choice among untried reviewed routes and terminates correctly when all reviewed routes are exhausted.
- Verified the final Worker-04 code diff for b78a93f changes only the required-RAG candidate calculation and no unrelated code.
- Created a temporary independent Worker-04 validation workflow because repository-wide CI was blocked by an unrelated runtime-mutation budget increase; removed the workflow immediately after validation.
- Verified all Worker-04 functional commits and the validation SHA remain ancestors of the moving main branch.

ROOT_CAUSES_CONFIRMED:
- filter_tool_schemas_for_role treated an explicit role with no reviewed route as a reason to return the complete raw tool surface. A misspelled/new/unconfigured role therefore gained rather than lost tool authority.
- _request_contracts treated missing role skill assignment as generic stage introspection even when a non-empty execution model role was supplied, leaking stage Skills into unreviewed-role metadata.
- External capability context used `not role_routes` as an allow-all condition, so an explicit unreviewed role could receive provider capability metadata despite having no reviewed server route.
- Required-evidence recovery mixed deterministic host policy with semantic model routing: after prose/no evidence the host named search_code_rag first and encoded that preference into an exact function tool_choice even when search_project_rag was equally reviewed and eligible.
- External MCP capability-manifest construction converted any manifest exception into an indistinguishable empty map, hiding the difference between an intentionally empty reviewed surface and degraded capability discovery.
- Initial semantic-choice refactor reused phase_tool_names without excluding state.attempted_sources, allowing an exhausted weak RAG source to be selected again instead of producing the intended evidence-exhaustion boundary.

DECISIONS_AND_EVIDENCE:
- Keep deterministic authorization, phase enforcement, evidence sufficiency, mutation requirements, and fail-closed behavior host-owned; model output cannot authorize a tool that was not on the reviewed surface.
- Keep semantic selection among multiple already-authorized read-only evidence routes model-owned. A required tool call is an execution invariant, not a reason for the host to decide which evidence source best matches the information need.
- Use generic native tool_choice="required" when multiple reviewed RAG choices exist; require exactly one reviewed RAG call on that bounded turn and disable parallel calls for the invariant check.
- Preserve generic model_role="" context for catalog/reachability diagnostics; only explicit execution roles fail closed when absent from agent_roles.yaml.
- Preserve exact host forcing only when one untried reviewed route remains.
- Exclude state.attempted_sources from later required-RAG candidate selection so a weak/exhausted route cannot be silently retried as if it were new semantic choice.
- Manifest failures expose only a stable category and exception type; raw exception messages are omitted to avoid leaking provider details while preserving observability.

COMMITS_ALREADY_PUSHED:
- 787c7fe663009fa7e807d9777875296bfda2f2bc fix: fail closed for unreviewed agent roles
- c7087732754c357a632e3157e4e31c964508fe0e test: deny tools for unknown agent roles
- 41fa59046d220ea67b009938b06ec9220971b737 fix: expose external MCP manifest degradation
- 88d16f58fa096089a454e98e4913c59496df30a8 test: surface fail-closed MCP manifest degradation
- 41b3279e092facb9d8c6c0a092a51838093db855 refactor: separate required evidence from RAG route choice
- affe72de805b37440704052f9438d465ffa856a4 test: let model choose among required RAG routes
- b78a93fe2b53350f8c9506be8c3ef6618c248def fix: exhaust weak RAG routes before retry

FINAL_VALIDATION:
- GitHub Actions Worker 04 Validation run 33327196364, checkout SHA 3618129e5a20efdc27d2e5522bbfcf82fb22f602: 13/13 PASS in 1.62s.
- PASS: tests/test_agent_tool_calling.py::test_required_rag_exhaustion_fails_before_hard_round_budget.
- PASS: tests/test_agent_required_rag_semantic_choice.py::test_required_evidence_does_not_force_one_semantic_rag_route.
- PASS: both tests in tests/test_agent_capability_manifest_degradation.py.
- PASS: all 9 tests in tests/test_agent_capability_context.py.
- Earlier CI run 33326415253 also showed Worker-04 semantic-choice/capability tests passing; its remaining failures were assigned to other worker scopes.
- Repository-wide CI at later main heads is currently blocked before pytest shards by runtime mutation surface 424 > reviewed budget 413. That gate failure is outside Worker 04 and does not involve Worker-04 files.
- b78a93fe2b53350f8c9506be8c3ef6618c248def and validation SHA 3618129e5a20efdc27d2e5522bbfcf82fb22f602 were both verified as ancestors of main after validation.

FILES_CHANGED_OR_VALIDATED:
- minecraft_mod_ai/agent_capability_context.py
- minecraft_mod_ai/agent_roles.py
- minecraft_mod_ai/progress_aware_tool_loop.py
- tests/test_agent_capability_context.py
- tests/test_agent_capability_manifest_degradation.py
- tests/test_agent_required_rag_semantic_choice.py
- tests/test_agent_tool_calling.py

KNOWN_CROSS_ROLE_DEPENDENCIES:
- progress_aware_tool_loop.py participates in coder localization/repair behavior owned in part by worker 07; Worker-04 changes preserve localization stages and alter only reviewed RAG invariant/selection behavior.
- Large top-level orchestration remains Worker 12 ownership.
- Repository-wide CI may remain red from independently changing runtime/planner/validation areas; those failures do not reopen Worker 04 unless they identify an Agent/MCP/tool-routing regression.

UNRESOLVED:
- none
