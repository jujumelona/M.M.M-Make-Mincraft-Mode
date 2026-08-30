WORKER: 04
ROLE: Agent + MCP + Tool Routing
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: b4524fbea02536d97ee93590f690eee07f319bfc

REOPENED_FOR_EXHAUSTIVE_QUALITY_AUDIT:
- Reopened after functional completion to audit clean code, dead/duplicated code, avoidable branching/state, exception handling, routing bottlenecks, unnecessary tool/model turns, allocation/serialization overhead, code size/complexity, and missing regression boundaries across the full Worker-04 ownership surface.
- Completion now requires not only functional regression PASS but also no confirmed Worker-04-owned cleanup/optimization defect remaining after static/dynamic review.

COMPLETED_FROM_PRIOR_PASS:
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
- Fixed required-RAG exhaustion so attempted weak RAG sources are excluded from later candidate selection.

COMMITS_ALREADY_PUSHED:
- 787c7fe663009fa7e807d9777875296bfda2f2bc fix: fail closed for unreviewed agent roles
- c7087732754c357a632e3157e4e31c964508fe0e test: deny tools for unknown agent roles
- 41fa59046d220ea67b009938b06ec9220971b737 fix: expose external MCP manifest degradation
- 88d16f58fa096089a454e98e4913c59496df30a8 test: surface fail-closed MCP manifest degradation
- 41b3279e092facb9d8c6c0a092a51838093db855 refactor: separate required evidence from RAG route choice
- affe72de805b37440704052f9438d465ffa856a4 test: let model choose among required RAG routes
- b78a93fe2b53350f8c9506be8c3ef6618c248def fix: exhaust weak RAG routes before retry

PRIOR_VALIDATION:
- GitHub Actions Worker 04 Validation run 33327196364: 13/13 PASS in 1.62s for the targeted Agent/MCP routing regressions.

AUDIT_SURFACE:
- minecraft_mod_ai/agent_capability_context.py
- minecraft_mod_ai/agent_roles.py
- minecraft_mod_ai/agent_tool_runtime.py
- minecraft_mod_ai/agent_intent.py
- minecraft_mod_ai/agent_routing_intent_contract.py
- minecraft_mod_ai/external_agent_bridge.py
- minecraft_mod_ai/external_mcp_router.py
- minecraft_mod_ai/model_router.py
- minecraft_mod_ai/progress_aware_tool_loop.py
- config/role/skill/MCP routing definitions and Worker-04 regression tests

KNOWN_CROSS_ROLE_DEPENDENCIES:
- progress_aware_tool_loop.py overlaps coder localization/repair semantics with Worker 07; do not alter localization ownership semantics without evidence and a narrow change.
- Large top-level orchestration remains Worker 12 ownership.

NEXT_EXACT_ACTIONS:
1. Perform static structural audit for duplicate/dead branches, broad exception swallowing, repeated allowlist/routing calculations, unnecessary serialization/copies, and stale compatibility code in the Worker-04 surface.
2. Trace hot execution paths for redundant model turns/tool calls and repeated O(N) work; distinguish material bottlenecks from harmless micro-optimizations.
3. Add regression/benchmark-style tests for each confirmed defect before or with the fix.
4. Run targeted lint/compile/tests plus Worker-04 dynamic regressions on a current-main descendant.
5. Mark COMPLETE only when no Worker-04-owned confirmed defects remain.

UNRESOLVED:
- Exhaustive clean-code/performance audit in progress.
