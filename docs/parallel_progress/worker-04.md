WORKER: 04
ROLE: Agent + MCP + Tool Routing
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: c7087732754c357a632e3157e4e31c964508fe0e

COMPLETED:
- Audited agent role routing, capability filtering, model tool execution phase enforcement, tool-result reinjection, and current forced-tool control flow.
- Confirmed tool results are appended as matching assistant tool_calls + tool observations before the next model turn.
- Confirmed the execution loop rejects model calls outside the exact current phase tool surface before runtime execution.
- Fixed explicit unreviewed model roles failing open to the entire raw tool schema surface.
- Made explicit unreviewed roles fail closed for Skill contracts and external MCP capability routing while preserving model_role="" catalog/introspection behavior.
- Added regression coverage for unknown-role tools, Skills, reviewed MCP servers, capabilities, and access maps.

IN_PROGRESS:
- Separate host-owned invariants (a tool/evidence/action is required) from model-owned semantic routing (which eligible tool to select), especially required-RAG turns.
- Audit external MCP manifest failure visibility and authority overlap without weakening fail-closed behavior.
- Wait for the checkpoint CI run to finish and inspect any failures attributable to worker 04.

ROOT_CAUSES_CONFIRMED:
- filter_tool_schemas_for_role treated an explicit role with no reviewed route as a reason to return the complete raw tool surface. A misspelled/new/unconfigured role therefore gained rather than lost tool authority.
- _request_contracts treated missing role skill assignment as generic stage introspection even when a non-empty execution model role was supplied, leaking stage Skills into unreviewed-role metadata.
- External capability context used `not role_routes` as an allow-all condition, so an explicit unreviewed role could receive provider capability metadata despite having no reviewed server route.

DECISIONS_AND_EVIDENCE:
- Keep deterministic authorization and phase enforcement host-owned; model output cannot authorize a tool that was not on the reviewed surface.
- Preserve generic model_role="" context for catalog/reachability diagnostics; only explicit execution roles fail closed when absent from agent_roles.yaml.
- Current agent_roles.yaml declares silent_fallback: forbidden and explicitly maps planner/researcher/coder/coder_safe/visual_critic/image_generator roles to reviewed routes.
- MCP 2026 guidance treats tool metadata/annotations as non-enforcement hints and places hard safety/authorization guarantees at the host/client boundary.

COMMITS_ALREADY_PUSHED:
- 787c7fe663009fa7e807d9777875296bfda2f2bc fix: fail closed for unreviewed agent roles
- c7087732754c357a632e3157e4e31c964508fe0e test: deny tools for unknown agent roles

TESTS_ALREADY_PASSING:
- GitHub Actions CI run 33325365785: Static and packaging audit PASS.
- GitHub Actions CI run 33325365785: Minecraft MCP evidence PASS.
- GitHub Actions CI run 33325365785: Planner host template contract PASS.
- GitHub Actions CI run 33325365785: Parallel repair safety PASS.
- Full CI still in progress at checkpoint write.

NEXT_EXACT_ACTIONS:
1. Finish CI verification for c7087732754c357a632e3157e4e31c964508fe0e and repair any worker-04 regression.
2. Refactor required-evidence routing so the host enforces the invariant without unnecessarily selecting one semantic retrieval tool when multiple reviewed choices exist; add regression coverage.
3. Audit external MCP manifest error handling and role/authority duplication; fix or document a handoff if the required change belongs to worker 12 shared core.
4. Re-fetch latest origin/main, verify worker commits are ancestors, rerun relevant CI, then mark this progress file COMPLETE only when worker-04 unresolved items are empty.

FILES_CURRENTLY_RELEVANT:
- minecraft_mod_ai/agent_capability_context.py
- minecraft_mod_ai/agent_roles.py
- minecraft_mod_ai/agent_tool_runtime.py
- minecraft_mod_ai/external_mcp_router.py
- minecraft_mod_ai/model_router.py
- minecraft_mod_ai/progress_aware_tool_loop.py
- minecraft_mod_ai/llama_server_hardware_policy.py
- tests/test_agent_capability_context.py
- tests/test_agent_tool_calling.py

KNOWN_CROSS_ROLE_DEPENDENCIES:
- progress_aware_tool_loop.py participates in coder localization/repair behavior owned in part by worker 07; tool-routing changes must preserve localization semantics.
- Any large top-level orchestration change belongs to worker 12 and must be handed off instead of rewritten here.

UNRESOLVED:
- Required-RAG host policy still names a specific retrieval tool after prose/no-evidence instead of expressing only the required-evidence invariant when multiple reviewed semantic routes are available.
- External MCP manifest construction currently converts broad exceptions into an empty capability map without exposing why routing degraded.
- Full checkpoint CI run 33325365785 not yet complete.
