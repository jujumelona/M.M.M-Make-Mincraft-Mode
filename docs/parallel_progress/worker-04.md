WORKER: 04
ROLE: Agent + MCP + Tool Routing
STATUS: COMPLETE
LAST_UPDATED_MAIN_SHA: a712361de8416e9496a9d078b7b683f1681bb184
HANDOFF: SAFE_FOR_WORKER_13

COMPLETED:
- Audited and hardened agent role routing, capability filtering, model tool execution phase enforcement, tool-result reinjection, forced-tool control flow, Skill routing, and external MCP discovery/execution.
- Explicit unreviewed model roles fail closed for tool schemas, Skill contracts, and external MCP capability/access routing while preserving model_role="" catalog/introspection behavior.
- Required production evidence remains host-owned while semantic choice among multiple reviewed RAG routes remains model-owned; exact host forcing is retained when only one reviewed route remains.
- Weak/exhausted RAG sources are excluded from later candidate selection so evidence exhaustion fails at the correct boundary instead of retrying a dead route.
- External MCP manifest degradation is explicit and fail closed without leaking provider exception/token-like text.
- Tool results are reinjected as matching assistant tool_calls plus role=tool observations before subsequent model reasoning.

EXHAUSTIVE_QUALITY_AND_PERFORMANCE_AUDIT:
- Initial Worker-04 Ruff audit found 22 findings; final audited surface is Ruff-clean.
- Vulture --min-confidence 90 reports no high-confidence dead code in the Worker-04 production surface. Live compatibility code was preserved when runtime evidence proved it was required.
- Capability/tool-surface preparation previously re-read agent_roles.yaml 10 times in one request path. Request-level role-policy snapshotting now reduces that contract to one policy load per preparation.
- Distinct read-only external_mcp_call operations are admitted to the parallel read wave; write/admin calls remain serial and are never promoted to the read wave.
- Exact duplicate read calls still retain the independent single-flight dedup optimization.
- Removed the router-wide lock around independent external MCP provider I/O, eliminating avoidable head-of-line serialization while retaining per-call transport/session ownership.
- ExternalAgentBridge lazy ExternalMCPRouter creation is double-checked under lock so concurrent first access constructs one router instance.
- temporary_skill_contract now uses the live parallel-read classifier instead of a stale duplicated allowlist, removing split routing authority.
- Fixed progress-loop closure late binding so phase tool surface and localization stage are captured per turn without changing Worker-07 localization semantics.
- Narrowed/annotated intentional exception boundaries, tightened typing and path predicates, and retained compatibility exports required by runtime monkey-patch contracts.
- Targeted Worker-04 modules compile successfully and git diff hygiene passes.

PERMANENT_REGRESSION_CONTRACT:
- tests/test_worker04_quality_contract.py verifies request-local role-policy loading, distinct read-only MCP parallelism, write/admin serial classification, absence of a router-wide provider-I/O lock, and thread-safe lazy router initialization.
- Existing Agent/MCP/RAG/hot-path/routing regressions remain included in the final validation set.

KEY_COMMITS:
- 787c7fe663009fa7e807d9777875296bfda2f2bc fix: fail closed for unreviewed agent roles
- c7087732754c357a632e3157e4e31c964508fe0e test: deny tools for unknown agent roles
- 41fa59046d220ea67b009938b06ec9220971b737 fix: expose external MCP manifest degradation
- 88d16f58fa096089a454e98e4913c59496df30a8 test: surface fail-closed MCP manifest degradation
- 41b3279e092facb9d8c6c0a092a51838093db855 refactor: separate required evidence from RAG route choice
- affe72de805b37440704052f9438d465ffa856a4 test: let model choose among required RAG routes
- b78a93fe2b53350f8c9506be8c3ef6618c248def fix: exhaust weak RAG routes before retry
- 059a03fe590ea3ddf753db7a1544a97ec3610a06 test: enforce worker04 routing quality invariants
- 3c5cae0 refactor: snapshot agent routing policy once per request
- a4f6a31d1968566854b2c76075b82f8026841b7f test: separate worker04 parallelism from read dedup
- 77522e16f1ab02cac979f87b273d96131a2b696d refactor: harden worker04 routing hot paths
- 2c39ef4ca7abcf057de644281e8480f6033cb83d chore: remove worker04 audit scaffolding
- a712361de8416e9496a9d078b7b683f1681bb184 chore: remove worker04 final validation scaffolding

FINAL_VALIDATION:
- GitHub Actions run 33329433932 validated clean production state without applying any patch.
- Ruff targeted Worker-04 audit: PASS / 0 findings.
- Behavioral regression: 49/49 PASS.
- Vulture --min-confidence 90: PASS / 0 high-confidence dead-code findings.
- Python compileall on the Worker-04 production surface: PASS.
- git diff --check: PASS.
- The validation SHA b4bd1f2902a5da17d1fc4417c315a846c9d420b7 was followed only by unrelated Worker-06 workflow metadata before validation scaffolding cleanup; no Worker-04 production/test file changed after validation at closure time.
- All temporary Worker-04 audit, patch, trigger, and validation workflow files were removed from .github after evidence collection.

AUDITED_PRODUCTION_SURFACE:
- minecraft_mod_ai/agent_capability_context.py
- minecraft_mod_ai/agent_roles.py
- minecraft_mod_ai/agent_tool_runtime.py
- minecraft_mod_ai/agent_intent.py
- minecraft_mod_ai/agent_routing_intent_contract.py
- minecraft_mod_ai/external_agent_bridge.py
- minecraft_mod_ai/external_mcp_router.py
- minecraft_mod_ai/model_router.py
- minecraft_mod_ai/progress_aware_tool_loop.py
- minecraft_mod_ai/temporary_skill_contract.py
- relevant role/Skill/MCP configuration and Worker-04 regression tests

CROSS_ROLE_BOUNDARIES:
- progress_aware_tool_loop.py overlaps Worker-07 coder localization; Worker-04 hardening preserved localization ownership semantics.
- Top-level orchestration remains Worker-12 ownership and was not rewritten by Worker-04.

UNRESOLVED:
- none
