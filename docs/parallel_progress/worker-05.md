# Worker 05 Progress

WORKER: 05
ROLE: LLM Runtime + Context + Tool Calling
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: f4253f2d5ab006ce4731ce2bdaeb34a810b6b714

COMPLETED:
- Audited native llama-server semantic/tool transport and the separate text streaming path.
- Confirmed tool/semantic completions are aggregated through SSE by llama_stream_efficiency_contract rather than executed from partial deltas.
- Confirmed current finite output-budget policy already protects apply_source_edit from the historical 523/151 token starvation; did not duplicate that fix.
- Implemented progress-aware completion transport: prompt-progress + SSE keepalive request fields and current /slots next_token.n_decoded parsing.
- Added focused regression tests for current and legacy slot shapes plus progress-aware chat payload injection.
- Installed the liveness contract through the existing worker-05 prefill telemetry installation point without modifying shared runtime bootstrap.

IN_PROGRESS:
- Audit generic completion-pending telemetry and retry/terminal semantics for any remaining duplicate or misleading state.
- Audit malformed-output recovery, forced-tool termination, speculative decoding, and KV-cache settings for correctness-first behavior.
- Verify broader regression coverage available from repository workflows/tests.

ROOT_CAUSES_CONFIRMED:
- Tool/semantic completions use SSE aggregation but did not request llama.cpp prompt_progress or SSE pings, so long prompt preparation or other silent periods could reach the read-inactivity deadline despite a healthy server.
- Tool liveness used stale root slot.n_decoded; current llama.cpp reports generation progress at slot.next_token.n_decoded, causing real generation to look like processing-no-new-prompt or unknown progress.

DECISIONS_AND_EVIDENCE:
- Do not raise the 120s timeout. Make liveness depend on observable progress instead.
- Keep partial native tool deltas non-executable; the existing aggregator returns a completed message only after [DONE].
- Use llama.cpp request-level return_progress and sse_ping_interval and retain legacy slot counters as a compatibility fallback.
- Avoid modifying shared runtime_bootstrap; activate the worker-05 liveness contract through llama_prefill_telemetry_contract, which is already installed immediately after stream efficiency.
- Current generation_output_budget already contains regression logic for the old 523-token apply_source_edit starvation.

COMMITS_ALREADY_PUSHED:
- 4d629b06772124fb7867ed040ab194001dfcf406 fix(llama): add progress-aware completion liveness
- 5a4dadf2e5dfd8d9f7211728eb505dd0075a942c test(llama): cover progress-aware completion liveness
- f4253f2d5ab006ce4731ce2bdaeb34a810b6b714 fix(llama): install progress-aware completion liveness

TESTS_ALREADY_PASSING:
- Local pure contract checks: nested next_token decode, legacy slot fallback, progress/ping payload, install wrapper.
- No pull-request-triggered GitHub Actions run exists for the direct-main checkpoint; repository workflow coverage remains to be inspected separately.

NEXT_EXACT_ACTIONS:
1. Audit duplicate generic completion-pending telemetry and remove it if it conflicts with progress-aware liveness.
2. Audit malformed-output and forced-tool retry termination for identical-failure loops.
3. Audit speculative decoding and KV-cache correctness guardrails.
4. Rebase semantics on latest main, validate owned tests/contracts, and update this file.

FILES_CURRENTLY_RELEVANT:
- minecraft_mod_ai/model_adapters/llama_cpp_adapter.py
- minecraft_mod_ai/llama_stream_efficiency_contract.py
- minecraft_mod_ai/llama_server_hardware_policy.py
- minecraft_mod_ai/llama_finish_reason_contract.py
- minecraft_mod_ai/generation_output_budget.py
- minecraft_mod_ai/forced_tool_execution_contract.py
- minecraft_mod_ai/llama_prefill_telemetry_contract.py
- minecraft_mod_ai/llama_completion_liveness_contract.py
- tests/test_llama_completion_liveness_contract.py

KNOWN_CROSS_ROLE_DEPENDENCIES:
- none currently; shared runtime_bootstrap does not need to change.

UNRESOLVED:
- Generic completion-pending telemetry remains to audit for duplicate or misleading logs.
- Malformed-output/tool termination and speculative/KV correctness guardrails are not yet fully audited.
- Direct-main GitHub Actions coverage is not available through the commit workflow-run query.
