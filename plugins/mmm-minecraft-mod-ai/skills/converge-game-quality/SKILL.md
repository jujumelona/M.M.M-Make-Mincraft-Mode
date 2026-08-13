---
name: converge-game-quality
description: Close every request-derived Minecraft capability and quality requirement with fresh, independently checked source, build, runtime, visual, accessibility, performance, compatibility, provenance, and release evidence. Use after planning or during repair of small through game-scale mods when completion must be measured instead of asserted.
---

# Converge Game Quality

Treat scope as a graph of user requirements, not a checklist copied from a genre
example. Read the proposal's production contract first. For every requirement,
preserve its source text, dependent modules, required artifact types, validators,
evidence freshness rule, and blocking dependencies.

1. Find uncovered requirements before adding features. Do not add any capability
   unless the request or a real dependency requires it.
2. Use project RAG and exact target-version evidence to locate the affected code
   and interfaces. A search result, generated file, or model statement is not
   proof of implementation.
3. Repair the smallest coherent requirement slice, then regenerate its evidence.
   Prefer module-specific static checks and GameTests; add server, client,
   multiplayer, save/reload, visual, accessibility, latency, memory, TPS/FPS, or
   long-soak evidence only when the requirement makes that dimension relevant.
4. Bind each result to the requirement ID, proposal hash, source revision,
   artifact hash, validator identity, command or review receipt, and timestamp.
   Reject stale, self-certified, unbound, skipped, or merely planned evidence.
5. Re-read the quality status after every iteration. Continue over independent
   shards while progress is possible. Stop retrying one signature when it
   repeats, preserve the diagnostics, and surface the exact missing dependency
   or external environment instead of deleting functionality.
6. Release only when every required dimension passes. `built`, `packaged`, and
   `complete` are different states; a distributable archive with an unresolved
   requested capability is not release-ready.

## Runtime policy

```yaml
schema_version: mmm/skill-policy-v1
activate_when:
  - A planned Minecraft mod needs requirement-to-evidence coverage before implementation or release.
  - A generated mod has unresolved source, build, GameTest, runtime, visual, accessibility, performance, compatibility, provenance, or packaging quality.
  - A large production run must iterate over independent requirement shards without losing scope.
inputs:
  - approved proposal and its request-derived production contract
  - current source revision, artifact hashes and work-ledger receipts
  - exact target-version research and project RAG index
required_rag:
  - exact Minecraft, Fabric, Yarn, Java and selected dependency evidence
  - project source symbols, tests, assets, resources and dependency relationships
  - fresh validator receipts bound to requirement and artifact identities
stages:
  - planning
  - research
  - generation
  - quality
  - runtime
  - release
allowed_tools:
  - read_quality_contract
  - quality_status
  - search_code_rag
  - repair_project
  - java_diagnostics
  - java_workspace_symbols
  - blockbench_list_tools
  - blockbench_execute
  - run_static_validation
  - run_gradle_build
  - run_gametest
  - inspect_jar
  - runtime_status
validators:
  - requirement_traceability
  - quality_convergence
  - evidence_freshness
  - no_self_certification
  - exact_version_evidence
  - source_provenance
retry_policy:
  max_attempts: null
  strategy: Retrieve fresh evidence, repair one coherent unresolved slice, rerun its strongest validator, and stop on a repeated failure signature.
  stop_on_repeated_error_signature: true
  require_fresh_evidence: true
approval_required:
  writes: true
  runtime: true
  read_only_research: false
forbidden_actions:
  - Declare completion from source generation, a package file, a generic smoke test, or model judgment alone.
  - Replace a request-derived requirement with a genre template or delete it to make validation pass.
  - Reuse stale evidence after the proposal, source, dependency, tool, test, or artifact identity changes.
  - Let the implementation author be the only authority for semantic, visual, security, license, or runtime acceptance.
  - Repeat an unchanged failing action or hide an external blocker.
  - Publish while any required quality dimension remains unresolved or failed.
exit_conditions:
  success:
    - Every request-derived requirement and required quality dimension has fresh artifact-bound passing evidence.
    - The quality status is PASS and the release artifact matches the validated revision and dependency lock.
  blocked:
    - A named external runtime, credential, licensed input, human review, hardware target, or unavailable dependency prevents a required validator from running.
  failed:
    - A required validator fails after evidence-informed repairs or repeats the same failure signature.
```
