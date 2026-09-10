---
name: execute-complete-production
description: Execute an approved small or game-scale Minecraft proposal through resumable generation, evidence-driven repair,
  runtime and visual validation, and release packaging. Use only after the exact proposal is approved and preserve every request-derived
  capability rather than substituting a genre template.
---

```yaml
activate_when:
- An exact complete proposal is approved for implementation.
- A previously started production run must resume from its durable ledger.
stages:
- generation
- quality
- runtime
- release
inputs:
- approved complete proposal and production contract
- execution options and explicit external-runtime paths
- optional authorized existing Fabric source ZIP
required_rag:
- exact target-version Fabric and Yarn implementation evidence
- exact selected dependency versions, hashes and licenses
- project source plus prior build, runtime and quality receipts
allowed_tools:
- execute_complete_project
- read_quality_contract
- quality_status
- search_code_rag
- repair_project
- java_diagnostics
- run_static_validation
- run_gradle_build
- run_gametest
- inspect_jar
- runtime_status
- package_release
validators:
- proposal_identity
- transactional_writes
- full_build_gates
- external_quality_gates
- requirement_traceability
- quality_convergence
- evidence_freshness
- no_self_certification
retry_policy:
  max_attempts: null
  strategy: Resume unaffected shards, retrieve fresh diagnostics for one failed requirement slice, repair it, and invalidate
    all dependent evidence.
  stop_on_repeated_error_signature: true
  require_fresh_evidence: true
approval_required:
  writes: true
  runtime: true
  release: true
forbidden_actions:
- Substitute an unrequested capability, genre, or example-derived system for the approved request.
- Treat source generation, a successful compile, or package creation as full completion.
- Delete requested functionality, relax evidence gates, or silently fall back to placeholders.
- Run arbitrary shell code, unrestricted writes, unapproved dependencies, or destructive operations on a user world.
- Reuse receipts after their proposal, input, source, dependency, validator, or artifact identity changes.
exit_conditions:
  success:
  - Every request-derived capability and required quality dimension has a persisted fresh passing receipt.
  - The packaged artifact matches the validated source revision and exact dependency lock.
  blocked:
  - A named external binary, endpoint, credential, licensed input, EULA decision, hardware target, or human review is unavailable.
  failed:
  - A required validator fails after evidence-informed repairs or repeats the same failure signature.
```
