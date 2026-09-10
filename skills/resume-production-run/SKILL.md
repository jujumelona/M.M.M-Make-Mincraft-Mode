---
name: resume-production-run
description: Inspect and safely resume an interrupted M.M.M production run from its durable ledger and verified checkpoints.
  Use after a Colab disconnect, process restart, timeout, failed shard, or Project already exists error, and when retrying
  or continuing a large mod without overwriting valid outputs or starting duplicate work.
---

```yaml
activate_when:
- A durable run was interrupted or has failed, expired, or pending work.
- The requested output directory already exists.
stages:
- frontdoor
- generation
- quality
inputs:
- run name and configured workspace
- immutable proposal and graph identities
- current output hashes, task states, leases, and receipts
required_rag:
- project-local source and prior task receipts
- fresh diagnostics for each retried shard
allowed_tools:
- work_status
- work_tasks
- work_cancel_run
- work_resume_run
- execute_complete_project
- java_diagnostics
- run_static_validation
- run_gradle_build
- run_gametest
- inspect_jar
validators:
- proposal_identity
- checkpoint_integrity
- downstream_invalidation
- no_duplicate_run
- path_containment
- final_receipts
retry_policy:
  max_attempts: null
  strategy: Retry only ready failed shards after fresh diagnostics and recursive descendant invalidation.
  stop_on_repeated_error_signature: true
  require_fresh_evidence: true
approval_required:
  writes: true
  runtime: true
  release: false
forbidden_actions:
- Delete or overwrite an existing run because its directory name collides.
- Reuse a checkpoint whose input or output hash changed.
- Run two workers under the same unexpired lease.
- Claim completion without final validation and release receipts.
exit_conditions:
  success:
  - Every required task and receipt is valid and no runnable or failed task remains.
  blocked:
  - Run identity, approval, external runtime, or required evidence is unavailable.
  failed:
  - Ledger integrity or a safety boundary fails, or the retry budget is exhausted.
```
