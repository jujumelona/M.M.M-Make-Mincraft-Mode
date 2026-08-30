# Worker 09 — State + Contracts + Provenance

Status: COMPLETE

Base main SHA: `763dffa290a07d17ce24b16329bd65aef0ce5bce`

Final focused verification SHA: `53927aab5defc1c8f90d640d1a019a44eb5f319f`

Pre-handoff moving `main` rechecked: `879a5e9690a387922181ce207e6bee6a623a611a`

## Root causes closed

1. Proposal approval state could diverge from its integrity receipt. Approved proposal state is fail-closed around the exact approval/provenance authority rather than accepting an unbound approval state.
2. Persisted proposal provenance could be rebound to current runtime values on deserialization/resume. Missing or stale provenance fails closed instead of silently surviving resume/revise boundaries.
3. Design-selection provenance could surface a caller-provided digest without proving it identifies the canonical comparative evaluation. Receipt identity/integrity is distinct from semantic selection correctness; evidence and requirement references remain independent semantic gates.
4. Durable work success/checkpoint state could persist receipt payloads without an independent receipt-integrity identity. Task/checkpoint receipt hashes are verified independently from artifact output hashes; legacy promotion occurs only when the legacy hash proves the same receipt payload; corrupt state is invalidated.
5. Public `ledger.task()` / `ledger.tasks()` views could expose a receipt tampered after runtime initialization, allowing a later replan consumer to treat raw DB state as authority. Successful task views now revalidate the receipt against its independent receipt hash; invalid state is reset and descendants are invalidated. Non-success states never expose stale receipts.
6. The exception-scoped execution-feedback path could read checkpoint `receipt_json` directly after initialization and therefore bypass receipt integrity. It now uses `cached_checkpoint(checkpoint_id, input_hash=...)` as the authoritative read boundary; tampered checkpoints are failed/cleared and cannot drive replan.
7. Portable work-receipt JSONL exports previously omitted the independent receipt identity. Export now carries `receipt_hash` separately from `output_hash` and audits integrity before writing.
8. Research shards could disagree on the declared full-corpus `fact_count`, with selection using `max(...)`, and a zero-fact corpus could bypass corpus-hash verification. All shards must agree on one full-corpus fact count, the exact deduplicated count must match it even at zero, and any present research corpus must match the approved corpus receipt.
9. Trajectory verifier chains treated a PASS as sufficient even when the same verification class also contained FAIL evidence. Producer and read-side qualification now fail closed on contradictory evidence; test/GameTest are one trust class for contradiction handling.
10. Remote trajectory production stamps `remote_format_version = "v3"`, while the read-side identity contract required an incompatible legacy schema string. Read-side identity now matches the actual v3 remote-store format while preserving the independent remote-record hash check.

## Hash semantics

Hashes in Worker 09 are identity/integrity receipts only. A matching digest never substitutes for semantic correctness, acceptance, executable verification, evidence authority, or approval state.

## Key implementation commits

- `7045bcac3b866e4726cc252b066048417a1e87b2` — proposal deserialization/approval authority.
- `96ae4c4d6278cd0f9354c1bfa6b8f7b07e39543e` — content-bound design-selection provenance.
- `7bc9a444b360f528c7b3c6ec3f5d6ce99e4e18be` — proposal/provenance regressions.
- `480a0f76edd24e46f75173f6c46558020e0190c7` — fail closed on contradictory verifier evidence at read time.
- `76f509346c3da97a7e45ce368715a4d18ca1a9f3` — mirror fail-closed verifier qualification at producer time.
- `4eb6d16039c62193156812bce4f698467da7cb9b` — preserve independent work receipt integrity in portable exports.
- `beccb5e405c2199981bb09ed6b53246e95e17fe5` — enforce research receipt consensus and zero-fact corpus integrity.
- `9f9ac49ce7f87b1cbeb3734763f263c66b54fcee` — research receipt consensus regression coverage.
- `9f5eb68349b6fe8ef4dccc490bd8fffd7223630e` — align remote trajectory identity with actual v3 store format.
- `c04522f84ff8a2be54be418567329941db3be0a5` — make task/task-page reads receipt-authoritative after initialization.
- `752ee884c0b93c09145b1112856dbc6b4b897c2c` — make exception-scoped replan checkpoint reads receipt-authoritative.
- `4e4dc8369dc29cc272068d4ec0b3b304c2c774d5` — live post-init tamper and replan authority regressions.
- `53927aab5defc1c8f90d640d1a019a44eb5f319f` — corrected retained Worker 09 final-verification workflow manifest.

## Files covered

Implementation/trust boundaries:

- `minecraft_mod_ai/spec.py`
- `minecraft_mod_ai/complete_spec.py`
- `minecraft_mod_ai/proposal_deserialization_contract.py`
- `minecraft_mod_ai/design_resolution_provenance_contract.py`
- `minecraft_mod_ai/evidence_task_receipt_contract.py`
- `minecraft_mod_ai/work_graph_state_transition_contract.py`
- `minecraft_mod_ai/work_graph_receipt_integrity_contract.py`
- `minecraft_mod_ai/research_ledger.py`
- `minecraft_mod_ai/trajectory_verification.py`
- `minecraft_mod_ai/trajectory_record_integrity.py`
- `minecraft_mod_ai/execution_feedback_replan_contract.py`
- `minecraft_mod_ai/execution_feedback_exception_scope_contract.py`
- proposal-store sharding/integrity surfaces exercised by Worker 09 regressions

Regression coverage:

- `tests/test_worker09_state_provenance_contract.py`
- `tests/test_worker09_proposal_store_integrity_efficiency.py`
- `tests/test_worker09_work_receipt_integrity.py`
- `tests/test_worker09_remaining_integrity.py`
- `tests/test_worker09_research_receipt_consensus.py`
- `tests/test_trajectory_store_v3.py`

## Final verification

GitHub Actions workflow `Worker 09 final verification`, run `33330803078`, on SHA `53927aab5defc1c8f90d640d1a019a44eb5f319f`:

- dependency/install: PASS
- `py_compile` on Worker 09 trust-boundary sources: PASS
- Ruff fatal/error rules (`F,E7,E9`): PASS (`All checks passed!`)
- focused Worker 09 regression suite: **30 passed / 30 total**
- proposal approval/provenance authority: PASS
- proposal-store sharding/read-once/cache integrity-performance: PASS
- task/checkpoint corruption + resume + portable export integrity: PASS
- post-initialization task receipt tamper invalidation: PASS
- post-initialization replan checkpoint tamper rejection: PASS
- verified current-exception checkpoint replan path: PASS
- contradictory trajectory verifier evidence, including re-hashed remote records: PASS
- local/sanitized-remote v3 trajectory identity: PASS
- research receipt consensus and zero-fact corpus integrity: PASS

Earlier focused runs exposed two real defects (remote-format identity mismatch and later live post-init receipt-authority bypasses) plus one workflow test-manifest typo. Completion was not accepted until the defects were fixed, the manifest was corrected, and the final 30-test run completed successfully.

## Concurrency / ownership

No branch was created and no force push was used. Changes were written directly to `main`. When concurrent workers advanced `main`, files were re-read and SHA-conditional updates were used rather than overwriting another worker's changes.

The retained `.github/workflows/worker09-final-verify.yml` provides a scoped regression gate for future changes to Worker 09 trust-boundary files. Repository-wide CI remains a separate shared integration concern and is not used as a waiver for Worker 09 correctness; the owned compile/lint/regression gate is green.

## Downstream handoff recheck

Before downstream handoff, `53927aab5defc1c8f90d640d1a019a44eb5f319f` was compared against moving `main` at `879a5e9690a387922181ce207e6bee6a623a611a`. The verification commit remains in `main` ancestry (`behind_by = 0`), and no Worker 09 implementation, regression-test, or retained workflow file changed after that verified snapshot. Only this progress document changed in Worker 09-owned paths. Key proposal, research-receipt, and live-receipt-authority implementation commits were also rechecked as ancestors of current `main`.

## Remaining

None in Worker 09 scope.

Worker 09 is ready for downstream Worker 13 dependency use.
