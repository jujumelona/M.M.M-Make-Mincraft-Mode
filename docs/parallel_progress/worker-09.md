# Worker 09 — State + Contracts + Provenance

Status: COMPLETE

Base main SHA: `763dffa290a07d17ce24b16329bd65aef0ce5bce`

Completion verified on Worker 09 integration SHA: `8b4484cea95c6a25ef617786c148a83763ca4da4`

## Root causes closed

1. Proposal approval state could diverge from its integrity receipt. Approved proposal state is now fail-closed around the exact approval/provenance authority rather than accepting an unbound approval state.
2. Persisted proposal provenance could be rebound to current runtime values on deserialization/resume. Missing or stale provenance now fails closed instead of silently surviving resume/revise boundaries.
3. Design-selection provenance could surface a caller-provided digest without proving it identifies the canonical comparative evaluation. Receipt identity/integrity is now distinct from semantic selection correctness.
4. Durable work success/checkpoint state could persist or export receipt payloads without an independent receipt-integrity identity. Task/checkpoint receipt hashes are verified independently from artifact output hashes; legacy promotion occurs only when the legacy hash actually proves the same receipt payload; corrupted state is invalidated on cache/resume boundaries; portable JSONL export includes the independent receipt hash.
5. Research shards could disagree on the declared full-corpus `fact_count`, with selection using `max(...)`, and a zero-fact corpus could bypass corpus-hash verification. All shards must now agree on one full-corpus fact count, the exact deduplicated count must match it even at zero, and any present research corpus must match the approved corpus receipt.
6. Trajectory verifier chains treated a PASS as sufficient even when the same verification class also contained FAIL evidence. Producer and read-side qualification now fail closed on contradictory evidence; test/GameTest are treated as one trust class for contradiction handling.
7. Remote trajectory production stamps `remote_format_version = "v3"`, while the read-side identity contract required an incompatible legacy schema string. The read-side identity contract now matches the actual v3 remote store format while preserving the independent remote-record hash check.

## Implementation commits from final hardening pass

- `480a0f76edd24e46f75173f6c46558020e0190c7` — fail closed on contradictory verifier evidence at read time.
- `76f509346c3da97a7e45ce368715a4d18ca1a9f3` — mirror fail-closed verifier qualification at producer time.
- `4eb6d16039c62193156812bce4f698467da7cb9b` — preserve independent work receipt integrity in portable exports.
- `e66b05d7d5dec40b963e1db1d6f29442a38b978d` — adversarial trajectory and receipt-export regression coverage.
- `beccb5e405c2199981bb09ed6b53246e95e17fe5` — enforce research receipt consensus and zero-fact corpus integrity.
- `9f9ac49ce7f87b1cbeb3734763f263c66b54fcee` — research receipt consensus regression coverage.
- `9f5eb68349b6fe8ef4dccc490bd8fffd7223630e` — align remote trajectory identity with actual v3 store format.
- `090decde89cfdbccd8a2aa576928af0e910d31b2` — remove temporary Worker 09 verification workflow after successful run.

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
- proposal-store sharding/integrity surfaces exercised by Worker 09 regressions

Regression coverage:

- `tests/test_worker09_state_provenance_contract.py`
- `tests/test_worker09_proposal_store_integrity_efficiency.py`
- `tests/test_worker09_work_receipt_integrity.py`
- `tests/test_worker09_remaining_integrity.py`
- `tests/test_worker09_research_receipt_consensus.py`
- `tests/test_trajectory_store_v3.py`

## Verification

GitHub Actions Worker 09 focused final verification run `33330315163` on SHA `8b4484cea95c6a25ef617786c148a83763ca4da4`:

- Python 3.11 dependency/install step: PASS
- `py_compile` on Worker 09 trust-boundary sources: PASS
- Ruff fatal/error rules (`F,E7,E9`) on implementation and regression files: PASS
- Focused Worker 09 regression suite: **27 passed / 27 total**
- Existing v3 local + sanitized-remote schema/identity path: PASS
- Adversarial contradictory-verifier path: PASS
- Research receipt consensus/zero-fact integrity path: PASS
- Proposal-store read-once/cache integrity-performance regressions: PASS
- Work task/checkpoint corruption/resume/export integrity regressions: PASS

The first focused run intentionally exposed the pre-existing remote-format contract mismatch; completion was not declared until that defect was fixed and the full focused suite passed on rerun.

## Hash semantics

Hashes in this scope are treated only as identity/integrity receipts. A matching digest never substitutes for semantic correctness, acceptance, executable verification, or authority state.

## Concurrency / ownership

No branch was created. Changes were written directly to `main`, and conflicts were handled by re-reading the latest file before retrying rather than overwriting another worker's changes. No unresolved Worker 09 cross-owner handoff remains.

The repository-wide CI runtime-mutation budget was temporarily above its shared baseline while multiple workers were landing changes concurrently; this prevented the standard aggregate test job from starting during one intermediate SHA. Worker 09 therefore used an isolated focused workflow to prove its owned regressions, then removed that temporary workflow after success. This note is not a correctness waiver for Worker 09; its owned compile/lint/regression gate is green.

## Remaining

None in Worker 09 scope.

Push verified on `origin/main`: YES. Worker 09 integration and cleanup commits were verified as ancestors of a later moving `main` head before this completion record was written.
