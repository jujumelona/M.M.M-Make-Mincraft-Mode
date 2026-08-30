# Worker 09 — State + Contracts + Provenance

Status: IN_PROGRESS

Base main SHA: `763dffa290a07d17ce24b16329bd65aef0ce5bce`

## Diagnosed root causes

1. Proposal approval state and its integrity receipt can diverge. Both base `Proposal.approve()` and `CompleteProposal.approve()` validate a supplied hash, but an awaiting proposal deserialized/constructed without a stored `approval_hash` can become `approved` while still carrying an empty receipt.
2. Persisted base proposals can omit provenance hashes that core deserialization backfills from the current runtime. In particular, a missing capability manifest hash can silently bind stale saved state to the current process instead of proving which manifest the saved proposal actually used.
3. Design-alternative provenance accepts a caller-provided `evaluation_sha256` without checking that it identifies the exact comparative-evaluation payload; it also synthesizes a receipt when none was persisted. A stale/arbitrary digest can therefore be surfaced as selection provenance.

## Intended fixes

- Make approved state require and retain the exact approval integrity receipt for both base and complete proposals.
- Fail closed when persisted proposal authority/provenance fields are absent instead of silently rebinding them during resume/deserialization.
- Require an explicit design-evaluation receipt and validate it against canonical evaluation content, while keeping semantic selection requirements independent from hash integrity.
- Add focused regression tests proving stale/missing receipts cannot authorize resume/approval/design selection and proving a valid hash alone is not semantic correctness evidence.

## Scope inspected

- `minecraft_mod_ai/spec.py`
- `minecraft_mod_ai/complete_spec.py`
- `minecraft_mod_ai/proposal_deserialization_contract.py`
- `minecraft_mod_ai/design_resolution_provenance_contract.py`
- `minecraft_mod_ai/evidence_task_receipt_contract.py`
- `minecraft_mod_ai/work_graph_state_transition_contract.py`
- related proposal/revision/task tests

No cross-owner implementation changes planned.
