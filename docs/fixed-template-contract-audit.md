# Fixed model response contracts

The supplied failure log identifies clean commit `3ebdec456c55fc0fb314fb25af552680b6c88489`.
Its first structured output failure is the `req_001` behavior worksheet: the model returns
an object under `specification`, but the response schema requires a string. The later
`PlanningStageError` propagates that same exception. This is a producer/consumer contract
mismatch; the recorded generation finishes after 790 output tokens.

## Changes

- All ten worksheet sections use fixed nested records from `planning_detail_slots.py`.
  Concern keys and record keys are required, extra keys are forbidden, and scalar field
  types are specified. Text remains the value type for descriptions and authored rules;
  the model cannot choose the surrounding protocol or substitute an opaque specification.
- Prompts embed the exact schema supplied to decoding. Host validation uses that same
  specification schema. Empty concerns require explicit, unique inapplicability reasons.
- Dependencies remain intact JSON, including newlines and trailing rules. The previous
  dictionary-to-prose conversion and character slicing are removed.
- Plan assembly and request-catalog handoff retain the canonical worksheet object.
- Legacy saved prose plans invalidate derived detail/coverage and regenerate, preserving
  research. Original checkpoint integrity is checked before rehashing; the refreshed
  checkpoint still passes through the normal state validator.
- Repair proposals, visual review, capability inference, coder summaries, and the model
  smoke response now supply explicit schemas. Repair proposals use the actual patcher's
  `sha256:` hash form and per-operation field layouts.
- Atomic coder summary aggregation preserves the final JSON envelope. Summary decoding
  precedes applying staged changes to the live project.
- Unreferenced `planner_field_worker.py`, `planner_hole_text.py`, and
  `semantic_source_partition.py` are removed. Repair's alternative top-level patch labels
  are removed from its model response path.

## Verification and limits

64 focused tests pass for worksheet schemas, DAG execution, handoff, checkpoint migration,
repair contracts, actual transactional patch application, and atomic coder aggregation.
The existing worksheet state-invariant test also passes separately (65 total).
An AST regression test requires an explicit schema on every direct JSON `generate_text`
call in the source tree. This checks call-site wiring, not semantic completeness of every
possible nested tool protocol.

The full Qwen/Colab run was not executed. The broader custom-module integration suite was
blocked while discovering external platform metadata. Qwen3.5's existing host-validated
JSON transport policy is unchanged; templates do not guarantee that every model response
will be valid or that authored gameplay semantics are correct.
