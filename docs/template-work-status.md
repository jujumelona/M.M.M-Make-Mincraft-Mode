# Small task template migration

Implemented and connected:

- Feature worksheet record layouts load from `minecraft_mod_ai/templates/feature/`.
- Each model call resolves one record for one declared concern. Completion, explicit
  inapplicability and missing-information blockage have distinct states.
- Authored records pass unchanged into worksheet assembly. Removed prose normalization,
  synthetic state/owner/test records and automatic inapplicability explanations.
- Unknown evidence identifiers and duplicate sections fail validation instead of being
  silently removed or merged. Repeated generated records stop without claiming completion.
- Existing research survives checkpoint migration; uncertified derived plans and progress
  are invalidated after checking checkpoint integrity.
- Minecraft task compilation reads artifact responsibility manifests. Removed game-specific
  task builder functions from `minecraft_template_steps.py`. Artifact profiles still supply
  the existing obligation classification; that classification has not yet been replaced.
- Distinct item and block model obligations retain distinct task names. Registry tasks
  retain registry-identity ownership anchors. Client-required profiles retain client tasks.
- The normal detailed-planning path persists each validated concern response before its
  next model call. Resume revalidates the saved response sequence, restores partial records
  and skips completed concerns. Its key binds the exact template, requirement/criterion
  context, evidence content and allowed evidence IDs; changed inputs start fresh work.
- Removed the requirement-wide batch cache, its generation entry point and its obsolete
  prose-output tests. Criteria now run independently at native model concurrency; worker
  record saves and coordinator state updates share a lock to preserve sibling progress.
- Removed the arbitrary 128-record cutoff. Explicit completion/inapplicability still needs
  a valid transition; repeated records, malformed outputs and unrecognized evidence fail.
- Transport timeouts, connection failures and interruptions retain accepted records without
  creating a permanent semantic blocker. Resume requires invoking the pipeline again; this
  does not promise an automatic retry or a completed real-model run.

- Missing-section repair now fills each exact criterion/section gap instead of placing an
  aggregate repair into criterion zero. Existing authored sections remain intact. Repairs
  use the same record checkpoint/replay path and validate the returned section before saving
  its assembled fragment. Removed the synthetic all-criteria repair prompt.

Latest verification: 41 focused tests pass across record execution/resume, authored criterion
contracts, concurrent checkpoints, worksheet assembly/repair and legacy checkpoint handling.
Old repair fixtures now supply authored records; explicit regressions reject prose output
and empty completion without silently rewriting or automatically retrying them.

Checkpoint for continuation: the baseline was `28f46c4c`; `b347fa95` adds normal-path record
resume and removes requirement batching. The subsequent repair change closes missing-section
resume and fixes criterion ownership. Public remote publication was blocked by automatic
approval review; these changes are locally committed pending destination/payload approval.

Remaining work (do not describe the whole migration as complete):

- Replace prompt parsing, research, reuse, translation, code planning, asset production,
  integration and final validation dispatch with their own small task manifests.
- Replace legacy capability-to-profile classification with explicit evidence-backed artifact
  mapping; infer neither a runtime owner from a model file nor state scope from a broad flag.
- Add evidence-backed feature decomposition and deterministic atomicity routing, distinguishing
  missing information from multiple responsibilities and preventing non-progress recursion.
- Connect record checkpoints to the other stages as their dispatch is migrated. Normal
  criterion generation and missing-section repair are connected.
- Narrow record input context to declared dependencies and retrieved evidence slices.
- Give every Minecraft responsibility a typed input/output contract and an individually
  executable proof predicate. Current manifests describe tasks; they do not certify semantic
  correctness, optionality or successful implementation by themselves.
- Complete the broader regression suite and execute a real model/target-loader build and
  gameplay run. Focused tests are not a substitute for those checks.
