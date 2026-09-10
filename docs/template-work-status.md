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

Latest verification: 35 focused tests pass across record execution/resume, authored criterion
contracts, concurrent checkpoints, worksheet assembly/repair and legacy checkpoint handling.
The old missing-section repair fixture now supplies authored records rather than prose.

Remaining work (do not describe the whole migration as complete):

- Replace prompt parsing, research, reuse, translation, code planning, asset production,
  integration and final validation dispatch with their own small task manifests.
- Replace legacy capability-to-profile classification with explicit evidence-backed artifact
  mapping; infer neither a runtime owner from a model file nor state scope from a broad flag.
- Add evidence-backed feature decomposition and deterministic atomicity routing, distinguishing
  missing information from multiple responsibilities and preventing non-progress recursion.
- Extend individual record checkpoints to the legacy missing-section repair path and the
  other stages as they are migrated. The normal criterion path is connected.
- Narrow record input context to declared dependencies and retrieved evidence slices.
- Give every Minecraft responsibility a typed input/output contract and an individually
  executable proof predicate. Current manifests describe tasks; they do not certify semantic
  correctness, optionality or successful implementation by themselves.
- Complete the broader regression suite and execute a real model/target-loader build and
  gameplay run. Focused tests are not a substitute for those checks.
