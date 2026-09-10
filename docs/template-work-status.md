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
- Minecraft task compilation reads artifact responsibility manifests. Concurrent structural
  routing commits were integrated through `c9cc8c6b`; `44c945ee` fixes their remaining consumers:
  deleted capability profiles are no longer imported, the structural compiler is bound
  directly, semantic labels no longer choose topology, and dependencies use explicit refs.
- Distinct artifact responsibilities retain distinct task names and registry ownership
  anchors. Client tasks derive from declared structural artifacts.
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

Published continuation baseline: `71440f1a` adds normal-path record resume and removes
requirement batching; `ee60ac3a` closes missing-section resume and fixes criterion ownership.
These commits were published to main after user approval.

- Prompt extraction now executes `templates/prompt/workflow.yaml`: exact host capture,
  one goal, individual authored facts, external references, explicit constraints, genuine
  prompt ambiguities, delivery requirements, and scope classification. The combined `submit_prompt_state`
  model call and its duplicated Python schema were removed. Host assembly derives its
  schema from the task files and retains the existing host-owned research routes.
- Prompt task input schemas project only declared context. Both single-value and record
  tasks revalidate saved results. Initial prompt progress is checksum-protected and resumes
  through `prepare_planning_state`; partial extraction is never treated as a researched
  planning state. Changed original prompts and tampered checkpoints are rejected.

- Research extraction executes `templates/research/workflow.yaml` through `research_template_pipeline.py`:
  reference identity, reference research, system extraction, gameplay loop extraction, progression,
  content, visual, audio extraction, and evidence checking. Each step produces deterministic,
  resumable receipts with proof-predicates.

- Feature discovery and recursive decomposition are connected through `feature_template_pipeline.py`:
  candidate features are discovered via `feature/discover`, completed across detail steps, and
  gated by host-owned `feature/atomic_check` (10-point checklist). Non-atomic features recursively
  split via `feature/decompose` until every leaf is atomic.

- Reuse evaluation executes `templates/reuse/workflow.yaml` through `reuse_template_pipeline.py`:
  queries, official docs, examples, existing mods, repo/file/class/method/dependency search,
  license checks, compatibility checks, and adaptation/integration plans.

- Downstream stage pipelines are connected through `stage_template_pipeline.py`:
  `code/workflow.yaml` (9 steps), `asset/workflow.yaml` (6 steps), `integration/workflow.yaml`
  (6 steps), and `validation/workflow.yaml` (10 steps).
- Host semantic proof predicate evaluators genuinely evaluate all 31 templates across code, asset,
  integration, and validation stages, as well as research and reuse pipelines. Proof predicates
  strictly evaluate file ownership, field types, registry points, conflicts, dependency graph cycles,
  resource references, client-server authority, compiler diagnostics, test failures, requirement/feature
  traces, and calculate completion dynamically, rejecting fake passes.

- Full regression verification: 280+ tests pass across record execution/resume, prompt pipeline,
  research pipeline, feature decomposition, reuse evaluation, translation runtime, and semantic stage workflows.

