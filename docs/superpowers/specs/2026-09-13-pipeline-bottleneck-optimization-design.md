# Pipeline Bottleneck Optimization Design

## Goal

Optimize the complete production pipeline as one critical-path system across runtime execution, CI, and shared infrastructure. Reduce avoidable serialization, duplicate expensive work, repeated setup, redundant validation, and unnecessary database/process/model/network round trips without weakening determinism, atomicity, fail-closed behavior, or production evidence requirements.

## Non-negotiable constraints

- Work directly on `main`; do not create feature branches.
- Optimization must be evidence-driven. A suspected bottleneck is changed only after its execution path and safety contract are identified.
- Preserve semantic ordering where later work depends on earlier model/research output.
- Preserve overlapping-path serialization and coarse-writer exclusion while allowing disjoint source mutations to remain concurrent.
- Preserve scheduler resource capacities, lease ownership, stage-global mutation safety, GPU exclusivity, and dependency visibility ordering.
- Do not trade correctness for lower wall-clock time. Any optimization that changes deterministic output, mutation atomicity, retry semantics, or evidence coverage is rejected.
- Production integration remains a reviewed/manual evidence gate; optimization may reuse shared contracts and caches but must not silently weaken its prerequisites.

## System model

Treat the pipeline as one graph:

```text
request
  -> planning/research
  -> work-graph construction
  -> scheduler admission/claim
  -> model | image GPU | CPU/IO workers
  -> source/artifact mutation
  -> shared index/checkpoint update
  -> verification/repair
  -> packaging/runtime evidence
  -> CI/audits
```

The optimization target is end-to-end critical-path latency and duplicated resource cost, not isolated microbenchmarks.

## Measurement contract

Before changing each hotspot, capture a reproducible baseline appropriate to that hotspot. Measurements should include, where applicable:

- scheduler claim SQL statement count and claim latency;
- number of ready-queue scans per successful/empty claim;
- active worker occupancy and idle capacity by resource lane;
- model/network/process call counts and repeated identical calls;
- executor/client/model-adapter construction counts;
- time spent waiting on coarse versus path-scoped locks;
- validation/checkpoint cache hits and misses;
- pytest slowest-test durations and shard balance;
- GitHub Actions workflow/job duration, duplicated setup/install work, and superseded-run cancellation;
- end-to-end production graph wall-clock time for deterministic fixtures where available.

Static audits are candidate generators, not proof of a performance defect. Runtime/CI changes require before/after evidence or an exact structural reduction such as removing a provably duplicate query/setup/gate.

## Axis A: Runtime critical path

### Scheduler admission

`DurableWorkLedger.claim_ready` is a primary hotspot because it is polled repeatedly and owns fairness, resource capacities, leases, and serial-stage admission.

Review and optimize the current flow in this order:

1. Count SQL operations for empty claims, successful claims, lease-renewal claims, and saturated lanes.
2. Verify whether the standalone active-serial-stage query changes the candidate set beyond the serial-stage `EXISTS` predicate already present in the ready query. If it is semantically redundant, remove it only after regression tests demonstrate identical admission behavior.
3. Avoid performing a complete `ready_node_id()` scan before `BEGIN IMMEDIATE` and then repeating the same scan inside the transaction when no maintenance is due. Replace duplicate probing with a cheaper existence/admission probe or a single transactional selection if concurrency semantics remain correct.
4. Preserve lane fairness and prevent saturated lanes from occupying executor workers.
5. Add query-count/behavior tests so later wrappers cannot reintroduce duplicate admission scans.

### Locks and mutation

Path-scoped source transactions are the default for independent files. Coarse project locks remain only for operations whose correctness genuinely requires whole-project exclusion.

Audit every production `project_write_lock` caller and classify it as:

- whole-project atomic operation: keep coarse lock;
- exact known path set: migrate to path-scoped locking;
- read/validation-only: remove write lock;
- wrapper around an already locked operation: remove duplicate outer lock after proving ownership.

Keep deadlock-free canonical path ordering and writer fairness.

### Expensive-call topology

Use the existing runtime-efficiency audit as the candidate inventory for:

- model/network/process calls inside loops;
- repeated identical expensive calls in one function;
- model/adapter/client construction during hot function execution;
- executor construction in loops;
- missing cache/pool reuse.

For each finding, classify it as dependency-ordered, retry-semantic, or independent. Only the independent class is batched/parallelized. Reuse long-lived model adapters, HTTP clients, subprocess/runtime metadata, and deterministic validation results when their lifecycle and invalidation key are explicit.

### Parallel work

Parallelism follows resource ownership rather than a global worker count:

- LLM concurrency follows the active backend/native slot contract.
- image GPU work respects shared-local-GPU exclusion.
- CPU/IO uses host-derived capacity while keeping stage-global mutators serialized per domain.
- commit/index visibility remains ordered before a node becomes dependency-visible as succeeded.
- disjoint file writes may overlap; overlapping file writes serialize before validation.

## Axis B: CI critical path

### Remove repeated environment/setup cost

Current CI jobs repeatedly perform checkout, Python setup, and editable dependency installation. Reduce this cost without collapsing independent failure domains by:

- standardizing setup versions and cache configuration;
- maximizing pip/cache reuse with dependency-lock/input keys;
- using one canonical reusable setup action/workflow only if it measurably reduces duplication and does not hide diagnostics;
- avoiding unnecessary `pip --upgrade` where the job does not depend on a newer installer;
- keeping Java/Node setup only in jobs that actually consume them.

### Eliminate duplicate validation

Build a test/gate ownership map across `ci.yml`, `full-debug-gate.yml`, runtime/liveness audits, template validation, and production integration.

For every command/test, assign exactly one role:

- fast universal correctness gate;
- domain contract gate;
- diagnostic audit;
- expensive integration/evidence gate.

Do not run the same compile/import/static audit repeatedly on the same push unless the second execution has a distinct environment/evidence purpose. Preserve full-debug evidence when its trigger scope is relevant, but reuse or avoid steps already proven by an equivalent required job.

### Change-aware execution

Introduce change classification only where dependency boundaries are explicit. Examples:

- docs-only changes skip runtime/model/test jobs that cannot be affected;
- workflow/audit changes run their own audit contracts;
- scheduler/orchestrator/source-patch changes always run concurrency/atomicity suites;
- template/HOST changes run template, planning, generation, and integrity contracts;
- packaging/runtime changes run packaging/runtime evidence gates.

The classifier must fail closed: unknown or cross-cutting changes select the broader gate set.

### Shard balance

Use existing pytest duration artifacts to rebalance `remaining-tests` by observed duration rather than only deterministic file partitioning if current shards are materially imbalanced. Dedicated safety-contract tests remain explicitly owned and excluded from generic shards.

## Axis C: Shared optimization infrastructure

### One performance evidence schema

Extend the runtime-efficiency/reporting infrastructure so one machine-readable artifact can record:

- candidate category and source location;
- baseline metric;
- optimized metric;
- semantic/safety contract protecting the change;
- relevant tests;
- regression budget/status.

Do not create a second competing profiler/audit system when the current runtime-efficiency audit can be extended.

### One change/gate ownership map

CI path selection and local verification should derive from the same ownership data where practical. Avoid maintaining separate handwritten lists that drift between workflows and test tooling.

### Regression budgets

Add budgets only for stable structural metrics, not noisy hosted-runner wall time. Preferred budgets include:

- maximum scheduler SQL operations for defined claim scenarios;
- no duplicate identical expensive call newly introduced on a hot path;
- no executor/client/model construction introduced inside prohibited loops;
- no new coarse project write lock around path-local transactions;
- no duplicate CI ownership of the same deterministic gate without an explicit exception.

Wall-clock measurements are recorded and compared, but fail-closed thresholds require enough stability to avoid flaky CI.

## Initial high-value hypotheses

The first implementation pass should test, not assume, these hypotheses:

1. Scheduler claims perform avoidable duplicate ready scans and/or a redundant active-serial-stage query.
2. CI repeats install/setup and compile/import/debug-audit work across jobs/workflows more than necessary.
3. Some runtime expensive calls or client/adapter constructors identified by `audit_runtime_efficiency.py` are reusable or batchable.
4. Remaining pytest shards may have avoidable duration imbalance despite deterministic sharding.
5. Residual coarse `project_write_lock` callers may serialize otherwise independent path-local mutations.

Each hypothesis is independently rejectable if measurement shows no meaningful benefit or correctness risk outweighs gain.

## Implementation order

1. Add/strengthen measurement and regression tests around scheduler claim behavior and SQL count.
2. Remove only proven scheduler duplicate work; rerun concurrency, lease, fairness, stage-serialization, and dependency-visibility tests.
3. Run the runtime-efficiency audit and inspect top actionable expensive-call/constructor/lock findings; optimize highest-impact safe items with focused tests.
4. Build CI gate-ownership inventory and eliminate exact duplicate setup/validation while preserving distinct evidence responsibilities.
5. Add fail-closed change classification for clearly isolated paths; unknown changes keep the broad gate set.
6. Rebalance test shards only from recorded duration evidence.
7. Run focused tests, full pytest/required audits as appropriate, then inspect GitHub Actions for the exact final `main` HEAD.

## Verification

Required regression coverage includes:

- scheduler fairness/resource-capacity tests;
- serial-stage exclusion tests;
- lease renewal/expiry/reclaim tests;
- shared GPU exclusion tests when affected;
- disjoint-path concurrency and overlapping-path serialization tests;
- index/checkpoint visibility ordering tests;
- runtime-efficiency audit parse/inventory regression tests;
- CI ownership/change-classifier tests;
- pytest shard completeness/no-duplication tests;
- package/import/static/template/integrity gates affected by changed files;
- exact final `main` GitHub Actions status inspection.

For performance claims, report both structural evidence and measured evidence. Do not state an end-to-end speedup when only a local operation count improved.

## Completion criteria

The optimization effort is complete when:

- the three axes have been audited against the actual production path;
- every changed bottleneck has a demonstrated baseline and preserved safety contract;
- avoidable scheduler/database scans are removed or justified;
- independent runtime work is not blocked by avoidable global serialization;
- expensive model/network/process/client setup is reused or batched wherever dependency semantics allow;
- CI has explicit gate ownership with unnecessary duplicate work removed;
- change-aware skipping is fail-closed and covered by tests;
- regression metrics prevent the same bottlenecks from silently returning;
- relevant local tests and audits pass;
- relevant GitHub Actions for the exact final `main` HEAD reach successful conclusions, or any external blocker is reported precisely rather than called complete.
