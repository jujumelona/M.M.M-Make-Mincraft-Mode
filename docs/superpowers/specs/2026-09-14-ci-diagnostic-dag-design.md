# CI Diagnostic DAG Design

## Goal
Turn CI into a fast fail-closed diagnostic system that simultaneously reduces wall-clock feedback time and prevents new architectural spaghetti, liveness hazards, hidden concurrency ownership, and performance regressions.

## Constraints
- Work directly on `main`; do not create branches.
- Preserve existing authoritative checks while consolidating duplicated scanner implementations.
- Existing technical debt is inventoried; regressions fail. New code is subject to hard ceilings.
- Expensive integration/runtime checks must not hide independent static/architectural failures.
- CI evidence must remain available as artifacts even on failure.

## Architecture
GitHub Actions remains the orchestration layer. Python analyzers own diagnostic semantics. `main-ci.yml` becomes a DAG of independent jobs: fast static checks, architecture/quality checks, runtime-safety checks, duration-balanced Python test shards, compatibility checks, integration/runtime checks, and a final aggregate gate.

A shared source-analysis core provides AST/dependency facts to quality and runtime-safety gates. Workflow YAML must not contain independent inline implementations of source scanners; standalone workflows call the same Python analyzers used by the main CI.

## Test sharding
`tools/ci_test_shard.py` gains deterministic duration-aware LPT partitioning. Historical duration data is used when available; missing durations use a deterministic fallback estimate and stable path tie-breaking. Every discovered test file belongs to exactly one shard and no shard overlaps another. Hash sharding remains available as a fallback for callers without timing data.

The CI test matrix runs shards independently and uploads JUnit, diagnostic logs, and timing data. The design targets lower critical-path time rather than merely equal test counts.

## Quality ratchet
The existing baseline-relative quality audit remains authoritative and is extended with cognitive/deep-control-flow pressure, module coupling/fan-out, dependency SCC growth, mutable module-state introduction, broad exception swallowing, and side-effect/resource ownership indicators where they can be detected with low false-positive rates.

Existing debt does not fail solely for existing. A worsening metric fails. New functions/files/modules use explicit hard ceilings. New import cycles, duplicate bodies, serial expensive loops, and unreviewed resource ownership remain fail-closed.

## Runtime safety
Executor ownership remains centralized. The same ownership model is extended only where semantics are reliable enough to gate: direct lock/process ownership and timeout/liveness hazards are inventoried and regression-gated rather than duplicated across ad-hoc scanners.

Dynamic tests verify distinct-resource concurrency, same-resource serialization, timeout/cancellation behavior, worker cleanup, and absence of leaked resources through production call paths where fixtures already exist.

## CI profiling
Each major diagnostic/test phase emits machine-readable timing evidence. The aggregate report identifies shard balance, slowest tests/phases, total runner time where measurable, and the observed CI critical path. Timing collection itself must not change pass/fail semantics except for explicit baseline performance gates.

## Workflow consolidation
Standalone audit workflows remain useful for targeted dispatch/path filters, but become thin wrappers around authoritative Python scripts. Production-source path filters must cover the code the scanner claims to protect. In particular liveness checks must run when `minecraft_mod_ai/**/*.py` changes.

## Failure behavior
Independent diagnostic jobs run in parallel so one failure does not erase evidence from unrelated checks. Expensive dependent work may be gated behind prerequisites where doing so saves substantial cost without losing independent diagnostics. The final CI gate fails unless all required jobs/shards succeed.

## Verification
The implementation is complete only when analyzer unit tests pass; synthetic fixtures prove each new regression gate fails when expected; shard union equals the discovered test set with zero overlap; duration balancing is deterministic; concurrency safety tests pass; workflow scanner path coverage is tested; and the resulting `main` CI reaches green after the changes.