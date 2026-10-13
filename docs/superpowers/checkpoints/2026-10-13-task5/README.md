# 2026-10-13 small-model construction integration

This checkpoint preserves the original recovery material. The production modules
have since been restored or integrated on `main`; do **not** use the old
`new-files/` drafts as the runnable implementation.

## Current implementation contracts

- `construction_planner.py`: original requirement to observable behavior
  contracts and dependency tasks; each behavior keeps its parent requirement.
- `source_reuse_evidence.py`, `reuse_discovery.py`,
  `grounded_source_reuse.py`, `authored_source_rag.py`: distinct provider
  failure/no-results status; pinned license/proof provenance; complete Java
  method retrieval rather than mid-method excerpts.
- `authored_reuse_bridge.py`, `construction_source_symbols.py`: installed,
  adapted files and hashes, exact callable Java symbols, and donor call
  obligations. An old unproven source is a reference, never an installed API.
- `construction_generation.py`, `construction_author.py`,
  `construction_porting.py`, `construction_java.py`: first-pass bounded
  typed Java member authoring; explicit target-API port helpers; owner-specific
  Fabric registration. No fallback to a full raw-Java completion when a
  typed renderer or target symbol is missing.
- `construction_classpath.py`, `construction_runtime.py`: real Gradle
  compile-classpath preparation and JDK/Fabric symbol inspection before
  coder inference.
- `construction_journal.py`, `construction_progress.py`,
  `artifact_job_checkpoint.py`: pinned per-task decisions, resumable
  accepted nodes, dependency-hash-bound output reuse.
- `complete_orchestrator.py`: finish independent ready generation after an
  unsuccessful shard; preserve failures and blocked dependents; on a new
  explicit resumed run retry only previously failed generation work.
- `construction_coverage.py`: bind source, donor calls, port helpers and
  per-child requirements to execution receipts. Generated files and builds
  never constitute behavioral proof.

## Status and verification boundary

**No real MiMo 9B or Google Colab execution has been asserted for this
integration.** The owner requested to perform those runs after code changes.
A feature remains `RUNTIME_UNVERIFIED` until exact child trigger/effect/state
evidence is available. Source-only packaging is not a behavioral PASS.

Useful generated files:
- `.minecraft_ai/construction-failures.json`: failed shards, blocked
  dependents, independent completed work
- `.mmm/construction_nodes/`: hash-bound per-member progress
- `.mmm/construction_inference/`: model decision transcript
- `.mmm/construction_coverage/`: parent and child work statuses
- `.minecraft_ai/construction-behavior-coverage.json`: final per-behavior gate

The live Colab notebook pulls the repository's current `main` on setup.
Before interpreting its result, confirm the printed checked-out commit.
