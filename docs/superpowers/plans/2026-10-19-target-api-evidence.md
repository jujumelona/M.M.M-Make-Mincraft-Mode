# Target API Evidence Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the file author actionable, exact-target member signatures before generation and after compiler failures.

**Architecture:** Resolve qualified imports and nested owners before considering ambiguous simple names. Reuse one focused member-evidence builder for initial and diagnostic prompts, with source-set/access checks and deterministic ranking. Keep Java compilation and runtime acceptance gates intact.

**Tech Stack:** Python, pytest, inspected Java symbol index, existing Java compiler verifier.

**Spec:** Production failure supplied on 2026-10-19 at git e91bd8c: AlienCombatHostileEntityRegistry received class-only initial evidence, then eight unrelated JDK Builder/Registry owners during two repairs, and failed on EntityType.Builder.create(MobCategory,String).

## Global Constraints

- Work on main as previously requested; the fresh clone is clean and isolated from existing user work.
- Preserve the exact selected Minecraft/Fabric target and immutable repair acceptance contract.
- Use only inspected symbols and materialized project prerequisites; no external mod source reuse.
- Do not certify generated code or gameplay from lexical evidence; compile before checkpointing.

## Review Focus

- Many JDK Builder classes preceding the Minecraft nested owner must not displace exact imported owners.
- A mistyped Yarn import must resolve only a unique inspected Minecraft counterpart.
- Private or client-only nested types must not leak into common source prompts.
- An invented member must not appear as verified evidence.
- Large classes must preserve referenced fields, overloads and builder factory/build signatures ahead of incidental members.

### Task 1: Share exact-owner member evidence between initial generation and repairs

**Files:** Create `minecraft_mod_ai/construction_file_api_evidence.py`; modify `minecraft_mod_ai/construction_file_author.py`; create `tests/test_construction_file_api_evidence.py`.

**Interfaces:** Existing `_evidence(context, intent, *, target_path, owner, router)` adds `class_members`; `_diagnostic_api_evidence(context, source, diagnostic)` retains its list-of-owner-records output.

- [x] Write regressions for the production import/nested-owner collision, initial builder signatures, deterministic order, access/side boundaries and unknown APIs.
- [x] Run regressions against the current source and observe assertion failures.
- [x] Implement exact import/receiver resolution and a reusable inspected member evidence builder; retain field/method signatures and functional/generic metadata.
- [x] Verify focused tests and existing construction tests, then run the repository suite and report any environment limitations explicitly.
- [x] Commit and push the repair; inspect Actions for this commit.

Progress: Baseline: 17 passed, 1 skipped, 3 pre-existing failures in test_construction_direct_input_producers because its fake index lacks by_owner. Update the test fixture to the real index protocol alongside this fix. New production regressions fail before implementation as expected.

Independent fallback review completed: four Important findings reproduced by five failing tests, then fixed; 37 passed, 10 JDK skips. First two commits pushed as 42050eb and c668dcb. Full-suite interruption and runtime limits documented in the production report.

Completed broader construction suite: 117 passed, 10 JDK-dependent skips across 19 files in 66.37 seconds. Code repair pushed as 60ac28a; original full-suite run was interrupted and is explicitly not reported as passing. Selected-model/gameplay verification remains external to this local run.
