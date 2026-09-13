# CI Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `main-ci` into an authoritative gate that blocks correctness, structural-quality, bottleneck, spaghetti-code, dependency-cycle, duplication, and CI-topology regressions.

**Architecture:** Add a parent-delta AST quality audit with hard ceilings for new code, test it with deterministic fixture repositories, then wire it into the existing authoritative audit job. Keep the existing full pytest/runtime/Fabric/root-cause coverage and make the final `CI Gate` depend on every required lane.

**Tech Stack:** Python AST, pytest, GitHub Actions, existing Ruff/Vulture/runtime audits.

**Spec:** `docs/superpowers/specs/2026-09-14-ci-quality-gate-design.md`

## Global Constraints

- Use only `main`; never create a branch.
- Entire `tests` directory remains an authoritative test target.
- Existing debt is reported; regressions introduced by the current change fail.
- New functions/files have hard ceilings for pathological complexity/size.
- Every required lane must feed the final `CI Gate`.

---

### Task 1: Quality-regression auditor

**Files:**
- Create: `.github/scripts/audit_code_quality_regression.py`
- Create: `tests/test_ci_quality_regression_audit.py`

**Interfaces:**
- Consumes: repository root and optional base ref.
- Produces: JSON report and non-zero exit code when HEAD worsens protected metrics or introduces hard-ceiling violations.

- [ ] Write tests proving a newly complex function, new import cycle, new duplicate body, and new serial expensive loop are rejected while unchanged historical debt is tolerated.
- [ ] Run the focused test and verify RED because the auditor does not exist.
- [ ] Implement AST metrics, parent snapshot loading through `git show`, regression comparison, JSON output, and exit semantics.
- [ ] Run the focused test and verify GREEN.

### Task 2: CI topology self-check

**Files:**
- Modify: `.github/scripts/audit_code_quality_regression.py`
- Modify: `tests/test_ci_quality_regression_audit.py`

**Interfaces:**
- Consumes: `.github/workflows/main-ci.yml` text.
- Produces: violations when the full `tests` target or required authoritative audit commands disappear, or final `CI Gate` dependencies no longer cover required jobs.

- [ ] Add failing topology tests.
- [ ] Implement topology checks.
- [ ] Run focused tests to GREEN.

### Task 3: Authoritative CI integration

**Files:**
- Modify: `.github/workflows/main-ci.yml`

**Interfaces:**
- Consumes: quality auditor exit status/report.
- Produces: `CI Gate` cannot turn green when engineering-quality regression audit fails.

- [ ] Add `fetch-depth: 2` where required and run the new quality audit in the repository-wide audit lane.
- [ ] Upload its JSON report with existing audit artifacts.
- [ ] Preserve complete pytest, Python compatibility, Fabric integrity, root-cause, runtime concurrency, runtime efficiency, static analysis, packaging, and bootstrap checks.

### Task 4: Verification

- [ ] Run focused tests in Actions.
- [ ] Run the complete authoritative CI on the resulting `main` SHA.
- [ ] Inspect failed job logs instead of weakening gates.
- [ ] Fix only demonstrated root causes.
- [ ] Verify the final `CI Gate` and every required dependency on the exact final SHA before claiming completion.
