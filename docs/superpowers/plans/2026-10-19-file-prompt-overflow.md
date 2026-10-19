# File Prompt Overflow Repair Plan

**Goal:** prevent the source author's inspected API context from overflowing the selected runtime while preserving exact signatures, requirements, failed source and compiler diagnostics.

**Evidence:** production at main 0ae8317 failed before the first coder completion: 49,422 exact input tokens exceeded the 32,768-token slot. The archive has 331 planner requests and no coder request. FileAPIEvidence serializes complete symbol index rows and every available member name, then repair messages append full API views and sources repeatedly.

## Task 1 — concise inspected contracts
- [x] Add failing regressions for repeated bytecode/index metadata and unbounded member-name catalogs.
- [x] Project API symbols to callable/type contracts; retain full host index and compile verification.
- [x] Verify and immediately commit/push the representation fix (`aa7c4db`).

## Task 2 — context-aware file-author recovery
- [x] Reproduce typed context overflow and repair-history growth before fixing.
- [x] Keep only current full source/diagnostic in repair requests; deduplicate inspected class contracts.
- [x] Recover measured context pressure by reducing supplemental API observations with strict size progress; never truncate requirements/source or consume compiler repairs for pre-inference rejections.
- [x] Independently review, verify the construction suite and inspect CI, commit and push results (`9b0dea4`).

Validation: 156 tests passed and 10 JDK-dependent tests skipped across the construction suite and selected existing llama context/boundary suites. Independent review findings about successive-repair contract loss and Java modifier loss were reproduced and fixed. Real MiMo/Fabric execution remains unverified; remote CI status is recorded separately from local results.

Constraints: main only, preserve selected target (26.2 in this log), no context-window increase, no disabled token checks, no external mod source reuse, no gameplay-success claim without actual execution.
