# File Prompt Overflow Repair Plan

**Goal:** prevent the source author's inspected API context from overflowing the selected runtime while preserving exact signatures, requirements, failed source and compiler diagnostics.

**Evidence:** production at main 0ae8317 failed before the first coder completion: 49,422 exact input tokens exceeded the 32,768-token slot. The archive has 331 planner requests and no coder request. FileAPIEvidence serializes complete symbol index rows and every available member name, then repair messages append full API views and sources repeatedly.

## Task 1 — concise inspected contracts
- [ ] Add failing regressions for repeated bytecode/index metadata and unbounded member-name catalogs.
- [ ] Project API symbols to callable/type contracts; retain full host index and compile verification.
- [ ] Verify and immediately commit/push the representation fix.

## Task 2 — context-aware file-author recovery
- [ ] Reproduce typed context overflow and repair-history growth before fixing.
- [ ] Keep only current full source/diagnostic in repair requests; deduplicate inspected class contracts.
- [ ] Recover measured context pressure by reducing supplemental API observations with strict size progress; never truncate requirements/source or consume compiler repairs for pre-inference rejections.
- [ ] Independently review, verify the construction suite/CI, commit and push results.

Constraints: main only, preserve selected target (26.2 in this log), no context-window increase, no disabled token checks, no external mod source reuse, no gameplay-success claim without actual execution.
