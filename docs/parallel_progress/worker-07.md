# Worker 07 — coder localization / mutation-target hardening

## Scope

Worker 07 owns repository-aware coder targeting, `apply_source_edit` localization safety, and repair-loop target stability. This checkpoint is intentionally limited to that boundary and does not change planner semantics, Minecraft API selection, or other workers' subsystems.

## Base

- synchronized base: `main@ffac1aa9a46c8e648360b1f53818eb5265181c91`
- branch policy: direct `main`; no worker branch

## Root causes found

1. `implementation_kind_boundary_contract.py` duplicated the complete fresh-target parser already owned by `progress_aware_tool_loop._fresh_owned_symbol_context`. The late wrapper therefore maintained a second copy of the same routing rule.
2. A task could carry `reuse_refs`/`component_refs`/`source_refs` while its production binding still said `reuse_action=fresh`. The old late wrapper accepted that contradiction as a mutation-ready new file, allowing source-body localization to be skipped.
3. `TargetMutationContext.merge` used field-wise fallback plus `self.is_new_file or other.is_new_file`. If localization moved from file A to file B, A's symbol/body/ranges could survive under B's path. Once a context became `is_new_file=True`, later repository evidence for an existing file could also leave the new-file bypass active.
4. A host-reserved fresh path could still be marked new even when the bounded initial exact-source page already contained that exact path. That can produce `READY` with no existing body inspection despite the file already being present.

## Changes

### `minecraft_mod_ai/implementation_kind_boundary_contract.py`

- Deleted the duplicate fresh-target parsing implementation and retained a thin compatibility delegation to the single host-owned resolver.
- Added a fail-closed fresh/reuse consistency guard. A prospective new-file context is rejected back to `NEED_FILE` when the evidence task or its fresh production binding already contains reuse/component/source references.
- Reuses the cached initial exact-source page from `ProjectIndex`; if the reserved fresh path itself is already present, it is treated as an existing target and body grounding is restored. Unrelated initial source cannot replace the reserved path.
- Hardened `TargetMutationContext.merge` at the existing late-runtime contract boundary:
  - a different target path replaces the localization context instead of mixing fields across files;
  - exact repository evidence for the same path can downgrade a prospective new file to an existing-file context, restoring source-body grounding;
  - same-target incremental file -> symbol -> body localization continues to accumulate normally.
- Reused the existing runtime wrapper marker instead of adding another repository scanner.

### `tests/test_worker07_target_context_hardening.py`

Added focused regressions for:

- an exact reserved path already present in the bounded initial source page is treated as existing, not new;
- incidental initial source cannot hijack a legitimate fresh reserved target;
- legitimate fresh target remains mutation-ready;
- task-level reuse evidence blocks fresh generation;
- binding-level source evidence blocks fresh generation;
- different-file localization cannot inherit stale symbol/body state;
- path-only retargeting resets prior body/symbol state;
- repository evidence can convert a prospective new target to an existing target;
- same-file hierarchical localization remains cumulative.

## Performance / clean-code impact

- No new repository walk, RAG query, index build, or model call was added.
- Removed a second O(number-of-bindings × anchors) fresh-target parser from the late wrapper path; the canonical host resolver remains the only parser.
- The new checks inspect only the already-present evidence-task payload, cached initial exact-source page, and mutation contexts.
- Cross-target resets prevent wasted repair attempts caused by stale body/symbol data being applied to a newly localized file.

## Verification

Pre-commit checks performed on the exact proposed files:

```text
python -m py_compile implementation_kind_boundary_contract.py test_worker07_target_context_hardening.py
PASS
```

Post-push repository CI/status must be checked against the resulting commit before Worker 07 is declared complete.

## Completion checklist

- [x] root cause localized
- [x] duplicate fresh-target implementation removed
- [x] fresh/reuse contradiction fails closed
- [x] exact existing reserved target cannot use the new-file bypass
- [x] cross-file context contamination blocked
- [x] existing-file evidence restores body-grounding semantics
- [x] focused regression tests added (9 cases)
- [x] no new repository-scan bottleneck introduced
- [x] changed-file syntax validation passed
- [ ] pushed commit CI/status verified
