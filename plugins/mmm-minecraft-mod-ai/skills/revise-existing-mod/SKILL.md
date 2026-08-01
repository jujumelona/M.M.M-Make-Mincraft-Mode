---
name: revise-existing-mod
description: Inspect an existing mod archive safely and produce a bounded, provenance-preserving revision plan.
---

# Revise Existing Mod

## activate_when
Use when the user provides an existing source ZIP or JAR and requests changes, migration, repair, or feature additions.

## inputs
- Workspace-local source ZIP or JAR.
- Requested change.
- Target remains Fabric 1.20.1 unless a validated migration profile exists.

## required_rag
Use `minecraft-dev` for JAR analysis, decompilation, remapping, and affected APIs. Use local RAG for build/metadata contracts.

## allowed_tools
- `mmm-local.inspect_existing_mod`
- `mmm-local.revise_plan`
- `mmm-local.search_project_rag`
- `minecraft-dev` read-only analysis tools

## output_schema
Return archive inventory, loader/version findings, source snapshot hash, affected subsystem graph, proposed revision, and explicit preservation constraints.

## validators
- Archive is inspected without executing code.
- Zip-slip, symlink, and size checks pass.
- Snapshot hash is bound into the proposal.
- JAR-only input is never described as source-editable.
- Existing files are not overwritten in place.

## retry_policy
If metadata is ambiguous, perform at most two additional targeted inspections. Block revision when loader/version cannot be established safely.

## approval_required
Inspection is read-only. Creating a revision candidate requires approval of the snapshot-bound proposal.

## forbidden
- Executing uploaded binaries.
- Decompiling or redistributing code without respecting its license.
- Writing into the original archive or project directory.
- Pretending that inventory analysis is a minimal semantic patch engine.

## exit_conditions
Exit with a revision candidate in a new workspace, or a precise block explaining why a safe revision cannot be made.
