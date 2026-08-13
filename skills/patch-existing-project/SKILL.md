---
name: patch-existing-project
description: Inspect, extract and modify an existing Fabric source project with hash-guarded transactional patches.
schema_version: mmm/skill-v2
---

activate_when:
  - The user supplied a Fabric source ZIP and requested modification.
  - The archive hash is bound into the approved complete proposal.

inputs:
  - source ZIP
  - approved complete proposal and archive SHA-256
  - requested module changes

required_rag:
  - extracted project source and metadata
  - Fabric 1.20.1 and Yarn symbols
  - prior diagnostics and build receipts

allowed_tools:
  - inspect_existing_mod
  - index_project_rag
  - search_code_rag
  - apply_source_patch
  - repair_project

output_schema:
  - archive inventory and source snapshot hash
  - patch operations with before/after SHA-256
  - preserved and changed paths
  - validation receipts

validators:
  - ZIP bomb, path traversal, symlink and credential rejection
  - exact archive and file hash preconditions
  - transactional rollback on any failed operation
  - existing functionality remains present

retry_policy:
  max_attempts: null
  strategy: minimal exact patch from new machine diagnostics
  stop_on_repeated_error_signature: true

approval_required:
  writes: true
  runtime: true
  read_only_research: false

forbidden_actions:
  - executing archive scripts, wrappers or JARs before review
  - broad rewrites without exact file hashes
  - deleting requested or pre-existing functionality to pass compilation

exit_conditions:
  success:
    - The modified source passes all selected build and runtime gates.
  blocked:
    - The ZIP is JAR-only, unsafe, incompatible or lacks sources.
  failed:
    - Patch rollback or validation fails after the retry limit.
