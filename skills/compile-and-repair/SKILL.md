---
name: compile-and-repair
description: Run JDT, Gradle and GameTest; apply finite exact minimal repairs from real diagnostics.
schema_version: mmm/skill-v2
---

activate_when:
  - Generated or modified Fabric source requires compilation or repair.
  - Minecraft Java 1.20.1, Fabric, Java 17 and Yarn 1.20.1+build.1 are pinned.

inputs:
  - approved proposal
  - project paths inside MMM_WORKSPACE
  - current source hashes and diagnostics

required_rag:
  - Fabric 1.20.1 official metadata
  - Yarn 1.20.1+build.1 symbols
  - exact optional dependency evidence
  - project-local source and prior receipts

allowed_tools:
  - java_diagnostics
  - java_workspace_symbols
  - search_code_rag
  - apply_source_patch
  - repair_project
  - run_gradle_build
  - run_gametest
  - inspect_jar

output_schema:
  - diagnostics and build evidence
  - exact before/after patch hashes
  - retry count and final status
  - unresolved gates

validators:
  - immutable approval and path containment
  - exact SHA-256 patch preconditions
  - transaction rollback on failure
  - loader, version and mappings consistency
  - no requested-functionality deletion

retry_policy:
  max_attempts: null
  strategy: progress-driven minimal-diff repair from fresh machine evidence only
  stop_on_repeated_error_signature: true

approval_required:
  writes: true
  runtime: true
  read_only_research: false

forbidden_actions:
  - silent fallback to another model or heuristic
  - arbitrary shell, script, browser code or unrestricted file access
  - mixing Fabric with Forge or another version
  - deleting functionality merely to make a build pass
  - modifying a real Minecraft world

exit_conditions:
  success:
    - JDT, Gradle, GameTest and JAR gates pass.
  blocked:
    - Required MCP, model, dependency, approval or runtime is unavailable.
  failed:
    - Retry limit or a safety boundary is reached.
