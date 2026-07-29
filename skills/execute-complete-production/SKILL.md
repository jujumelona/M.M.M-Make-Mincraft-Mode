---
name: execute-complete-production
description: Execute one approved complete proposal through generation, repair, runtime, playtest, visual review and distribution.
schema_version: mmm/skill-v2
---

activate_when:
  - A complete proposal has an exact user-approved SHA-256.
  - The target is Minecraft Java 1.20.1 Fabric Java 17.

inputs:
  - approved complete proposal
  - execution options and explicit external-runtime paths
  - optional existing Fabric source ZIP

required_rag:
  - Fabric 1.20.1 APIs and Yarn 1.20.1+build.1 symbols
  - exact optional-library versions and licenses
  - project source and prior build/runtime receipts

allowed_tools:
  - execute_complete_project
  - java_diagnostics
  - run_gradle_build
  - run_gametest
  - inspect_jar
  - runtime_status

output_schema:
  - complete pipeline result
  - per-module receipts
  - build, runtime, playtest, visual and distribution evidence
  - unresolved gates

validators:
  - complete proposal hash and existing-input hash
  - source containment and transactional writes
  - JDT, Gradle, GameTest and JAR gates
  - required Blockbench, runtime, Mineflayer and visual gates

retry_policy:
  max_attempts: 3
  strategy: bounded exact-patch repair from new diagnostics
  stop_on_repeated_error_signature: true

approval_required:
  writes: true
  runtime: true
  read_only_research: false

forbidden_actions:
  - silent fallback or capability deletion
  - treating source generation as runtime completion
  - arbitrary shell, scripts or unrestricted file writes
  - modifying a real user world

exit_conditions:
  success:
    - Every requested and configured gate has a persisted passing receipt.
  blocked:
    - An external binary, endpoint, token, EULA approval or runtime is unavailable.
  failed:
    - Validation or the finite repair budget fails.
