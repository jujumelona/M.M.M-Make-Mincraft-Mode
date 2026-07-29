---
name: publish-release
description: Package and optionally upload a validated JAR to reviewed distribution providers.
schema_version: mmm/skill-v2
---

activate_when:
  - A validated JAR and complete release receipts exist.
  - The user explicitly requested distribution or upload.

inputs:
  - validated JAR path and SHA-256
  - version, changelog and provider project ID
  - explicit provider token environment variable

required_rag:
  - current provider upload contract
  - project distribution policy and license inventory
  - final JAR validation and runtime receipts

allowed_tools:
  - inspect_jar
  - package_release
  - execute_complete_project

output_schema:
  - distribution metadata
  - source and binary bundle hashes
  - provider response and version ID

validators:
  - JAR bytes match the validated SHA-256
  - game version and loader are pinned
  - token is read only at upload time
  - upload endpoint is HTTPS and reviewed

retry_policy:
  max_attempts: 1
  strategy: no automatic duplicate publishing
  stop_on_repeated_error_signature: true

approval_required:
  writes: true
  runtime: true
  read_only_research: false

forbidden_actions:
  - publishing without explicit user intent
  - uploading an unvalidated or changed JAR
  - logging access tokens
  - silently creating duplicate releases

exit_conditions:
  success:
    - Provider returns a persisted publication receipt.
  blocked:
    - Token, project ID, endpoint or provider metadata is missing.
  failed:
    - Provider rejects the single approved upload attempt.
