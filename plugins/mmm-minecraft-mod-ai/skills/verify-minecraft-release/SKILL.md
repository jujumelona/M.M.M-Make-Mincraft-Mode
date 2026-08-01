---
name: verify-minecraft-release
description: Enforce source, build, GameTest, JAR, provenance, and packaging gates before a release is labeled installable.
---

# Verify Minecraft Release

## activate_when
Use when a generated project is ready for validation, build, play-test preparation, JAR inspection, or release packaging.

## inputs
- Approved proposal and hash.
- Generated project path.
- Optional candidate JAR.
- Build and test receipts.

## required_rag
Use version-pinned build and GameTest evidence. Use `minecraft-dev` only for resolving source/JAR discrepancies.

## allowed_tools
- `mmm-local.run_static_validation`
- `mmm-local.run_gradle_build`
- `mmm-local.run_gametest`
- `mmm-local.inspect_jar`
- `mmm-local.package_release`

## output_schema
Return one release verdict: `VERIFIED`, `SOURCE_READY`, `FAILED`, or `BLOCKED`, with all report paths, hashes, exit codes, and missing gates.

## validators
- Source validation passes.
- Gradle clean build exits 0.
- GameTest exits 0.
- JAR contains valid `fabric.mod.json`, compiled classes, and required resources.
- JAR bytes do not change during validation.
- Release ZIP contains a manifest bound to the proposal hash.
- Runtime client/VLM/Mineflayer claims remain absent until compatible runtime tooling actually runs.

## retry_policy
Rerun a failed command only after a concrete repair. Maximum three repair cycles. Never rerun indefinitely or rewrite evidence.

## approval_required
Build, GameTest, and packaging require approval. Publishing is outside this skill and requires a separate explicit user action.

## forbidden
- Labeling source-only output as installable.
- Packaging a failed or uninspected JAR.
- Converting mock/static tests into runtime-playtest claims.
- Omitting failed logs or changing hashes after approval.

## exit_conditions
Exit only after every requested gate has an explicit result and the release label matches the weakest gate.
