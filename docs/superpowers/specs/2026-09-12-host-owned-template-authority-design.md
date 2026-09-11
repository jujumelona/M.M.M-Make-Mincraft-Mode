# HOST-Owned Template Authority Design

## Goal

Make deterministic Minecraft/Fabric target facts a single HOST-owned authority so the model only decides semantic/content intent. Exhaustively audit production templates, remove only proven redundant/dead responsibilities, populate target-specific facts from evidence, and fail closed whenever a supported target lacks a complete executable binding.

## Non-negotiable constraints

- Work directly on `main`; do not create feature branches.
- Never delete a production template or field merely because it looks duplicated. Deletion requires producer/consumer/reachability evidence and a verified replacement where applicable.
- Never extrapolate target-specific Minecraft/Fabric facts across releases without evidence.
- A target is executable-supported only when every required deterministic fact and canonical leaf is admitted and evidence-backed.
- Unsupported or unreviewed combinations fail closed instead of falling back to a nearby target or model inference.
- Model-owned templates contain semantic/content decisions only; HOST owns compatibility, API spelling, target paths, dependency coordinates and executable lowering choices.

## Authority boundary

### HOST-owned deterministic facts

HOST owns, per executable target:

- Java/JVM requirement and Gradle/toolchain constraints.
- Loader, Fabric API, Loom and mappings coordinates.
- Maven repositories and Gradle configurations.
- Capability to concrete implementation/library/dependency mapping.
- Minecraft/Fabric class, method, field, registry and API symbols.
- Resource/data schemas and pack-format constraints.
- Artifact rules and target path/naming rules.
- Canonical leaf to executable implementation/template binding.
- Template admission/hash constraints.
- Replacements and compatibility constraints.
- Compile/runtime requirements that are target dependent.

The canonical target row is conceptually:

```text
target
 ├─ toolchain
 ├─ mappings
 ├─ loader
 ├─ libraries
 ├─ repositories
 ├─ capabilities
 │   └─ capability
 │       ├─ implementation
 │       ├─ dependency
 │       ├─ configuration
 │       ├─ symbols
 │       ├─ schemas
 │       └─ compatibility_constraints
 ├─ api_symbols
 ├─ schemas
 ├─ artifact_rules
 ├─ leaf_bindings
 └─ replacements
```

### Model-owned facts

The model may decide only facts that depend on user intent or creative/semantic design, including gameplay purpose, feature decomposition, content values, behavior, visual/audio intent, and selection among options explicitly admitted by HOST.

### Derived facts

Facts mechanically derivable from canonical HOST or model-owned values are computed in code. They are not requested from the model and are not duplicated as a second authority in templates.

## Template exhaustive audit

Every production template and every declared field is classified as exactly one of:

1. `HOST-owned`: remove from model output/input responsibility and inject/project from HOST.
2. `model-owned`: retain as a semantic decision.
3. `derived`: calculate deterministically in code.
4. `invalid/redundant`: delete after proving no unique production responsibility remains.

For each template, build a producer -> transform -> consumer graph and verify:

- no field has multiple competing authorities;
- no consumed field lacks a producer;
- no produced field lacks a consumer unless explicitly diagnostic;
- no later stage asks the model to rediscover an earlier decision;
- no target-specific API/dependency/schema fact is model-generated;
- every non-standalone production template is runtime reachable;
- every canonical artifact kind lowers to admitted executable leaves or an explicit generator handoff;
- schema, prompt, renderer placeholders and consumer contracts agree;
- fallbacks/defaults do not hide missing required model or HOST facts.

Deletion happens only after the graph proves the responsibility is dead or fully replaced and contract tests protect the replacement.

## Target-specific API epochs

Do not use one Java snippet across incompatible API epochs. HOST leaf bindings select an implementation appropriate to the target and mapping regime.

The item path already demonstrates a real boundary: Fabric documents pre-1.21.2 item registration separately from 1.21.2+, where item settings require a registry key. Current Fabric documentation also uses ResourceKey/setId in newer Mojang-mapped examples. These boundaries must be represented as HOST facts/bindings rather than mixed inside one generic template.

Each supported target therefore receives evidence-backed symbol/binding rows. If exact syntax cannot be independently established for a target, the affected leaf remains `not_reviewed`/`unsupported` and product readiness fails for that target.

## HOST completeness contract

`SUPPORTED_MINECRAFT_VERSIONS` is a product promise, not documentation. The HOST coverage/readiness audit must require every supported target to exist and every required canonical leaf to be `admitted` with a valid implementation/evidence record.

The audit must reject:

- missing supported target bundles;
- missing required leaf bindings;
- `not_reviewed` or `unsupported` required leaves;
- admitted bindings without valid implementation metadata/evidence;
- template hash/admission drift;
- missing required symbols/schemas/capabilities;
- model/template attempts to override HOST-owned deterministic facts.

## Consumer migration

Consumers must receive deterministic facts through `ResolvedVersionContext`/target context projections. Direct independent lookup, hardcoded target guesses, latest-target fallback, and model-produced compatibility facts are removed.

`version_template_context` (or its current canonical replacement) projects the complete deterministic domain required by templates/consumers rather than a small ecosystem subset.

## Evidence policy

Use primary/official sources where available: Mojang/Minecraft release metadata, Fabric documentation, Fabric example-mod branches/releases, Fabric Maven metadata and Gradle/Loom documentation. Repository evidence records must identify the target and fact they justify.

A nearby release is not evidence for another release. Shared facts may be deduplicated internally only when each target explicitly resolves to the reviewed shared fact/epoch.

## Verification

Changes are test-driven and committed in independently reviewable slices. Required gates include:

- template catalog/contract validation;
- producer-consumer/reachability checks;
- HOST support-matrix completeness;
- HOST catalog parsing/admission/integrity checks;
- target-specific lowering/rendering tests;
- generation/integration tests for all supported canonical artifact kinds;
- compile/GameTest/runtime gates where the repository CI provides them;
- final GitHub Actions inspection for the exact final `main` HEAD.

A workflow starting is not success. Completion requires the relevant workflows/jobs to reach successful conclusions, or an exact unresolved failure report if an external gate cannot be made green.

## Completion criteria

The migration is complete only when:

- deterministic target facts have one HOST authority;
- every supported target has a complete evidence-backed executable row;
- no model template requests deterministic compatibility/API facts;
- no duplicate/dead production responsibility remains;
- every required field has exactly one producer and an actual consumer;
- every required canonical leaf has an admitted target-specific executable binding;
- unsupported/unreviewed combinations fail closed;
- relevant tests and CI for the final `main` HEAD are green.
