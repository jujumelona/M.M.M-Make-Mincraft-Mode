# Production API evidence failure — 2026-10-19

## Observed failure

The uploaded production archive was executed from clean main commit `e91bd8c44374fe07304dac783143763f4e9b3356`, targeting Minecraft 26.2 / Fabric API 0.159.0+26.2 / Java 25. The classpath preparation Gradle commands passed. Production generation failed in `AlienCombatHostileEntityRegistry.java`; the project build and gameplay verification were not reached.

The three code responses changed the same file:

1. Generated Yarn imports and `EntityType.Builder.create(MobCategory.HOSTILE, ...)` despite the selected official mappings.
2. Corrected imports, leaving the invalid enum constant and invented builder call.
3. Replaced HOSTILE with MONSTER, leaving `Builder.create(MobCategory,String)` unresolved.

The archive contains 334 model requests: 331 planner calls and 3 coder calls. This count describes this archive only and is not a new performance target.

## Host defects verified from prompts and source

- Initial evidence contained six CLASS declarations, including EntityType/MobCategory/Registry, but no callable signatures or nested Builder/EntityFactory contracts.
- Both repair requests retained the failed source and compiler diagnostics, but supplied eight unrelated JDK Builder/Registry owners. The first-eight limit was applied after scanning by simple class name in classpath insertion order.
- Therefore the model did not receive the relevant builder factory/build signatures before its repair allowance was exhausted. Increasing that allowance alone would not repair the evidence selection.

## Implemented repair

Commit `42050eb` shares an inspected member-evidence builder between first generation and compiler feedback. Qualified imports and nested type references resolve before ambiguous simple names. The initial prompt carries exact methods, fields, constructors, generic metadata and functional factory contracts. Private and client-only declarations remain excluded from common sources. Existing Java compile/checkpoint gates and selected-target contracts remain authoritative.

Regression tests reproduce the JDK namesake collision, wrong mapping import recovery, initial builder/factory signatures, deterministic ordering, nonexistent APIs and member prioritization. Direct Implementation Migration CI now executes the construction regression and compiler test files.

Validation for that commit: 28 tests passed; 1 real-JDK smoke test was skipped because the local runtime has no javac. Ruff passed. This is not evidence that MiMo generated the complete requested mod or that Minecraft gameplay succeeded.

## Follow-up validation

Further boundary checks and full-suite/Actions results are recorded in follow-up commits. The first independent reviewer could not run due usage limits. A fallback reviewer completed review and reproduced four additional boundary defects, which are handled with failing tests before fixes.
