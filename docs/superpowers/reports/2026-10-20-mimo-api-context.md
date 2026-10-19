# MiMo entity generation failure: cause and verification

Source evidence: user `production-logs (7)(1).zip` and pasted runtime log, commit `17072afb8fb0`, selected Minecraft 26.2 / Fabric / JDK 25. The archived run contains 331 planner requests and four coder requests. This change does not claim a new live-model run.

## Confirmed causes

- `jar_api_extractor.py` discarded classfile `Signature` attributes. Constructor evidence lost `EntityType<? extends Mob>` and supplied only erased `EntityType`.
- File API evidence filtered protected super constructors even for a generated subclass. Dependencies of the first selected class could consume the discovery view before Mob/Monster contracts were shown.
- The entity-type leaf received entity registry Builder APIs despite a separate dependent registry artifact. MiMo emitted registry code in the entity class, including nonexistent `hostile`, `noDespawn`, and a wrong Identifier overload.
- Compiler discovery accumulated unrelated server/storage classes. The actual coder inputs after decode recovery were 38,503, 75,967 and 89,147 characters; the last two repairs left errors unresolved.

## Changes

Preserve and render generic class/field/method/constructor contracts while keeping erased verifier identities unchanged. Remap generic owners through the selected Tiny namespace. Show inspected entity superclass constructors and abstract obligations without registry Builder context. Admit protected contracts through actual inheritance, preserve used static imports, and keep required roots before supplemental dependencies. Reject registry Builder construction in the entity-type artifact before publication. Merge only relevant new diagnostic owners; attach unavailable member names to their own compiler location and remove stale absence facts.

The archived 75,967-character repair input becomes 51,454 characters with the same system instructions and host policy accounted for and the old discovery view preserved; a new entity-type task uses the scoped inheritance view instead. Source, gameplay requirements, binding values and target versions remain complete. No token cap or retry increase is used as the remedy.

The entity-handoff test still expected a registry-only route on the baseline commit; update it to require the already-established type-then-registry sequence.

## Verification

- Added a minimized actual MiMo request/source/diagnostic fixture and 19 regression cases. JDK compiler bytecode exercises generic bounds, arrays, fields, methods, namespace case handling, protected access, unused imports, static imports, root starvation, distinct diagnostic ownership and stale absence facts.
- Actual javac reproduction: a subclass constructor using `EntityType<?>` fails; using the generic constructor contract extracted from bytecode compiles. This uses a small representative Java API fixture, not a Minecraft 26.2 game run.
- Related author/API/entity/classpath/checkpoint tests: **124 passed, 1 skipped**. The skipped optional check requires a javac launcher; the new bytecode tests use the available JDK compiler module and did run.
- Replayed other broad-scan failures in isolation: **36 passed** after updating the preexisting stale entity route assertion. Baseline isolated replay had **1 failed, 4 passed**, with the failure being that stale assertion.
- Actual JAR integrity/descriptor/namespace regressions: **3 passed**.
- Ruff and changed-source compilation passed. Independent code review found no remaining Critical/Important issue after fixes. The review's diagnostic cross-association note was treated as a correctness issue and fixed with failing-then-passing tests.
- Whole-suite attempt displayed 1,004 passes, 8 failures and 10 skips before interruption around 24%; it is **not a green whole-suite result**. One failure was the subsequently fixed import-resolution regression, one was the baseline stale route test; the other isolated non-E2E replays passed. Suite-interaction failures are not certified resolved.
- Both production Gradle/GameTest E2E checks were attempted and failed during JDT owner preflight because Maven/Gradle dependency hosts were unreachable in this execution environment. No JAR, GameTest, runtime or live MiMo success is claimed.

## Remaining verification

Run the user-selected 26.2 pipeline with the actual MiMo-9B server and working Minecraft/Maven dependency access. Unit and bytecode regressions establish information delivery and scope correctness; they do not establish end-to-end generation success or the model's capability limit.
