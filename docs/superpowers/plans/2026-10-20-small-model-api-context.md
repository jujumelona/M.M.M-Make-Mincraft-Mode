# Small-model Java API context repair

**Goal:** Make the selected target's actual inheritance and generic contracts available to the small coder without irrelevant registry or accumulated dependency catalogs.

**Evidence:** User production-logs (7)(1).zip, revision 17072afb8fb0. Entity source retained fabricated hostile/noDespawn and Identifier overloads. Initial coder context was 38,503 characters after decode-repetition recovery; subsequent compiler repairs grew to 75,967 and 89,147. Classfile Signature attributes were discarded, protected super constructors filtered out, and incidental signature dependencies displaced required base classes.

**Constraints:** Preserve the user's selected Minecraft/Fabric/JDK version, full source, binding values and gameplay responsibilities. No external mod source reuse, fabricated APIs, silent stub acceptance, additional model or credential dependency. Compiler and runtime verification remain distinct.

1. [x] Add failing real-JDK bytecode-to-prompt tests and a minimized production input fixture. Cover generic constructors, intersection bounds, fields, method signatures, protected inheritance, root starvation and repair catalog growth.
2. [x] Preserve Signature attributes in jar_api_extractor.py; render Java generic contracts in construction_context.py through a dedicated java_generic_signature.py parser. Keep erased verifier identities unchanged and remap namespace references.
3. [x] Scope entity type evidence to actual superclass constructors and abstract obligations. Include protected contracts only for authorized inheritance. Keep every requested root ahead of optional signature dependencies.
4. [x] Project repair API observations against the current source and diagnostics. Preserve complete failed source and requirements, useful discovered contracts and overloads; omit unrelated API catalogs. Explicitly identify indexed missing names without suggesting a guessed replacement.
5. [x] Run new regressions, related construction/bytecode tests, broader suite and source checks. Record unavailable live MiMo/Minecraft verification accurately, then commit and push verified changes to main.

**Review focus:** Nested/array/wildcard generic signatures; namespace remapping; private versus protected access; import-only noise; preserving requirements and source through context recovery.

The existing task authorization covers implementation. Execution is inline in the fresh checkout, with independent review after verification.

Verification checkpoint: 124 related checks passed, 1 skipped; separate broad-failure/current replay passed 36 checks. Final report records interrupted whole-suite scan and Maven-blocked production E2E.
