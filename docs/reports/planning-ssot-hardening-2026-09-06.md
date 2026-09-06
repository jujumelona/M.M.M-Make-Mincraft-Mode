# Planning contract hardening

The current main already had a prompt-first planning pipeline. This change strengthens that pipeline instead of introducing another planner.

## Ownership

- `target_contract.py` owns provider fields including `source_api_family`, validation, mapping alias agreement, naming applicability, Java minimum, pack metadata validation and provider receipt serialization/deserialization.
- `platform_catalog.PlatformAdapter` is an import alias of `TargetContract`; it has no independent contract implementation.
- `target_grounding_contract` adapts shared validation errors to planner errors. Execution receipt reconstruction calls the shared decoder.
- The planning state remains authoritative when lowering a request catalog. A conflicting cached catalog is rejected.

## State invariants

Initial and restored states validate source spans, unique IDs, reason/route/source agreement, information needs before queries, evidence/unknown links, resolution receipts, decision citations, and exact coverage links. Unresolved user intent blocks requirements and plan readiness. Contradictions cannot be resolved by the generic scope policy. Unknown direct-fallback semantics raise instead of producing `custom.semantic_*`.

`PlanningPipeline` retains its last checkpoint for retries of the same original prompt. Callers may supply `existing_state` and a checkpoint callback for persistence. Finished requirement research is preserved; blocked retrieval can be retried. A ready state is reused without model calls. This does not introduce a UI for answering unresolved personal preferences.

## Detailed planning and coding

Every requirement fills a fixed evidence-linked engineering worksheet: behavior contract, state model, algorithm, integration, authority/network, persistence, resources/UI, failures/limits, reuse assessment, and verification. Missing sections and invented citation IDs are rejected. Worksheet and reuse references are carried through the request catalog, gaps, compiled tasks, compact coder context, and coder execution template.

Retrieved patterns remain source evidence, not proof that the final implementation works. Target-specific compilation, reuse eligibility checks and runtime verification remain necessary. A detailed worksheet alone does not prove that a small model interpreted every sentence correctly.

## Validation

Focused target/state suite: 38 passing tests, including native/mapped provider round trips, mapping alias rejection, user-only gates, contradiction route rejection, forged resolution/coverage, missing information needs, worksheet completeness, grounded handoff and resume.

Two broader-suite failures reproduced on the original main commit `5f339a109d0c55b26bda255c74aa92d7f2164d04`: an old `_expand_batches(prompt=...)` invocation and a coder fixture without target coordinates. The weather-compass test was updated because accepting synthetic unresolved semantics is intentionally no longer valid.

No live small-model inference, live retrieval corpus, Minecraft build or in-game end-to-end run was performed in this environment. CI results must be assessed separately at the final published commit.
