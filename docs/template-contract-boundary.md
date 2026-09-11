# Template authority and model boundaries

Production YAML lives only in `minecraft_mod_ai/templates/`. The repository-level
`templates/` mirror is removed. Fixtures must use an explicitly named fixture tree.

`template_contract_validation.validate_catalog` checks all catalog IDs, sequence
references, duplicate sequence steps, cycles, and model output schema closure.
`runtime_consumer_roots` binds coverage to host dispatch tables and direct template
entry points. Every template must be reachable from those entry points or declare
`standalone: true`. Five retained standalone descriptors explain their status in
their YAML. Adding a file does not automatically register it as consumed.

Catalog loading validates record/value model schemas and slot schemas before use.
Asset renderer input names must exactly match placeholders. Missing, null, empty,
wrongly typed and extra asset inputs fail validation; whitespace inside a placeholder
uses the same syntax in validation and rendering. Fabric render fields must be
declared by `requires` or an AI slot. Target paths and dependency/port bindings remain
host execution context, separate from renderer-only inputs.

Atomic slot schema repair keeps the supplied bounded evidence, the failed leaf,
its schema and one structured diagnostic (`code`, `artifact`, `path`, `expected`,
`actual`, `repair_scope`). It replaces the preceding repair message instead of
accumulating failures. Transport errors propagate; they are not misclassified as
schema failures. Exhaustion exposes the last diagnostic. Context serialization
rejects unknown objects and non-finite numbers instead of coercing them into text.
Unsupported texture kinds fail instead of acquiring an item sprite mold.

These checks do not make every existing descriptor an executable JSON Schema.
Research/stage/translation descriptors still have host-specific contract formats;
Minecraft responsibility templates still use their existing host execution gates.
Repository-wide typed IR migration, executable criterion coverage, and field-level
repair outside the atomic slot executor require separate integration work. This
patch does not establish real Gradle, GameTest or Minecraft runtime success.

Validation on this patch: the extended local selection reported 109 passed and two
existing failures. `test_prompt_template_pipeline` expects model-owned
`evidence_refs`, while the current response schema has three fields;
`test_four_axes_final_contract` expects literal `target_count = 1`, while the current
runner uses the cardinality record. Both failures reproduced with the original
catalog loader. Neither production behavior nor those tests were changed to hide
the mismatch. The dedicated template contract selection plus the latest upstream
Minecraft inventory/lowering checks reported 56 passed. Ruff and diff checks pass.
