# Request-scoped structured design

The production log stops at `ACTION_UI_EFFECT_UNBOUND`. Its design contains one
currency variable, launch-only updates, and a purchase/trade/build UI with no
matching transaction. The planner sends all requested features to every bounded
worksheet and accepts one row for the entire request. The worksheet's documented
unit is one requirement, but its caller uses a whole mod as that unit.

Author each preserved source requirement in its own worksheet dependency graph.
Keep the original request as context and identify one active requirement as the
scope authority. Merge accepted records without a model-generated summary.
Share prior state declarations across requirements; identical declarations are
reused and conflicting definitions are rejected. Stamp accepted concern records
with their source requirement identity. UI identity, input derivation, effect
preflight and deterministic lowering must all require the same source identity
and trigger. The runtime owner of an already stamped record is a host fact; do
not ask a model to reassign it. Legacy unstamped plans retain legacy validation. UI interactions must still pass
the existing source-effect checks. Do not relabel a launch as a purchase or
manufacture transactions to pass a gate.

Approaches considered: relaxing the UI gate would accept missing behavior;
retrying the global worksheet retains the same loss of scope. Requirement-scoped
authoring fixes the producer's unit of work while retaining the typed compiler.
This is an implementation decision under the user's existing instruction to fix
the generator generally and commit to main.

Keep each model response's current four-record and field-size bounds. Accumulated
canonical records may exceed a single response's limit. Checkpoint completed
requirements with their shared-state inputs so an interrupted later requirement
does not regenerate accepted earlier ones. Existing plans remain readable.

Verification must exercise the real planner with bounded model-response fixtures,
more than four requested behaviors, shared state, conflicting definitions, and
checkpoint resume. Fixtures prove orchestration and compilation contracts, not
MiMo's ability to invent a complete mod. A live Colab/model/runtime run remains a
separate verification step.
