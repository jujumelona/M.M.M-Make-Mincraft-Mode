Player actions behind eligibility conditions
===========================================

Observed failure
----------------

The uploaded October 25 run used main ``1d4b21c`` and discovered 112 common
Fabric callbacks. Its trade behavior retained the eligibility condition
``Player has accumulated money and wants to build or upgrade a spacecraft``
as its trigger, and the requested player transaction in its effects. MiMo
returned ``SOURCE_REQUIRED`` for the hook. No existing authored occurrence
method matched, so the host reached its input-producer path.

The target-ABI preflight also required the trigger to match a small positive
list of explicit control verbs. It rejected this behavior before activation
planning or source generation. The original player action had not disappeared;
the host was treating an eligibility condition as the only possible evidence
of that action. This differs from the previously repaired source/bytecode
authority and generated-source handoff defects.

Implementation
--------------

ABI eligibility and activation planning are separate. Existing explicit-control
plans keep their original decision channel and receipt bytes. Other behavior
wording receives a bounded semantic choice over the complete original trigger,
intent, effects and source quote. A voluntary player action may be expressed
in any of these fields; adding a gameplay-specific verb or noun to a positive
allowlist is unnecessary.

The host accepts only a quote from the chosen original field. It binds that
evidence to the canonical hash of the entire behavior and includes it in the
source recipe. Installation, replacement and final source reading replay the
same checks without another model call. A changed behavior, misattributed quote
or modified recipe cannot silently refresh that evidence. The receipt also
records activation scope across all four original fields. Structural modality
checks use the quoted action's own lexical anchor, rather than a fixed list
of gameplay verbs, and preserve the original clause and its offsets for replay.
Automatic prerequisites and consequences of another operation do not prohibit
the requested voluntary action. Automatic or prohibited activation of that
action, including restrictions in another original field, remains a veto.
A planner may decline a passive occurrence, leaving it for its actual runtime
source owner.

The behavior remains unchanged. Its original trigger still goes to the
separate predicate author, and its original effects still go to effect authoring.
The existing host renderer creates the real form, validated server transport,
Fabric event and startup entrypoints. The shared missing-occurrence exception
lives in a dependency-free module and remains exported from
``source_owned_hook``; the evidence planner does not join the authoring engine's
existing import cycle.

Exact quotes and hashes establish provenance, not the semantic correctness of
a language model's interpretation. An accepted action plan is not a gameplay
receipt. The generated source must still pass compilation, final attribution,
effect checks and the requested runtime gates.

Verification
------------

The exact uploaded behavior and previously unlisted English/Korean actions
failed against the old code before the change. Regression coverage exercises
real producer rendering, installation and replay, immutable original conditions,
quote-field attribution, changed-behavior evidence, tampered recipes and scope
assessments, automatic prerequisites/consequences, and activation restrictions
in other original fields. Prohibition checks do not require enumerating the
action's verb. A separate host regression reaches the same
missing-source branch using a different gameplay theme.

An exact-version probe replays the uploaded ``SOURCE_REQUIRED`` decision with
the real 164,408-symbol Minecraft 26.2/Fabric 0.159.0+26.2 index and Loader 0.19.5.
Baseline code fails at the same target check with the same behavior identity
and reason. Fixed code compiles seven common sources and one client source,
passes five companion sources into final attribution and checks 108 symbols.
Two invocations through the generated server action reach the real registered
Fabric event: an ineligible probe actor cannot apply the effect, and an eligible
one can. An undeclared client field is rejected before the effect.

This probe supplies controlled neutral predicate/effect bodies at the external
authoring boundary and a compiler-probe actor. It is not a live MiMo replay,
the complete trade implementation or a running Minecraft player/server.

The existing v1 recipe and all five generated producer files were byte-identical
to baseline. The normal publisher regenerated 43 HOST contexts, with only
authority hashes and context identifiers changing. Independent publication
must reproduce both metadata files byte-identically. Full-suite and hosted-CI
results are recorded in the associated commit and execution report. The final
focused group passed 173 tests after the review corrections, including isolated
MCP scope, published catalogs, the prior declaration-authority repairs and
generated-source handoff.
