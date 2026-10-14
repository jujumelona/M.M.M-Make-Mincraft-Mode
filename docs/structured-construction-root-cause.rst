Structured construction: native grammar loss
===========================================

Follow-up: discovery must start from implementation purpose
----------------------------------------------------------

The supplied console transcript ``20261014-130354`` runs ``095f1534a33f``.
It records 101 body decisions without one accepted operation node: 93 symbol
lookups, five binding reads and three source reads. Two registry bodies keep
changing invented search strings. The first content node eventually fails
with ``ARTIFACT_GRAPH_WORK_TIMEOUT``; later background inference does not turn
that failed node into successful progress.

The old body loop required a literal API/member substring before it could
return a candidate. Blocking repeated queries still allowed a new invented
string on every turn. That lookup criterion cannot discover an unknown API
from the function the author needs to implement.

Construction now ranks actual source/type evidence from the behavior and a
concrete operation purpose. It exposes the real package/type hierarchy,
actual class declarations, member signatures and input/output type relations.
The native author selects among returned branches and signatures by meaning;
name overlap is an ordering hint rather than an admission requirement.
Zero-overlap candidates and every source tail remain reachable through literal
pages. Existing CPU dense policy is preserved; hierarchy selection works with
the existing native author without loading a new CPU embedding model.

An operation goal includes its exact graph slot. Evidence reads serve that
goal until the node is written; a fresh guessed search is no longer a primary
body action. The selected target API fixes its symbol, result and receiver/
argument count, including signatures above the display page size. Its missing
inputs become concrete next slots with type/mode obligations. Existing named
values and callback signatures are bound to actual declarations. Complete
source reads, assigned bodies, donor obligations and final semantic checks
remain part of authoring. Resume persists selectors and the unfinished goal,
then re-resolves them against inspected inputs.

Ranking prepares immutable owner/source terms once, scores members lazily for
displayed owners and retains one active owner ordering. Source linkage uses
maximum relevance per owner so repeated imports cannot win by popularity.
These are static design and code findings. Syntax/diff inspection and an
independent static review do not certify a new generation, build or gameplay
run; no such run was performed under the user's standing instruction.

Observed failure
----------------

The supplied ``production-logs (3)(1).zip`` and console transcript were produced
from ``82083c95f95a`` against Minecraft 26.2. They contain two failed work nodes:

* ``generate-content-00000000``: the exact returned data was
  ``{"action":"symbols","query":"EntityType extends Entity"}``, which violates
  the query's closed lexical schema.
* ``generate-content-00000003``: a worker exhausted its deadline while waiting
  for shared model capacity. Another inference was still emitting a repeated
  ``EntityType$Builder`` sequence inside a query. That partial query had grown
  past 45,000 characters although its schema declared ``maxLength: 240``.

Gradle classpath inspection completed successfully in this run. The attached
failure does not identify a Gradle lock or Java compilation as its first cause.

Cause in the production path
----------------------------

``llama_server_hardware_policy._server_payload`` copied the raw host schema
directly into ``json_schema``. Native-safe projection existed only in an
optional tuning wrapper and was therefore not an invariant of direct
production calls.

Construction's trie-complement character classes used Python ``re.escape``.
The actual requests consequently contain class escapes such as ``\~`` and
``\.`` that the pinned native compiler does not support. In addition, the
failed class-header query pattern has 111 nested groups. The pinned compiler's
maximum is 100.

The upstream compiler handles an unsupported pattern by accepting an arbitrary
string. It visits ``pattern`` before its string-length branch, so that fallback
also loses ``maxLength``. This explains how a nominally bounded data page can
emit a forbidden query or hold the model while generating repeated text.
The supplied logs do not include the server's compiler warning or its binary
revision; this server behavior was traced in the project's pinned source,
not observed in a new live inference.

Primary source: ``ggml-org/llama.cpp`` at
``d81235049384534c167caea52b85a694f6103d14``,
``common/json-schema-to-grammar.cpp``:
``gbnf_escape_length``, ``_visit_pattern``, ``MAX_PATTERN_DEPTH`` and the
``KIND_STRING`` branch of ``visit``.

The lookup result also put ``com.sun.jna.platform.win32.WinError`` fields ahead
of the exact ``EntityType`` class. A class-header author therefore received
eight irrelevant field/method records and no inheritance candidates despite
the type being in the inspected inventory.

Change
------

* Every direct JSON payload now goes through the same native-safe projection
  before output budgeting and transport. The original host schema remains the
  response-admission contract. Optional tuning no longer owns this invariant.
* Construction character classes use only native-supported escapes. Deep trie
  tails are flattened into alternatives; no excluded query, name or length
  bound is removed to satisfy the compiler's group-depth limit.
  Symbol selectors also exclude raw JSON quotes and backslashes, since the
  native pattern compiler emits class atoms directly inside quoted values.
* Projection rejects over-deep patterns locally so finite length bounds survive
  for other bounded schemas as well. Strict data requests cannot disable their
  native schema through legacy replay metadata.
  Supported integer minimum/maximum bounds survive projection, including the
  Java array-dimension contract of 0 through 255.
* Exact symbol/type matches lead the first lookup page, followed by exact
  member names and owner matches. All substring results remain paginated in a
  deterministic order.
* The construction protocol changes so old inference transcripts and page
  cursors cannot drift into the new decisions. Declaration-bound operation
  graphs and completed artifacts retain their existing checkpoint identities.

No timeout extension, hidden retry, prompt-specific gameplay patch or removal
of requested behavior is part of this change.

Verification scope
------------------

Changed Python files were parsed with ``ast.parse`` and the changes were checked
with ``git diff --check``. Tests and model inference were not run, following the
user's standing instruction. These checks do not certify a completed MiMo
generation, Minecraft build or in-game behavior.

Follow-up: declaration cursors and prompt grammar growth
-------------------------------------------------------

The subsequent ``production-logs (4)(1).zip`` reports execution of ``325ef05d733e``.
The observed failures are now:

* Exact input accounting reports 32,773 input tokens for a 32,768-token slot.
* A member planner selects declaration offset zero twice in the same phase
  and revision, triggering ``CONSTRUCTION_AUTHOR_NO_PROGRESS``.

The declaration reader already rejected repeated pages, but ``member_schema``
still offered an arbitrary nonnegative declaration offset after the first read.
The operation-author schema had the same mismatch. A shared host cursor now
offers only the next unread declaration page, using a literal offset. The
action disappears once those pages are exhausted and becomes available again
when accepted progress changes the phase/revision. Every declaration remains
reachable; the host's final repeat and offset checks remain in place.

The final completed scout class-header request before the overflow contained
a 37,761-character query pattern. Its final system message was 40,578 characters,
while the actual task/state user message was 6,144 characters. The pattern
is a native trie complement that excludes consumed selectors, not gameplay
instructions. ``generate_structured_data`` copied it into a textual schema in
the model prompt, causing the input to grow with lookup bookkeeping.

Native structured-data prompts now contain the decision fields, exact choices and
length/range bounds as a schema outline. Native lexical productions stay in
the real response schema passed to the sampler and exact host validator. No
schema constraint, requested behavior or accepted declaration is removed.
The prompt-only projection walks schema positions so a field or literal
object named ``pattern`` is preserved.
Transformers adapters retain the full textual schema because their generation
path has no native schema sampler.

The construction protocol is advanced to separate previous inference journals
and lookup cursors from the corrected decisions. Declaration-bound accepted
operation graphs retain their existing identities. The context limit,
timeouts, retry policy and exact live-context guard are unchanged.

This follow-up receives syntax/diff checks and static review only. No tests,
model inference, Minecraft build or gameplay run was executed for it.

Follow-up: phase decisions must reach authored work
-------------------------------------------------

The in-progress console log ``20261014-090235`` runs ``37b10f9fb03f``.
Its first content node fails in the class-header phase with
``CONSTRUCTION_RENDERER_EXTENSION_REQUIRED`` even though the selected reason
says no further superclass or interfaces are needed. The generic ``extension``
escape was available alongside phase completion without identifying any
unsupported construct. Header decisions now use the explicit ``finish_header``
transition; signature selection phases have no unsupported-syntax escape.
Actual operation-body unsupported syntax retains its existing fail-closed path.
The header instruction distinguishes required inheritance from dependency API
names, which describe callable/stored APIs rather than required supertypes.

After the content failure, another artifact issues 914 consecutive
``construction_member_next`` requests, with 913 completed responses totaling
9,071.3 seconds. No type, signature, parameter or operation-author call occurs
in that consecutive segment. In the inspected loop, selecting a new member
immediately calls its type/signature planner, and selecting a queued body
leaves the loop. Therefore this segment consists of lookup-only decisions;
the console alone does not identify which queries/pages were selected.

Member planning previously offered the entire dependency symbol browser before
a missing helper had even been identified. That browser is now confined to
the subsequent type and body decisions that need those signatures. Member
planning retains finite own-declaration and binding pages, helper declarations
and a concrete ``begin_bodies`` transition selecting a queued member. Every
other assigned body remains queued. No lookup-count limit, shortened requested
behavior, automatic success, retry or timeout increase is introduced.

Each structured construction decision now reports its selected action/cursor
and the current phase, revision, declaration count and accepted node count.
This distinguishes response completion from actual accepted construction work
in the console without requiring a full response archive for that distinction.
Old planning journals/cursors are separated by the construction protocol;
declaration-bound accepted graph checkpoints retain their existing identities.

Verification remains syntax/diff inspection and static review. No tests,
model inference, Minecraft build or gameplay run was executed for this change.

Type lookup grounding
---------------------

Class headers and member types now search the inspected class inventory rather
than every field/method descriptor mentioning the query. Operation-body lookups
retain the complete member inventory. The lookup result identifies its symbol
kinds, and the phase instructions identify the requested information. All
matching classes remain pageable; no API/type is removed from the inventory.

The inventory expresses nested class owners as source names with ``.`` while
bytecode descriptors and model queries may use ``$``. Lookup matching, ranking,
consumed-selector identity and the native exclusion grammar now share the same
normalization. A short nested name such as ``Outer$Inner`` can resolve the exact
inspected ``package.Outer.Inner`` class before unrelated descriptor mentions.
The two spellings cannot reopen the same consumed lookup. Ordinary identifier
grammar retains its own Java identifier rules.

Console decisions also include selected node, behavior and port identifiers,
so body construction and source-port assignment choices can be distinguished
from lookup requests without logging full source bodies.

Follow-up: body lookup contracts and actual dependency facts
----------------------------------------------------------

The ``20261014-111017`` console log runs ``2cf3ba11d725`` and reaches
``construction_operation_next``. All three observed body authors request the
same symbol query and offset zero twice at revision zero with zero accepted
nodes. The queries are ``EntityTypeBuilderMixin``, ``EntityType#registerEntity()``
and the signature of a newly planned helper. No Java operation is accepted.
The two owning content nodes report ``CONSTRUCTION_AUTHOR_NO_PROGRESS``.

The declaration chooser had native unread-page constraints, while the body
chooser still used a separate enum with unrestricted query/offset fields.
Body decisions now reuse the symbol and binding chooser and expose exact
unread declaration, graph-node and source selectors. New catalogue searches
start at zero; subsequent pages are literal successors of real host results.
The bounded recent-history display no longer controls cursor reachability:
current query cursors are stored separately and reconstructed against current
inspected inputs on resume. Persisted selectors contain no source/API rows.
An empty catalogue query is available for the initial whole-inventory read,
then excluded from fresh searches like every other consumed selector.

Dependency lookups exclude unmaterialized planned declarations. Own members
remain available through the current declaration and pageable declaration
inventory. Their qualified and local signatures are excluded from new symbol
queries. Real type previews from the artifact's dependency requirements and
installed donor owners provide an ``inspect_dependency`` choice, with exact
inspected names rather than guessed mixin or method names. Free discovery and
all remaining target/source pages stay available.

The latest inspected dependency page remains visible while graph nodes are
accepted, and its selector is re-resolved on resume. Node IDs, references and
Java names are constrained to the existing renderer's accepted lexical rules;
an already accepted node ID cannot be emitted again. Runtime/gameplay meaning
still belongs to the authored operations and existing semantic checks.

This change is inspected statically with syntax/diff checks. Tests, inference,
Minecraft compilation and gameplay execution are not run.

Artifact-owned class headers
---------------------------

The same console log shows both entity registry containers selecting
``net.minecraft.world.entity.EntityType`` as their superclass. Neither plans
a constructor. Inspected target metadata declares an EntityType constructor
requiring fifteen arguments, so the implicit no-argument superclass call in
these containers could not satisfy that API. The entity registry manifest
requires calling/storing EntityType, not becoming an EntityType subtype.

Registry, registration and callback manifests now declare
``host_requirements.java_class_header: plain``. The native class composer
honors this host-owned responsibility by retaining the implicit Object
superclass and its existing explicit host lifecycle interfaces, then starting
member/body authoring. It does not ask the model to invent inheritance for a
registration container. This is a manifest contract, not a class-name ban or
a gameplay implementation. Type/renderer artifacts retain authored headers,
and missing metadata preserves the existing authored-header behavior. Unknown
policy values are rejected before inference and during manifest validation.

All assigned bodies, inspected dependency APIs, source reuse and port checks
remain in their existing authoring paths. Verification is limited to Python
syntax, YAML parsing, diff checks and static review; generation, compilation
and gameplay have not been executed.
