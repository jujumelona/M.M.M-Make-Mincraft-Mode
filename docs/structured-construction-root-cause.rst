Structured construction: native grammar loss
===========================================

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
