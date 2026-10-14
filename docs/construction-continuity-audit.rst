Construction context and retrieval continuity audit
==================================================

Scope and evidence
------------------

The baseline is main ``8c05b9bdd12aca958eea2b8cde3fa0e12fec7712``.
The supplied production requests run the older construction-state v8. Both
initial decisions ask to inspect real context via ``extension`` before any
node is accepted. The registry request contains a correctness/negative-example
query and an unrelated Windows constant vocabulary. Those old failure paths
are compared with the actual production entry point, not with an isolated RAG
helper.

The user's standing instruction permits source/data/AST inspection only.
No project import, function execution, tests, model inference/downloads,
embedding/reranking model, Java compilation, Gradle or Minecraft run is used.
Static findings cannot certify model interpretation or gameplay execution.

Confirmed remaining defects and repair design
--------------------------------------------

* Fresh structured requests contain no previous chat. A full node read is
  overwritten even by the automatic API preparation before binding. Reading
  A then B also loses A's node/source/API/declaration data while the schema
  forbids rereading its consumed selector. Full binding pages accumulate in
  every prompt instead, eventually exhausting the context window.
* A host-owned evidence memory will retain exact selectors and atomic facts.
  Its active page and catalogue use the existing display page size; every
  earlier fact remains reachable through literal catalogue/recall actions.
  Resume rehydrates facts against the current task/source inventory. Remembered
  API/helper signatures feed actual binding candidates. Discovery bookkeeping
  remains distinct from memory navigation: recall does not create a new query.
* Names from all callbacks, branches and catches are globally pooled. A sibling
  callback's same-type parameter is a valid native choice but is rejected only
  after its node is permanently appended. Project the lexical environment at
  the current forward-reference slot, intersect shared use-sites, and preserve
  statement order, initialization and branch/callback/catch boundaries.
* Completed sibling graphs are absent from later member prompts. For example,
  a writer choosing state encoding 1 versus 2 leaves an identical reader
  request. Add accepted sibling task/node evidence as inspectable source units;
  keep its authority distinct from installed dependency bytecode.
* Bytecode members are discarded before caller package access is known.
  Retain inspected members and let the actual caller/access rules admit them.
  Check every enclosing class, use exact case-sensitive parents for typing,
  and build nested type navigation from inspected package/enclosing metadata.
* Input/label subdecisions have only a recent tail and no evidence reads. Keep
  their accepted sequence inspectable during the same decision and exclude
  labels already selected before another native choice is accepted.

Implementation sequence
-----------------------

1. Correct inspected inventory, enclosing access, exact type identity and real
   nested scopes; inspect the same admission rules in retrieval and final IR.
2. Project names at their current graph slot and reject newly introduced
   lexical contradictions before append/checkpoint.
3. Add bounded active evidence memory, complete catalogue recall and bound
   resume selectors; connect planning, operation and subdecision schemas.
4. Publish accepted sibling member graphs into the next body's source context.
5. Independently review counterexamples, parse changed source AST, inspect
   diffs, commit meaningful changes and publish main without running CI/tests.

Runtime limits
--------------

Literal evidence access and static admission do not prove that a small model
will select the right API or implement the user's mechanics. Finite discovery
prevents invented-query drift; it does not prove semantic progress or bound
all future model decisions. Large indivisible source units still depend on
the existing exact context guard. No retries, timeout increase, model load,
prompt-specific gameplay rule or fabricated runtime observation is a repair.

Completed static verification
-----------------------------

The reviewed code snapshot is local main
``b20f1bccd7949f4b8fc0a2f841db4de8b906994a``, tree
``d1ef9c93628cbb5ae235f79c4821ba1e30e7fb32``. Two fresh independent reviewers
reported no confirmed remaining Important/Critical finding within the assigned
continuity and retrieval/native repair scopes at that exact snapshot. This
documentation update changes no Python code. Publication verifies every
content-addressed tree, since connector-authored commit IDs differ from local
commit IDs.

All 596 Python files under ``minecraft_mod_ai`` were parsed with standard-library
``ast.parse``. The 24 changed Python files and the complete diff were checked;
``git diff --check`` was clean. The sorted changed-source path/SHA-256 manifest
has digest
``1a698f77eaf7b10b002835a773f1d8fa1e78c3dfdb953d29f486925ad31f2df9``.
These are syntax/content checks, not executed regression tests.

The actual production path remains ``integrity_dispatcher`` ->
``canonical_generators`` -> ``construction_generation.compose`` ->
``author_task_operations`` -> ``ModelRouter.generate_structured_data``.
The following counterexamples were traced through their producers, native
choices, prompt payloads, restart state and consumers:

* **Earlier facts disappear:** ``ConstructionMemory`` retains exact individual
  API/node/source/declaration/binding facts on the host. The active view has
  eight facts, and every archived fact remains reachable through the paged
  catalogue and literal ``recall``. Recall does not register a fresh RAG query.
  Recalled facts feed API/helper/type/header choices; restore re-resolves their
  selectors against the bound task inventory. Full binding history no longer
  accumulates in every prompt. Node-index summaries remain visible.
* **Wrong callback/catch/branch variable:** both partial admission and final IR
  use ``construction_lexical.lexical_sites``. It follows declaration order,
  deferred loop-body scopes, initialization, final assignment, captures and
  actual callback/catch boundaries. Shared holes intersect declaration
  identities. Contradictory names and loop-control uses are checked before a
  node is appended/checkpointed; native choices exclude unavailable names.
* **Sibling state encoding differs:** each accepted member publishes its task
  and individual typed-node units as ``accepted_sibling_graph`` evidence before
  the following body. The sibling graph hashes affect that following body's
  checkpoint key. This makes the earlier representation inspectable; it does
  not force a model to read or correctly use it. Compose keeps these units
  local and does not contaminate the caller's prerequisite source dictionary.
* **Stream history/termination changes on restart:** ordered label/catch/input
  sequences are pageable while authoring. Their binding, entries, completion
  flags and exact ``finally`` choice are saved before final-node inference.
  Resume skips finished sequences. Integer labels use disjoint remaining
  intervals; char/String labels keep decoded ``not/enum`` exclusions on the
  host and use canonical JSON-wire complements in native decoding. Quotes,
  backslashes, controls and Unicode scalar text retain valid spellings.
* **The sampler ignores the host exclusion:** direct native data payloads and
  output reserve use the same effective schema. Bounded regex groups retain
  their measured alternative widths. Preprojected native tool schemas carry
  their decoded string contract for host validation; the planner retains its
  original logical response schema. The pinned llama.cpp pattern/compiler
  source was read, never compiled or invoked.
* **RAG deletes/legalizes the wrong API:** inspected package-private members
  are retained and JVM ``<clinit>`` is excluded. Retrieval and IR inspect every
  enclosing class; subtype admission follows exact case-sensitive parent
  identities. Nested navigation uses actual package/enclosing metadata.
  Inherited public receiver projections retain the real declaring source,
  respect field hiding/ambiguity and most-specific interface returns, and
  retain original declaring-package protected access through rendering.

Earlier fixes remain in place: ranking uses positive implementation purpose,
atomic member documents and normalized frequencies; exact-name overlap is not
an admission requirement. Zero-overlap candidates and every surviving discovery
tail remain reachable. Kind narrowing retains the selected scope and root
frontier. A fixed slot/purpose has finite real inventory selectors, not freely
invented search strings. Reversible memory navigation can revisit facts, so
this does not assert that all future model decisions converge.

The result is a static closure of the identified host/protocol failure paths.
Actual model relevance, whether the small model consults the available facts,
complete generation, target compilation and observed gameplay remain
``NOT_RUN``/``NOT_OBSERVED``. No runtime success is manufactured from these
checks or from the independent review reports.
