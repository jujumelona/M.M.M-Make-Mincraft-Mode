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
