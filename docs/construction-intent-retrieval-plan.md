# Construction intent retrieval implementation plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. The user's no-test instruction overrides test execution in this plan.

**Goal:** Let an author discover usable source and target APIs from the behavior it must implement, and turn that evidence into typed operation nodes without guessing dependency names.

**Architecture:** Rank complete source declarations and real target type/member descriptions using the existing RAG scoring, actual type/reference relationships and the current behavior. The native model follows the actual package/type hierarchy and selects real signatures by meaning. Operation selection starts a concrete authoring goal; additional evidence reads belong to that goal, and the selected API fixes its result type and argument count. Missing graph inputs become the next concrete authoring slots.

**Tech stack:** Existing Python/RAG scoring, inspected Java symbols, complete AST source units, native JSON schemas and construction checkpoints. Existing embedding device policy remains in effect; lexical/structural discovery and native semantic selection work without loading a new CPU model.

**Spec:** The user-approved research synthesis in this conversation: functionality as retrieval input; real source/API candidates with target compatibility; evidence acquisition connected to the current authored operation.

## Global constraints

- Main branch only; commit each meaningful change and publish the reviewed result.
- No tests, project-function probes, model inference or Minecraft builds during this work. Verify Python syntax, data contracts and diffs; obtain independent static review.
- No lookup-count caps, retries, timeout increases or prompt-specific gameplay patches.
- Preserve every assigned body, helper, donor integration, source-port obligation and semantic/completion check.
- Preserve exact target/version/side/access authority. Reference-only old code is evidence, never callable target API authority.
- All real inventory/source entries remain reachable through literal successor pages; page size is presentation granularity, not an inventory cap.
- Advance the construction protocol when request/checkpoint contracts change. Accepted node graphs keep their existing checkpoint identity.

## Review focus

- A natural-language purpose without any actual API identifier still receives real candidates and usage evidence.
- CPU dense retrieval disabled: do not call blocked embedding/reranker loaders or report an embedding score that was not computed.
- Static/instance calls, constructors and fields: exact symbol kind, accessible owner/member, result type and argument count; signatures above eight arguments remain representable.
- Resume during a partially written operation: reconstruct goal/selectors against the current target and retain source-read obligations.
- Large inventories and reference-only sources: no fabricated authority, silently lost tail, new class-name ban or removed body.

## Task 1: Shared evidence ranking and construction retrieval

**Files:** Modify `minecraft_mod_ai/rag_index.py`; create `minecraft_mod_ai/construction_retrieval.py`.

**Interfaces:**
- `rank_context_chunks(query, chunks, *, query_vector=None, relation_scores=None)` returns all real input chunks in stable evidence order, including zero lexical overlap.
- `ConstructionEvidenceIndex(symbols, source_pages, router, *, target_path, authoring_contract)` owns reusable type/member and complete source evidence.
- `prepare(purpose, *, kinds)` returns a deterministic root goal; `page(goal_id, offset)` returns actual symbols. `scope_page` exposes real package/type branches and type relationships; `source_page` exposes all complete source pointers. Every catalogue has literal successors.
- Goals export only their purpose/kind/scope selectors. Resume recomputes candidate ordering from current inspected inputs.

- [x] Add public ranking over the existing lexical/cosine/relationship scoring without requiring an exact identifier match.
- [x] Build source/owner descriptions, rank from behavior purpose, carry actual source/type relationships, and filter side/access/constructor eligibility.
- [x] Provide source evidence and target APIs with separate authority; honor existing embedding device policy.
- [x] Parse changed Python with `ast.parse`, inspect the diff, run `git diff --check`, then commit.

## Task 2: Connect discovery to declarations and authored operations

**Files:** Modify `construction_state.py`, `construction_declaration_choices.py`, `construction_generation.py`, `construction_operation_choices.py`, `construction_author.py`; create `construction_operation_binding.py` and `construction_operation_contracts.py`.

**Interfaces:**
- State binds the evidence index and current goal. Declaration/type choices read purpose-derived real candidate pages.
- Operation-next selects only a concrete operation/purpose, completion or genuinely unsupported syntax.
- API binding selects actual candidate IDs or exact next context pages for that one operation. No new arbitrary symbol/source/node search string is an operation-next action.
- API-bound payload schemas set `symbol_ref`, `typed_result` and exact receiver/argument count from the inspected declaration.
- Checkpoints retain an unfinished operation goal and source selectors; all existing graph/source ownership checks still execute.

- [x] Connect the evidence index to the class/type and operation contexts; advance the protocol.
- [x] Replace independent body search with operation-owned evidence acquisition and native real-ID selection.
- [x] Ground API operations and their signature shapes; preserve forward references and all remaining operation kinds.
- [x] Keep source reads, donor/port reachability and every queued body intact.
- [x] Parse changed Python, inspect selectors/resume and run `git diff --check`, then commit.

## Task 3: Static review and main publication

**Files:** Update `docs/structured-construction-root-cause.rst` and this plan's progress notes.

- [ ] Document observed `095f153` timeout/lookup-only behavior and the corrected criteria, including backend/verification limits.
- [ ] Obtain independent static review of the complete change. Resolve material findings without executing tests or inference.
- [ ] Verify the reviewed local tree is clean and unchanged; publish exact blobs/trees with a guarded, non-force main update.
- [ ] Verify remote main and synchronize the local checkout; report the commit and remaining runtime verification limit.

## Execution rulings

- The repository ignores new Markdown documents; explicitly track this durable plan in the `docs` directory.
- User authorization is to implement the researched correction now; proceed inline without another approval handoff.
- Verification is deliberately static under the user's instruction. No test or runtime success is claimed.
- Task 1: implemented shared evidence ranking and target/source navigation. Syntax parsing passed; no project code ran. Ranked inventory caching retains one ordering and reconstructs older goals so graph growth does not duplicate the full API inventory per node.
- Task 1 review corrections in Task 2: source successors now cover the full ranked source map; native package/type navigation addresses the zero lexical overlap case. Immutable owner/source token sets are prepared once, member scoring is lazy for displayed owners, and repeated source imports use maximum relevance rather than popularity sums.
- Task 2: operation goals bind an exact node slot; pending inputs carry actual signature/control contracts and are filled before another root. API results/arity, named value types and callback signatures are closed choices. Resume re-resolves intent/scope/source selectors, retaining the unfinished goal. Static review remains required; runtime behavior is not certified.
