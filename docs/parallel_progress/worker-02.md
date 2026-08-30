# Worker 02 Progress

WORKER: 02
ROLE: RAG + Research + Evidence
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: 4a20c0c5a82961ccf7942fcfde2d3a4514dc9881

COMPLETED:
- Audited the live pre-design retrieval composition from approved requirement catalog through query rewrite, grounded retrieval, evidence materialization, fusion, and model consumer context.
- Confirmed the normal guarded production path rewrites raw authored text into approved English multi-query retrieval plans; the raw prompt remains provenance rather than the external search query.
- Confirmed zero-content pre-design evidence already fails closed and complete source bodies are materialized before bounded model research; did not duplicate those owners.
- Diagnosed baseline CI failure in test_research_grounded_rag_contract.py: the stale `mob` fixture bypassed the current single `request` pre-design domain and therefore selected the generic source mode instead of planning discovery.
- Updated the stale fixture to the live request-domain contract and pushed checkpoint commit 4a20c0c5a82961ccf7942fcfde2d3a4514dc9881.

IN_PROGRESS:
- Preserve requirement-to-query provenance through pre-design evidence fusion.
- Fail closed when one approved requirement has no content-bearing evidence even if sibling requirements have hits.
- Add regression tests for requirement-level sufficiency and provenance drift.

ROOT_CAUSES_CONFIRMED:
- Approved requirement search queries are flattened into a domain query list before grounded evidence fusion, so the fused evidence currently lacks an explicit requirement-to-query sufficiency receipt.
- Existing domain-level grounding can be satisfied by content from a sibling query; it does not prove that every approved requirement has at least one content-bearing evidence path.
- One existing CI regression fixture still modeled the retired per-mechanic pre-design domain instead of the live single request domain.

DECISIONS_AND_EVIDENCE:
- Keep the current single grounded-RAG owner; recent main already removed the duplicate runtime owner.
- Do not restore generic query suffix expansion; current query variants correctly preserve the approved query without generic `source implementation` spam.
- Do not weaken source-content gating: requirement sufficiency will use claim-bearing/content-bearing evidence records, not metadata-only hits or hashes.
- Keep query planning authority in the approved requirement graph and add a host-owned requirement sufficiency receipt at the canonical fusion boundary.

COMMITS_ALREADY_PUSHED:
- 4a20c0c5a82961ccf7942fcfde2d3a4514dc9881 test: align grounded RAG fixture with live request domain

TESTS_ALREADY_PASSING:
- Baseline CI before worker changes: 707/708 tests in Remaining tests 2/3 passed; the single failure was the stale grounded-RAG fixture now corrected by 4a20c0c5.

NEXT_EXACT_ACTIONS:
1. Add requirement-query provenance and content-bearing sufficiency evaluation to the canonical pre-design RAG quality facade without creating a second retrieval owner.
2. Add focused regression tests for sibling-hit masking, full requirement coverage, and provenance drift.
3. Verify focused and repository CI on the pushed checkpoints; rebase semantically on latest main if parallel workers advance it.
4. Update this progress file with test results and pushed SHAs, then continue worker-02 audit until no owned root causes remain.

FILES_CURRENTLY_RELEVANT:
- minecraft_mod_ai/authored_scope_research_contract.py
- minecraft_mod_ai/pre_design_research_pipeline.py
- minecraft_mod_ai/pre_design_domain_research.py
- minecraft_mod_ai/pre_design_rag_quality_contract.py
- minecraft_mod_ai/pre_design_rag_fusion.py
- minecraft_mod_ai/research_grounded_rag_contract.py
- tests/test_research_grounded_rag_contract.py

KNOWN_CROSS_ROLE_DEPENDENCIES:
- Shared runtime finalization installs the worker-02 contracts; no shared-core modification is currently required.
- Approved requirement graph remains owned by the requirements/planner layer; worker 02 consumes it as retrieval authority rather than redefining requirements.

UNRESOLVED:
- Requirement-level provenance/sufficiency receipt is not yet implemented.
- CI has not yet been observed on checkpoint 4a20c0c5.
