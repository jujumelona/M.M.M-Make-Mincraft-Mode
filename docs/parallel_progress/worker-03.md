# Worker 03 — Requirements + Planner + Game Design

## 2026-08-31 — STARTED

- Base: `main` at `c7924e6481c8272be9b629220a969611dbb2eb2d`.
- Scope: requirement extraction/graph, planner stages, game design depth, task decomposition, acceptance traceability.
- Confirmed runtime ordering: the authoritative requirement graph is frozen before game-design planning; semantic authority, planner-graph integrity, and deep-design execution contracts are installed in production finalization.
- Confirmed defect: `agentic_research_game_design._generate_section()` silently fills missing required design fields with empty lists/maps, and `_validate_section_types()` accepts semantically empty required sections. This permits non-empty raw planner output to become an empty/mostly-empty accepted design and allows downstream retrieval/task planning to proceed without production-depth design.
- Confirmed test gap: `tests/test_agentic_research_game_design.py` currently treats empty `modules`, `combat`, and `mod_context` as a valid sectioned design fixture.
- Next invariant: reject/repair lossy section outputs; bind approved requirement IDs into design generation and require every approved requirement to reach at least one implementation-bearing design leaf before retrieval/coding; preserve optional emptiness only where the authored request truly does not require the surface.
