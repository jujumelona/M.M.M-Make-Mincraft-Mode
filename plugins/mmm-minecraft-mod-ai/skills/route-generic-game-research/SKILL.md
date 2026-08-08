---
name: route-generic-game-research
description: Decompose any small or game-scale Minecraft mod request into request-derived research domains, then route each domain to exact-version RAG, compatible open-source ecosystem search, licensed-media search, and unresolved-evidence gates. Use for broad, unfamiliar, cross-genre, reference-game, multimodal, library-selection, asset, 3D, audio, or plugin research before production planning.
---

# Route Generic Game Research

Do not use a genre example as a template. Derive domains from the user's actual
request, including any mechanics, simulation, vanilla-integration behavior, AI, social systems,
economy, progression, UI, networking, persistence, visuals, 3D, animation, VFX,
audio, accessibility, performance, compatibility, licensing, tests, and release
requirements that are truly relevant. Empty categories stay empty.

1. Build a dependency graph of request-derived research domains. Preserve every
   distinct requirement; group only genuinely repetitive catalogs.
2. Route Minecraft API and build questions to exact 1.20.1 Fabric/Yarn evidence,
   local behavior to project RAG, code/library candidates to Modrinth and GitHub,
   gameplay/domain meaning to Wikipedia as a secondary reference, and media
   references to reviewed license-aware catalogs.
3. Page every applicable provider with its cursor until the domain's coverage
   criteria are met. Follow every returned cursor, reject repeated cursors, and
   preserve a receipt for each page. A page-size limit is not a project-size
   limit.
4. Classify every result as dependency candidate, implementation reference,
   visual/audio reference, reusable artifact candidate, rejected, or unresolved.
5. Before selecting a dependency, inspect the exact compatible version, loader,
   transitive requirements/conflicts, origin license, and immutable file hash.
6. Before reusing media, verify the license at the origin and record creator,
   attribution, modification/share-alike duties, origin URL, and content hash.
7. Run corrective searches for uncovered or contradictory domains. Keep the gap
   explicit if evidence remains inadequate; original generation is allowed but
   does not turn an unverified third-party item into usable material.
8. When the request actually uses AI or speech, invoke
   `select-compatible-ai-technique`. Decompose ASR, VAD, TTS, transport,
   translation and optional consented adaptation; do not force them into an
   unrelated mod or select a model from recency alone.
9. Compile the result into a request-derived capability contract. Every selected
   result must name the requirement it supports, its implementation consumer,
   exact evidence IDs, required executable validators, and unresolved gates.
10. When current techniques target a newer Minecraft line, keep them as design
    candidates only until their concepts and APIs are translated back to the
    pinned target and exact target-version evidence passes. Continue corrective
    retrieval until every requirement is covered or explicitly unresolved.

## Runtime policy

```yaml
schema_version: mmm/skill-policy-v1
activate_when:
  - A request is broad, unfamiliar, cross-genre, reference-game based, or game-scale.
  - Planning needs compatible libraries, plugins, source references, images, 3D, animation, VFX, or audio evidence.
  - Initial retrieval leaves a request-derived domain uncovered or contradictory.
inputs:
  - original user request and current design domains
  - Minecraft 1.20.1, Fabric, Yarn and Java target profile
  - optional existing authorized project inventory
required_rag:
  - exact-version official implementation evidence for technical claims
  - project-local source and dependency relationships
  - provider metadata, origin license, compatibility and immutable artifact evidence
stages:
  - research
allowed_tools:
  - build_technology_radar
  - discover_ecosystem_resources
  - inspect_modrinth_project
  - inspect_github_repository
  - inspect_huggingface_model
  - assess_technology_compatibility
  - search_project_rag
  - index_project_rag
  - search_code_rag
  - inspect_existing_mod
validators:
  - exact_version_evidence
  - source_provenance
  - retrieval_coverage
  - retrieval_not_authority
  - requirement_traceability
  - quality_convergence
retry_policy:
  max_attempts: 3
  strategy: Reclassify the uncovered domain and continue from a fresh provider cursor or more precise evidence query.
  stop_on_repeated_error_signature: true
  require_fresh_evidence: true
approval_required:
  writes: false
  runtime: false
  read_only_research: false
forbidden_actions:
  - Force an unrequested capability, genre, or example-derived feature into the request.
  - Treat a search page, repository license badge, or Openverse record as final reuse permission.
  - Copy proprietary code, branding, characters, maps, art, writing, or audio from a named reference game.
  - Download, execute, install, or copy a discovery candidate.
  - Stop discovery merely because a page-size, model-context, or Colab-session boundary was reached.
  - Hide an unknown license, unresolved dependency conflict, weak source, or uncovered domain.
exit_conditions:
  success:
    - Every request-derived domain is covered by provenance-bearing evidence or an explicit original-generation and validation plan.
    - Every selected external candidate has exact compatibility, license, dependency and immutable-hash gates.
  blocked:
    - A required domain remains unsupported, contradictory, or legally ambiguous after corrective retrieval.
  failed:
    - A candidate crosses the read-only, license, provenance, secret, or host boundary.
```
