---
name: route-generic-game-research
description: Decompose any small or game-scale Minecraft mod request into request-derived research domains, then route each
  domain to exact-version RAG, compatible open-source ecosystem search, licensed-media search, and unresolved-evidence gates.
  Use for broad, unfamiliar, cross-genre, reference-game, multimodal, library-selection, asset, 3D, audio, or plugin research
  during game ideation and before production planning.
---

```yaml
activate_when:
- A request is broad, unfamiliar, cross-genre, reference-game based, or game-scale.
- Planning needs compatible libraries, plugins, source references, images, 3D, animation, VFX, or audio evidence.
- Initial retrieval leaves a request-derived domain uncovered or contradictory.
stages:
- planning
- research
inputs:
- original user request and current design domains
- the exact host-selected PlatformLock target profile
- optional existing authorized project inventory
route_before_search:
- Classify each unresolved question before choosing a tool or provider.
- User/gameplay intent that needs no external fact -> planner reasoning only; do not search.
- Vanilla mechanic/fact -> reviewed vanilla knowledge route.
- Minecraft symbol, mapping, registry, source behavior, version difference -> minecraft-dev/exact-version RAG route.
- Fabric/NeoForge implementation pattern -> reviewed modding-docs route.
- Current project code/reuse -> project-local code RAG route.
- Compatible external mod/library/plugin candidate -> ecosystem discovery/technology radar route, then inspect the candidate.
- Licensed media/model candidate -> the matching media/model discovery route with provenance/license gates.
required_rag:
- exact-version official implementation evidence for technical claims
- project-local source and dependency relationships when the question is project-specific
- provider metadata, origin license, compatibility and immutable artifact evidence for selected external candidates
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
  max_attempts: null
  strategy: Reclassify only the uncovered question and use a more precise reviewed route. Continue only while a new query,
    route, provider cursor, or admissible evidence item changes the unresolved evidence frontier.
  stop_on_repeated_error_signature: true
  require_fresh_evidence: true
approval_required:
  writes: false
  runtime: false
  release: false
forbidden_actions:
- Force an unrequested capability, genre, or example-derived feature into the request.
- Treat a search page, repository license badge, or Openverse record as final reuse permission.
- Copy proprietary code, branding, characters, maps, art, writing, or audio from a named reference game.
- Download, execute, install, or copy a discovery candidate.
- Stop discovery merely because a page-size, model-context, or Colab-session boundary was reached.
- Hide an unknown license, unresolved dependency conflict, weak source, or uncovered domain.
- Fan one question out to every available provider without a route classification.
- Merge unrelated sibling requirements into a search query.
- Send the complete task-wide discovery cache through an LLM relevance scan.
exit_conditions:
  success:
  - Every request-derived domain is covered by provenance-bearing evidence or an explicit original-generation and validation
    plan.
  - Every selected external candidate has exact compatibility, license, dependency and immutable-hash gates.
  blocked:
  - A required domain remains unsupported, contradictory, or legally ambiguous and the unresolved evidence frontier reaches a
    fixed point with no fresh admissible evidence.
  failed:
  - A candidate crosses the read-only, license, provenance, secret, or host boundary.
```
