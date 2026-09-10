---
name: route-generic-game-research
description: Decompose any small or game-scale Minecraft mod request into request-derived research domains, then route each
  domain to exact-version RAG, compatible open-source ecosystem search, licensed-media search, and unresolved-evidence gates.
  Use for broad, unfamiliar, cross-genre, reference-game, multimodal, library-selection, asset, 3D, audio, or plugin research
  before production planning.
---

```yaml
activate_when:
- A request is broad, unfamiliar, cross-genre, reference-game based, or game-scale.
- Planning needs compatible libraries, plugins, source references, images, 3D, animation, VFX, or audio evidence.
- Initial retrieval leaves a request-derived domain uncovered or contradictory.
stages:
- research
inputs:
- original user request and current design domains
- the exact host-selected PlatformLock target profile
- optional existing authorized project inventory
required_rag:
- exact-version official implementation evidence for technical claims
- project-local source and dependency relationships
- provider metadata, origin license, compatibility and immutable artifact evidence
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
  strategy: Reclassify the uncovered domain and continue from a fresh provider cursor or more precise evidence query.
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
exit_conditions:
  success:
  - Every request-derived domain is covered by provenance-bearing evidence or an explicit original-generation and validation
    plan.
  - Every selected external candidate has exact compatibility, license, dependency and immutable-hash gates.
  blocked:
  - A required domain remains unsupported, contradictory, or legally ambiguous after corrective retrieval.
  failed:
  - A candidate crosses the read-only, license, provenance, secret, or host boundary.
```
