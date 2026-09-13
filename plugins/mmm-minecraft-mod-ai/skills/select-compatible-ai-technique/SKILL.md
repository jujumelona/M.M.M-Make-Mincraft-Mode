---
name: select-compatible-ai-technique
description: Research, compare, and fail-closed gate request-derived AI, agent, speech, translation, or consented voice-adaptation
  techniques for a Minecraft Java mod on the host-selected executable target. Use when a requested feature may need model
  inference, tool use, semantic memory, ASR, VAD, TTS, voice transport, LoRA, voice conversion, a local sidecar, an in-process
  Java runtime, or a remote API.
---

```yaml
activate_when:
- A requested Minecraft feature needs AI inference, agents, semantic memory, speech, translation, or voice adaptation.
- Planning must choose between Java in-process, localhost sidecar, remote API, or offline build-time inference.
stages:
- planning
- research
inputs:
- original user request and request-derived research domains
- exact approved Minecraft version, loader, mappings, and Java target
- current hardware, network, privacy, language and latency constraints
required_rag:
- exact target-version Minecraft implementation evidence
- immutable runtime and model-card metadata
- code, model, dataset, adapter and media license evidence
- reproducible runtime and quality benchmarks
allowed_tools:
- build_technology_radar
- discover_ecosystem_resources
- inspect_huggingface_model
- inspect_github_repository
- assess_technology_compatibility
- search_project_rag
- search_code_rag
validators:
- exact_version_evidence
- immutable_model_revision
- separate_license_closure
- execution_boundary
- data_flow_and_consent
- measured_runtime_quality
- deterministic_fallback
retry_policy:
  max_attempts: null
  strategy: Correct the capability query or execution boundary and retrieve fresh evidence; never relax a failed license,
    consent, hash, or target-version gate.
  stop_on_repeated_error_signature: true
  require_fresh_evidence: true
approval_required:
  writes: false
  runtime: false
  release: false
forbidden_actions:
- Select a product because it is merely newest, popular, or first in search.
- Download weights, execute repository code, enable trust_remote_code, or deserialize pickle during discovery.
- Put model inference or blocking network calls on the Minecraft server tick.
- Place provider secrets in a client mod or let model output directly mutate the world.
- Clone, adapt, or imitate a voice without explicit provenance-bearing consent.
- Describe transcription, procedural tones, or ordinary TTS as voice cloning.
- Treat this read-only planning Skill as authorization to run or integrate a runtime.
exit_conditions:
  success:
  - Every selected technique has exact compatibility, immutable identity, license, privacy, benchmark, fallback and test evidence.
  blocked:
  - A required license, consent, immutable artifact, language intersection, runtime budget, or target-version proof remains
    unresolved.
  failed:
  - A candidate crosses the execution, network, secret, provenance, consent, or artifact-safety boundary.
```
