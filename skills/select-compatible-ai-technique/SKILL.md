---
name: select-compatible-ai-technique
description: Research, compare, and fail-closed gate request-derived AI, agent, speech, translation, or consented voice-adaptation techniques for a Minecraft 1.20.1 Fabric mod. Use when a requested feature may need model inference, tool use, semantic memory, ASR, VAD, TTS, voice transport, LoRA, voice conversion, a local sidecar, an in-process Java runtime, or a remote API.
---

# Select Compatible AI Technique

1. Split the requested pipeline into independent capabilities. Keep inference,
   tool use, memory, ASR, VAD, TTS, translation, transport, and optional voice
   adaptation separate. Omit every capability the user did not request.
2. Search current runtime and model catalogs, then inspect promising candidates
   at an immutable revision. Treat search rank and update time as discovery
   signals, never compatibility or quality proof.
3. Compare four execution boundaries: Java 17 in-process, approved localhost
   sidecar, remote API with explicit data consent, and offline build-time tool.
   Keep Minecraft ticks deterministic and return only small typed intents to the
   server thread.
4. Prove Minecraft 1.20.1, Fabric, Yarn, Java 17, client/server placement,
   networking authority, OS/architecture, model format, CPU/GPU/RAM, startup,
   latency, concurrency, offline behavior, and deterministic fallback.
5. Record code, model weights, base model, dataset, adapter, and media licenses
   separately. Pin revision and expected file digests. Reject missing or custom
   license terms until a human reviews their exact text. Never execute discovered
   code or load pickle/remote code during research.
6. Benchmark candidates against the request: tick p95/p99, timeout and fallback,
   ASR partial/final latency and language WER, TTS first-audio latency, model
   quality, peak RAM/VRAM, cancellation, and multiplayer authority as relevant.
7. For voice, advertise only the full input-to-output language intersection.
   Keep speaker identity in the TTS/voice model and expression movement in an
   utterance-local PatternTrace time series (`time`, `energy`, `entropy`, `f0`,
   `attack`, `pause`), never one embedding or a rolling conversation average.
8. Start voice adaptation or LoRA only when the speaker is authorized, explicit
   consent is recorded, and provenance, allowed purpose/scope, AI-voice
   disclosure, revocation, deletion, retention, and local/remote data-flow
   receipts all pass. Owning a recording file alone is not speaker consent.
   Otherwise block adaptation while leaving ordinary TTS or non-voice gameplay
   available.
9. Attach the normalized decision, evidence IDs, unresolved gates, topology,
   fallback, and immutable receipts to the proposal before implementation. A
   discovered candidate is evidence, not authorization to download or integrate.
   Accept executed-test receipts only when a code-owned MAC binds the requirement,
   candidate ID, revision, artifact digest, and evidence digest; a caller-computed
   public hash is not execution proof.

## Runtime policy

```yaml
schema_version: mmm/skill-policy-v1
activate_when:
  - A requested Minecraft feature needs AI inference, agents, semantic memory, speech, translation, or voice adaptation.
  - Planning must choose between Java in-process, localhost sidecar, remote API, or offline build-time inference.
inputs:
  - original user request and request-derived research domains
  - exact Minecraft 1.20.1, Fabric, Yarn and Java 17 target
  - current hardware, network, privacy, language and latency constraints
required_rag:
  - exact target-version Minecraft implementation evidence
  - immutable runtime and model-card metadata
  - code, model, dataset, adapter and media license evidence
  - reproducible runtime and quality benchmarks
stages:
  - planning
  - research
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
  max_attempts: 3
  strategy: Correct the capability query or execution boundary and retrieve fresh evidence; never relax a failed license, consent, hash, or target-version gate.
  stop_on_repeated_error_signature: true
  require_fresh_evidence: true
approval_required:
  writes: false
  runtime: false
  read_only_research: false
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
    - A required license, consent, immutable artifact, language intersection, runtime budget, or target-version proof remains unresolved.
  failed:
    - A candidate crosses the execution, network, secret, provenance, consent, or artifact-safety boundary.
```
