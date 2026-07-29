# Production model, MCP and plugin stack

## Local T4

| Role | Model | Adapter |
|---|---|---|
| planner / world planner / visual critic | Qwen/Qwen3.5-4B | multimodal, 4-bit |
| coder | Qwen/Qwen2.5-Coder-7B-Instruct | text, 4-bit |
| safe coder / researcher | Qwen/Qwen2.5-Coder-3B-Instruct | text, 4-bit |
| embedding | Qwen/Qwen3-Embedding-0.6B | CPU |
| reranker | Qwen/Qwen3-Reranker-0.6B | CPU |
| asset generator | FLUX.2-klein-4B | exclusive GPU + CPU offload |
| speech input | Whisper-small | exclusive GPU |

No automatic model substitution is allowed. The operator selects another
profile explicitly when the T4 quality model cannot load.

## MCP

- `mmm-local`: approval, generation, build, runtime, training receipts.
- `minecraft-dev`: 1.20.1 source, mappings and JAR evidence.
- `playwright`: documentation and web UI evidence.
- `github`: optional official repository and CI operations.
- `jdtls`: real Java diagnostics and symbols.
- `blockbench`: localhost-only restricted modeling operations.
- `minecraft-runtime-1201`: disposable local server/client processes.
- `mineflayer-1201`: localhost-only player task completion.

## Release rule

A generated feature is not release-ready until its plugin-specific gates,
Gradle, GameTest, JAR validation and any required runtime/visual tests pass.
`binding-gated`, `runtime-gated` and `configuration-required` are deliberate
states, not aliases for implemented.
