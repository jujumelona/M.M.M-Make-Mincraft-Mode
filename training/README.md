# Verified Minecraft fine-tuning

Training data is not scraped blindly. A trace enters the dataset only after the
pinned Fabric 1.20.1 build, JDT diagnostics, request-fidelity checks, GameTest,
registry-reference validation and JAR validation all pass.

## Models

- T4-safe coder experiment: `Qwen/Qwen2.5-Coder-3B-Instruct`
- Higher-quality coder: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Multimodal planner/critic: `Qwen/Qwen3.5-4B`

## Workflow

1. Run generation and repair through the approved MCP pipeline.
2. Preserve exact prompt, response, patch, source license, source commit and all
   machine receipts.
3. Record the trace through `record_training_trace` or `mmm-training record`.
4. Export through `export_training_dataset`.
5. Run the matching LLaMA-Factory configuration.
6. Evaluate on held-out projects with build, GameTest and runtime playthrough.
7. Never merge an adapter merely because training loss decreased.

Decompiled Mojang source, unlicensed mods, original Minecraft assets and mixed
Forge/Fabric datasets are forbidden.
