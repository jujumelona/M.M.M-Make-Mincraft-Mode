# M.M.M AI and MCP assignment

This matrix is executable configuration, not a marketing capability list. Model IDs live in `config/model_registry.yaml`; agent permissions live in `config/agent_roles.yaml`; MCP tools live in `minecraft_mod_ai/mcp_server.py`.

| Agent | Local T4 model | Adapter | MCP | Concrete responsibility |
|---|---|---|---|---|
| GameDesignPlanner | `Qwen/Qwen3.5-4B` | `AutoProcessor` + `AutoModelForMultimodalLM`, bnb 4-bit | `mmm-local`, `minecraft-dev` | Reference-image-aware game design, module status, buildable slice |
| ResearchAgent | `Qwen/Qwen3-4B-Instruct-2507` | `AutoTokenizer` + `AutoModelForCausalLM`, bnb 4-bit | `minecraft-dev`, `mmm-local` RAG | Version-pinned API/source evidence |
| MinecraftCoder | `Qwen/Qwen3-4B-Instruct-2507` | text causal LM, bnb 4-bit | `minecraft-dev`, `mmm-local` | Fabric Java/datagen/GameTest/build repair |
| VisualCritic | `Qwen/Qwen3.5-4B` | multimodal LM, bnb 4-bit | `mmm-local` | Texture/GUI/model/screenshot review |
| AssetGenerator | `black-forest-labs/FLUX.2-klein-4B` | Diffusers | `mmm-local` | Concept PNG and 16x16 texture candidates; exclusive GPU |
| SpeechAgent | `openai/whisper-small` | Transformers ASR | `mmm-local` | Optional transcription only |

## MCP policy

- `mmm-local`: real stdio FastMCP server. It exposes planning, approval, RAG, archive inspection, Fabric source generation, assets, validation, Gradle, GameTest, JAR inspection, and packaging.
- `minecraft-dev`: enabled through `npx -y @mcdxai/minecraft-dev-mcp` for 1.20.1 source, mappings, decompilation, search, and JAR analysis.
- Runtime Minecraft MCP: disabled. The reviewed upstream runtime projects target different Minecraft versions. A 1.20.1 fork must pass initialize, tools/list, command, screenshot, player-control, and disconnect tests before it is configured.
- Mineflayer runtime MCP: disabled until an explicit 1.20.1 compatibility run passes.

## T4 residency rule

The router does not keep all models resident. Text/multimodal roles load for one call and release GPU memory. Image generation and speech use a process-wide exclusive GPU lock. FLUX preflight requires a nearly empty T4 and CPU offload; failure is returned directly.

## Honest implementation boundary

M.M.M plans and builds requested Fabric mod systems, resources, and validation evidence. It does not design map layouts, arenas, or edits to a user's Minecraft world. Native Minecraft modules are considered only when the request explicitly calls for them.
