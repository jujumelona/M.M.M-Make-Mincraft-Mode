# M.M.M Make Minecraft Mode

**English** | [한국어](README_KO.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

M.M.M is a role-routed, fail-closed Minecraft Java Fabric `1.20.1` production
system. It turns an approved natural-language brief into source, runs deterministic
validation, Gradle, GameTest and JAR inspection, and packages only the evidence that
actually passed.

## Model roles

`config/model_registry.yaml` is the only source of model IDs.

| Role | T4 local model |
|---|---|
| Game design, world planning, visual review | `Qwen/Qwen3.5-4B` |
| Minecraft code generation | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Research and memory-safe coding | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| Code retrieval | `Qwen/Qwen3-Embedding-0.6B` |
| Retrieval reranking | `Qwen/Qwen3-Reranker-0.6B` |
| Concept images | `black-forest-labs/FLUX.2-klein-4B` |
| Speech recognition | `openai/whisper-small` |

A backend failure is reported as a `ModelBackendError`; it is never replaced by a
hidden heuristic result. `HeuristicPlanner` exists only as an explicitly selected
diagnostic/test backend.

## MCP and tool boundaries

The repository exposes a real stdio MCP server:

```bash
python -m minecraft_mod_ai.mcp_server
```

`.mcp.json` enables the local server, the version-aware Minecraft source/mapping
server and Playwright research. `config/external_mcp_registry.yaml` also records
reviewed, gated integrations for GitHub, JDT Language Server, Blockbench, a
disposable Fabric `1.20.1` runtime and Mineflayer `1.20.1`.

External integrations are fail-closed. Missing executables, wrong protocol
versions, unavailable servers, EULA refusal, path escape or unsupported MCP tools
return an error rather than a simulated success.

## Implemented production modules

- Fabric `1.20.1` source, resources, datagen, validation, Gradle, GameTest, JAR and release packaging
- Version/license-aware project code RAG with optional embedding and reranking
- JDT Language Server diagnostics and workspace-symbol adapter
- Restricted Blockbench MCP client allowlist
- GeckoLib `4.8.2` entity source/resource generation foundation
- WorldDesignIR compiler producing real gzipped structure NBT, template pools and a data-pack/world ZIP
- Quest, class/skill, economy/shop, GUI/networking and party/guild typed contract generators
- Disposable runtime instance manager with explicit EULA and allowlisted commands
- Mineflayer `1.20.1` bridge for connect, status, movement, interaction and inventory checks
- Verified build-trace store, reward calculation and LLaMA-Factory QLoRA configurations
- Seventeen executable Skill contracts under `skills/`

A generated foundation is not advertised as runtime-complete until JDT, Gradle,
GameTest, client/runtime and multiplayer gates required by that plugin pass.

## Colab

Open the notebook, set the prompt and model profile, and run all cells. The
notebook clones the selected repository ref, installs the declared extras, prints
the actual role registry, creates a plan and builds only when `APPROVE_PLAN=True`.

There are no direct model-ID fields and no silent fallback.

## Python API

```python
from minecraft_mod_ai import ModAISession

session = ModAISession.with_local_model(
    output_root="/content/mmm-output",
    minecraft_version="1.20.1",
    profile="t4_local",
)
reply = session.plan("Create two frost items, two blocks and a 41x41 arena.")
print(reply.message)

if reply.ready_to_build:
    result = session.build(reply, source_only=False)
    print(result.release_zip)
```

## Install and test

```bash
python -m pip install -e ".[dev,ui]"
python -m compileall -q minecraft_mod_ai tools mcp_gateway.py download_resources.py
python tools/build_colab_notebook.py --check
python -m pytest
```

Local model and production extras:

```bash
python -m pip install -e ".[local-model,rag,image,speech,training]"
./download_models.sh t4_local
```

## Target and safety boundary

The executable target is exactly Minecraft Java `1.20.1`, Fabric, Java `17`, Yarn
`1.20.1+build.1`. Other loaders, versions and mapping namespaces are rejected
unless a separate reviewed profile is added.

The ordinary CPU CI proves contracts, parsers, MCP handshake, source validation and
deterministic generators. Actual T4 model loading, Blockbench, Minecraft client,
server and Mineflayer execution require the manual self-hosted integration workflow
and its captured artifacts. A queued or unrun integration workflow is not evidence
of runtime success.

## Documentation

- [Production stack](docs/PRODUCTION_STACK.md)
- [MCP security](docs/MCP_SECURITY.md)
- [Training pipeline](training/README.md)
- [AI/MCP role matrix](docs/AI_MCP_MATRIX.md)

## License

Repository code is MIT. External programs and model weights retain their own
licenses and are invoked as separately installed components.
