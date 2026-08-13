# M.M.M Make Mincraft Mode

[한국어](README_KO.md)

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

> **Runtime boundary:** M.M.M is a Fabric mod-production system. World/map generation is not part of the runtime contract; structure assets may be consumed only as ordinary mod resources when a mod explicitly needs them.

## Local Colab quick start

The canonical notebook is `M.M.M_Make_Mincraft_Mode_Colab.ipynb`. It pulls `main`, validates the checkout, installs the selected runtime profile, and uses the verified native `llama-server` path for local GGUF profiles.

For the current local T4 profile, the default model is `Qwen3.5-9B_6GB` (`unsloth/Qwen3.5-9B-MTP-GGUF`, `Qwen3.5-9B-UD-Q4_K_XL.gguf`).

## Repository structure

- `minecraft_mod_ai/`: planner, research, coding, validation, packaging, MCP, and native llama runtime contracts.
- `config/`: model/runtime/agent configuration.
- `skills/`: packaged production Skills used by the agent runtime.
- `integrations/`: runtime bridges such as Mineflayer.
- `tools/`: Colab setup and verified native llama bundle loader.
- `.github/workflows/`: CI and verified native CUDA bundle builds.

See [README_KO.md](README_KO.md) for the detailed Korean guide.
