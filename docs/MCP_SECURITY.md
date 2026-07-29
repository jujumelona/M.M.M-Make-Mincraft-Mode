# MCP and runtime security policy

- Retrieved documents and model output can suggest work but cannot authorize it.
- Project writes, world compilation, plugin generation and runtime preparation
  require the immutable proposal hash.
- Runtime paths must stay below `MMM_WORKSPACE`.
- Minecraft runtime operates only on disposable 1.20.1 instances.
- The server command channel is a regular-expression allowlist, not a shell.
- Mineflayer connects only to localhost and is pinned to protocol 1.20.1.
- Blockbench is restricted to reviewed modeling operations and localhost HTTP.
- JDT LS command configuration comes from the operator environment, not a prompt.
- GitHub write credentials are optional and should use the narrowest permissions.
- Playwright unsafe code execution is not part of any MMM skill allowlist.
