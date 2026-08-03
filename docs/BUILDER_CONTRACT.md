# Central Agent → Builder Contract

## Boundary

M.M.M interprets user language and images in the central VLM/agent layer. Architecture
RAG is also central-only. The external Builder model never receives user prompts,
captions, style prose, scene-meaning prose, or retrieved passages.

The only Builder input is a validated `buildspec_v2` object plus referenced `.npz`
artifacts. Builder returns block deltas and validation predictions.

```text
user language / images
        ↓
central VLM agent
        ↓
architecture RAG + MCP tools
        ↓
validated buildspec_v2 + world NPZ artifacts
        ↓
external Builder AI
        ↓
add/remove/replace NPZ + port/constraint predictions
```

## Builder input

```json
{
  "schema_version": "buildspec_v2",
  "world": {
    "origin": [0, 64, 0],
    "bbox": [0, 0, 0, 160, 80, 160],
    "context_blocks_ref": "world_context.npz",
    "terrain_ref": "terrain.npz",
    "protected_mask_ref": "protected.npz"
  },
  "zones": [],
  "components": [],
  "parts": [],
  "relations": [],
  "ports": [],
  "patterns": [],
  "operators": [],
  "task": {
    "type": "generate",
    "target_component_ids": [],
    "completed_component_ids": [],
    "open_port_ids": []
  },
  "constraints": {
    "hard": [],
    "soft": []
  }
}
```

Natural-language keys such as `prompt`, `brief`, `caption`, `description`, `style`,
`style_description`, `scene_meaning`, and `natural_language` are rejected recursively.
Strings inside the BuildSpec must be compact machine tokens.

## Builder output

```json
{
  "add_blocks_ref": "add.npz",
  "remove_blocks_ref": "remove.npz",
  "replace_blocks_ref": "replace.npz",
  "resolved_ports": [],
  "remaining_open_ports": [],
  "validation_predictions": {
    "supported": true,
    "connected": true,
    "constraint_violations": []
  }
}
```

`resolved_ports` and `remaining_open_ports` must be disjoint and together must exactly
partition the input task's `open_port_ids`.

## MCP server

`.mcp.json` exposes `mmm-builder-contract`, implemented by
`minecraft_mod_ai.builder_mcp_server`.

Tools:

- `discover_builder_contract`
- `search_buildspec_rag`
- `plan_architecture_buildspec`
- `validate_architecture_buildspec`
- `prepare_external_builder_handoff`
- `validate_external_builder_result`

`prepare_external_builder_handoff` writes canonical `buildspec.json` and reports
`NOT_EXECUTED_BY_MMM`. The separate Builder AI remains an external execution component.
