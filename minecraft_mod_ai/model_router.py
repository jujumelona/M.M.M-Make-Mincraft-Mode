from __future__ import annotations

# Public compatibility facade. The existing router implementation is preserved in
# model_router_core; only the retrieve/act/observe policy is overridden here.
from .model_router_core import *  # noqa: F401,F403
from .model_router_core import (
    _DEFAULT_AGENT_TOOL_ROUNDS,
    _EXTERNAL_RAG_CAPABILITIES,
    _GPU_EXCLUSIVE_LOCK,
    _MAX_AGENT_TOOL_ROUNDS,
    _MIN_AGENT_TOOL_ROUNDS,
    _NATIVE_TOOL_ADAPTERS,
    _PARALLEL_READ_TOOLS,
    _RAG_EVIDENCE_TOOLS,
    _ROLE_TOOL_STAGE,
    _agent_tool_round_limit,
    _execute_tool_waves,
    _external_mcp_result_has_content,
    _external_rag_capability,
    _inject_system_context,
    _parallel_read_workers,
    _positive_env_int,
    _tool_schema_names,
    _usable_external_rag_result,
    _usable_rag_result,
)
from .model_router_core import ModelRouter as _CoreModelRouter


class ModelRouter(_CoreModelRouter):
    """Model router with progress-aware adaptive retrieval."""

    def _generate_with_tools(self, *, adapter, request, runtime, stage: str, role: str) -> str:
        from .progress_aware_tool_loop import generate_with_tools

        return generate_with_tools(
            self,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )
