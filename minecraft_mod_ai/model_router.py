from __future__ import annotations

# Public compatibility facade. The existing router implementation is preserved in
# model_router_core; only the retrieve/act/observe policy is overridden here.
from . import model_router_core as _core
from .model_router_core import *  # noqa: F401,F403
from .model_router_core import ModelRouter as _CoreModelRouter

# The progress-aware loop imports the small private helper surface from this public
# module so callers keep one canonical model_router import path during the migration.
_RAG_EVIDENCE_TOOLS = _core._RAG_EVIDENCE_TOOLS
_agent_tool_round_limit = _core._agent_tool_round_limit
_execute_tool_waves = _core._execute_tool_waves
_external_rag_capability = _core._external_rag_capability
_tool_schema_names = _core._tool_schema_names
_usable_external_rag_result = _core._usable_external_rag_result
_usable_rag_result = _core._usable_rag_result


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
