from __future__ import annotations

from functools import wraps
from typing import Any, Iterable


_LOCAL_AI_SIDECAR = "mmm_local_ai_sidecar"


def install(work_graph_module: Any) -> None:
    """Assign generation work to the narrowest safe execution lane.

    Custom LLM generation is isolated by ``performance_final_contract`` and therefore
    may run in the LLM lane while shared deterministic project mutations use the short
    commit lane. Asset targets and synthesized OGG files are disjoint by contract, so
    their expensive generation can overlap safely with both LLM and commit work.
    """

    original_stage = work_graph_module._module_stage
    if not getattr(original_stage, "_mmm_final_stage_contract", False):

        @wraps(original_stage)
        def final_module_stage(module: Any) -> str:
            stage = original_stage(module)
            # Research shards and the code-owned local AI sidecar are deterministic.
            # Every other integration is model-backed in module_node_action; route it
            # to the LLM lane up front instead of hiding a long coder call inside the
            # serialized commit lane.
            if (
                getattr(module, "kind", "") == "integration"
                and not work_graph_module.is_research_shard(module)
                and getattr(module, "config", {}).get("integration_type")
                != _LOCAL_AI_SIDECAR
            ):
                return "custom"
            return stage

        final_module_stage._mmm_final_stage_contract = True
        work_graph_module._module_stage = final_module_stage

    original = work_graph_module._node
    if getattr(original, "_mmm_module_mutation_contract", False):
        return

    @wraps(original)
    def mutation_safe_node(
        node_id: str,
        stage: str,
        dependencies: Iterable[str],
        payload: dict[str, Any],
    ):
        normalized = dict(payload)
        kind = str(normalized.get("kind", ""))
        generation_stage = str(normalized.get("generation_stage", ""))

        if kind == "module-shard":
            normalized["resource_class"] = (
                "llm" if generation_stage == "custom" else "commit"
            )
        elif kind == "asset-shard":
            normalized["resource_class"] = "image_gpu"
        elif kind == "audio-synth":
            normalized["resource_class"] = "cpu_io"
        elif kind in {"audio-finalize", "audio-shard"}:
            normalized["resource_class"] = "commit"

        return original(node_id, stage, dependencies, normalized)

    mutation_safe_node._mmm_module_mutation_contract = True
    work_graph_module._node = mutation_safe_node