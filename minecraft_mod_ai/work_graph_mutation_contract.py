from __future__ import annotations

from functools import wraps
from typing import Any, Iterable


def install(work_graph_module: Any) -> None:
    """Assign generation work to the narrowest safe execution lane.

    Custom LLM generation is isolated by ``performance_final_contract`` and therefore
    may run in the LLM lane while shared deterministic project mutations use the short
    commit lane. Asset targets and synthesized OGG files are disjoint by contract, so
    their expensive generation can overlap safely with both LLM and commit work.
    """

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