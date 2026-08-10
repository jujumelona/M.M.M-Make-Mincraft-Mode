from __future__ import annotations

from functools import wraps
from typing import Any, Iterable


_PROJECT_ROOT_MUTATING_KINDS = frozenset(
    {
        "module-shard",
        "asset-shard",
        "audio-synth",
        "audio-finalize",
    }
)


def install(work_graph_module: Any) -> None:
    """Route every direct project-root writer through the single commit lane.

    Current module, asset and audio generators write directly into the generated
    Fabric project. Those writes invalidate ProjectIndex snapshots and can collide
    on shared files even when the logical work items are independent. Until these
    generators are split into parallel staging/intent creation followed by a
    path-aware commit phase, all direct project-tree mutation must be serialized.

    This is deliberately narrower than globally serializing the pipeline: planning,
    retrieval, validation outside the generation scheduler and other non-mutating
    work remain unaffected. A future staging contract can move image/audio synthesis
    back to their dedicated lanes without reintroducing filesystem races.
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
        if str(normalized.get("kind", "")) in _PROJECT_ROOT_MUTATING_KINDS:
            normalized["resource_class"] = "commit"
        return original(node_id, stage, dependencies, normalized)

    mutation_safe_node._mmm_module_mutation_contract = True
    work_graph_module._node = mutation_safe_node
