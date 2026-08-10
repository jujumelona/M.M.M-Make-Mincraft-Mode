from __future__ import annotations

from functools import wraps
from typing import Any, Iterable


def install(work_graph_module: Any) -> None:
    """Force direct-writing module shards through the single commit lane.

    Current module generators perform read/modify/write operations directly against
    shared project files (registries, language files, catalogs and the main
    initializer). Running those generators in different executor lanes is unsafe
    even when their logical modules are independent, because they can share those
    physical files. Assets and audio synthesis remain in their dedicated parallel
    lanes; this contract only serializes module shards until generation is split
    into pure intent creation plus a path-aware commit phase.
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
        if str(normalized.get("kind", "")) == "module-shard":
            normalized["resource_class"] = "commit"
        return original(node_id, stage, dependencies, normalized)

    mutation_safe_node._mmm_module_mutation_contract = True
    work_graph_module._node = mutation_safe_node
