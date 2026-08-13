from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Any, Iterator, Sequence


_VALIDATION_CHECKPOINTS = frozenset({"validate-source", "validate-jdt"})


def _active_llm_slots() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _dependency_wave_shards(
    work_graph_module: Any,
    modules: Sequence[Any],
    *,
    policy: Any,
) -> Iterator[tuple[str, tuple[Any, ...]]]:
    """Create bounded shards and emit complete dependency-ready waves in order.

    Exact external dependency sets are indexed in O(1). Newly unlocked groups are
    deferred to the next wave so work that was already ready is never overtaken by a
    dependent group. Custom LLM shards expose the selected native decode slots while
    keeping the normal Java/entity shard ceilings.
    """

    staged = [(item, work_graph_module._module_stage(item)) for item in modules]
    stage_counts = Counter(stage for _, stage in staged)
    groups: list[dict[str, Any]] = []
    module_group: dict[str, int] = {}
    open_by_key: dict[tuple[str, frozenset[int]], int] = {}

    def shard_size_for(stage: str) -> int:
        if stage == "entity":
            return max(1, int(policy.entity_shard_size))
        if stage == "custom":
            count = max(1, int(stage_counts[stage]))
            slots = min(_active_llm_slots(), count)
            return min(
                max(1, int(policy.java_shard_size)),
                max(1, math.ceil(count / slots)),
            )
        return max(1, int(policy.java_shard_size))

    for item, stage in staged:
        missing = [
            dependency
            for dependency in item.depends_on
            if dependency not in module_group
        ]
        if missing:
            raise work_graph_module.WorkGraphError(
                "Module sharding requires topological order; unresolved dependencies for "
                f"{item.module_id}: {missing[:4]}"
            )

        shard_size = shard_size_for(stage)
        dependency_groups = {
            module_group[dependency] for dependency in item.depends_on
        }
        candidates: set[int] = set()

        exact_key = (stage, frozenset(dependency_groups))
        exact = open_by_key.get(exact_key)
        if exact is not None and len(groups[exact]["members"]) < shard_size:
            candidates.add(exact)

        for index in dependency_groups:
            group = groups[index]
            if group["stage"] != stage or len(group["members"]) >= shard_size:
                continue
            if (dependency_groups - {index}).issubset(group["external_groups"]):
                candidates.add(index)

        chosen = max(candidates) if candidates else None
        if chosen is None:
            chosen = len(groups)
            external_groups = set(dependency_groups)
            groups.append(
                {
                    "stage": stage,
                    "members": [],
                    "external_groups": external_groups,
                    "first_order": len(module_group),
                }
            )
            open_by_key[(stage, frozenset(external_groups))] = chosen

        group = groups[chosen]
        group["members"].append(item)
        module_group[item.module_id] = chosen

        group_key = (str(group["stage"]), frozenset(group["external_groups"]))
        if len(group["members"]) >= shard_size:
            if open_by_key.get(group_key) == chosen:
                open_by_key.pop(group_key, None)
        else:
            previous = open_by_key.get(group_key)
            if previous is None or chosen > previous:
                open_by_key[group_key] = chosen

    dependents: dict[int, list[int]] = {index: [] for index in range(len(groups))}
    indegree = [0] * len(groups)
    for index, group in enumerate(groups):
        dependencies = sorted(set(group["external_groups"]))
        indegree[index] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(index)

    ready = sorted(
        (index for index, degree in enumerate(indegree) if degree == 0),
        key=lambda index: int(groups[index]["first_order"]),
    )
    emitted = 0
    while ready:
        next_ready: set[int] = set()
        for index in ready:
            group = groups[index]
            yield str(group["stage"]), tuple(group["members"])
            emitted += 1
            for dependent in dependents[index]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.add(dependent)
        ready = sorted(
            next_ready,
            key=lambda index: int(groups[index]["first_order"]),
        )

    if emitted != len(groups):
        raise work_graph_module.WorkGraphError(
            "Module shard dependency graph contains a cycle."
        )


def _literal_false_validator(callback: Any) -> bool:
    """Recognize the legacy `lambda _cached: False` without overriding real policy."""

    code = getattr(callback, "__code__", None)
    if code is None or code.co_freevars or code.co_names:
        return False
    constants = tuple(value for value in code.co_consts if value is not None)
    return constants == (False,)


def _file_digest(module: Any) -> str:
    path_value = getattr(module, "__file__", "")
    if not path_value:
        return "missing"
    try:
        return hashlib.sha256(Path(path_value).resolve().read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _validation_implementation_fingerprint(checkpoint_id: str) -> str:
    """Fingerprint live validation code plus MMM runtime policy, not project output.

    Persistent checkpoint reuse is safe only when both generated inputs and validation
    semantics are unchanged. Hashing the relevant implementation files and all MMM_*
    policy variables makes a checkout/configuration change an automatic cache miss.
    Raw environment values are never persisted; only this digest enters the checkpoint
    input hash.
    """

    from . import complete_orchestrator, java_lsp, scalable_validator, scale_policy, validator

    modules = [complete_orchestrator]
    if checkpoint_id == "validate-source":
        modules.extend((scalable_validator, validator, scale_policy))
    elif checkpoint_id == "validate-jdt":
        modules.append(java_lsp)
    else:
        return ""

    digest = hashlib.sha256()
    for module in modules:
        digest.update(str(getattr(module, "__name__", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(module).encode("ascii"))
        digest.update(b"\0")
    for name, value in sorted(
        (name, value)
        for name, value in os.environ.items()
        if name.startswith("MMM_")
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"=")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if checkpoint_id == "validate-source":
        return value.get("status") == "PASS"
    if checkpoint_id == "validate-jdt":
        return (
            value.get("schema_version") == "mmm/java-diagnostics-v2"
            and isinstance(value.get("diagnostics"), dict)
        )
    return False


def _install_validation_checkpoint_reuse(orchestrator_module: Any) -> None:
    current = orchestrator_module.run_named_checkpoint
    if getattr(current, "_mmm_exact_validation_checkpoint_reuse", False):
        return

    @wraps(current)
    def run_named_checkpoint(
        ledger: Any,
        checkpoint_id: str,
        *,
        stage: str,
        input_value: Any,
        action: Any,
        encode: Any,
        decode: Any,
        validate_cached: Any = None,
    ) -> Any:
        if (
            checkpoint_id in _VALIDATION_CHECKPOINTS
            and _literal_false_validator(validate_cached)
        ):
            scoped_input = (
                dict(input_value)
                if isinstance(input_value, dict)
                else {"input_value": input_value}
            )
            scoped_input["_mmm_validation_implementation"] = (
                _validation_implementation_fingerprint(checkpoint_id)
            )
            return current(
                ledger,
                checkpoint_id,
                stage=stage,
                input_value=scoped_input,
                action=action,
                encode=encode,
                decode=decode,
                validate_cached=lambda value: _cached_validation_is_reusable(
                    checkpoint_id,
                    value,
                ),
            )
        return current(
            ledger,
            checkpoint_id,
            stage=stage,
            input_value=input_value,
            action=action,
            encode=encode,
            decode=decode,
            validate_cached=validate_cached,
        )

    run_named_checkpoint._mmm_exact_validation_checkpoint_reuse = True  # type: ignore[attr-defined]
    run_named_checkpoint.__wrapped__ = current  # type: ignore[attr-defined]
    orchestrator_module.run_named_checkpoint = run_named_checkpoint


def install(*, work_graph_module: Any) -> None:
    """Install dependency-aware sharding and exact-input resume reuse once."""

    from . import complete_orchestrator

    current_shards = work_graph_module._module_shards
    if not getattr(current_shards, "_mmm_dependency_wave_shards", False):

        def module_shards(modules: Sequence[Any], *, policy: Any):
            yield from _dependency_wave_shards(
                work_graph_module,
                modules,
                policy=policy,
            )

        module_shards._mmm_dependency_wave_shards = True  # type: ignore[attr-defined]
        work_graph_module._module_shards = module_shards

    _install_validation_checkpoint_reuse(complete_orchestrator)


__all__ = [
    "install",
    "_active_llm_slots",
    "_cached_validation_is_reusable",
    "_dependency_wave_shards",
    "_install_validation_checkpoint_reuse",
    "_literal_false_validator",
    "_validation_implementation_fingerprint",
]
