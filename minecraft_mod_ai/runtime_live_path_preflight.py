from __future__ import annotations

"""Model-free ownership checks for the final live model/tool path."""

from typing import Any


class RuntimeLivePathError(RuntimeError):
    pass


def _implementation(value: Any) -> tuple[str, str]:
    code = getattr(value, "__code__", None)
    if code is None:
        return "", ""
    filename = str(getattr(code, "co_filename", "")).replace("\\", "/")
    return filename.rsplit("/", 1)[-1], str(getattr(code, "co_name", ""))


def _chain(value: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[int] = set()
    current = value
    while callable(current):
        marker = id(current)
        if marker in seen:
            raise RuntimeLivePathError("live model wrapper chain contains a cycle")
        seen.add(marker)
        result.append(current)
        current = getattr(current, "__wrapped__", None)
    return tuple(result)


def run_runtime_live_path_preflight() -> None:
    """Require one executable progress loop plus outer context compaction.

    Per-turn causal routing and writable forced-tool wrappers are intentionally not
    part of the production path. Tool selection happens before this loop and the same
    reviewed set remains authoritative throughout retrieve/act/observe execution.
    """

    from .model_router import ModelRouter

    implementations = tuple(_implementation(layer) for layer in _chain(ModelRouter._generate_with_tools))
    compaction = tuple(
        index
        for index, item in enumerate(implementations)
        if item == ("small_model_compacting_adapter.py", "generate_with_compaction")
    )
    adaptive = tuple(
        index
        for index, item in enumerate(implementations)
        if item == ("adaptive_retrieval_contract.py", "_generate_with_tools")
    )
    if not compaction:
        raise RuntimeLivePathError("live ModelRouter tool path has no executable context-compaction wrapper")
    if not adaptive:
        raise RuntimeLivePathError("live ModelRouter tool path has no adaptive retrieval owner")
    if min(compaction) > min(adaptive):
        raise RuntimeLivePathError(
            "context compaction exists only below the adaptive replacement loop and is not executable"
        )


__all__ = ["RuntimeLivePathError", "run_runtime_live_path_preflight"]
