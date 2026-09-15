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
    """Require one executable progress loop with single canonical context recovery.

    Per-turn causal routing and writable forced-tool wrappers are intentionally not
    part of the production path. Tool selection happens before this loop and the same
    reviewed set remains authoritative throughout retrieve/act/observe execution.
    """

    from .model_router import ModelRouter

    chain_layers = _chain(ModelRouter._generate_with_tools)
    implementations = tuple(_implementation(layer) for layer in chain_layers)
    progress = tuple(
        index
        for index, layer in enumerate(chain_layers)
        if getattr(layer, "_mmm_progress_aware_tool_loop_owner", False)
        or implementations[index] in {
            ("model_router.py", "_generate_with_tools"),
            ("progress_aware_tool_loop.py", "generate_with_tools"),
        }
    )
    if not progress:
        raise RuntimeLivePathError("live ModelRouter tool path has no progress-aware tool loop owner")


__all__ = ["RuntimeLivePathError", "run_runtime_live_path_preflight"]
