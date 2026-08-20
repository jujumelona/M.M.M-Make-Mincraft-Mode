from __future__ import annotations

"""Model-free checks for semantic ownership of the final live model-call path.

Marker-only checks are insufficient because ``functools.wraps`` copies a wrapped
callable's ``__dict__``. A replacement wrapper can therefore inherit an integration
marker while bypassing the callable that actually implemented that integration.
This preflight identifies wrapper implementations from their code objects and checks
ordering on the executable ``__wrapped__`` chain.
"""

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
    from .model_router import ModelRouter

    chain = _chain(ModelRouter._generate_with_tools)
    implementations = tuple(_implementation(layer) for layer in chain)

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
    route_integrity = tuple(
        index
        for index, item in enumerate(implementations)
        if item == ("coder_tool_route_integrity_contract.py", "generate_with_route_integrity")
    )

    if not compaction:
        raise RuntimeLivePathError(
            "live ModelRouter tool path has no executable context-compaction wrapper"
        )
    if not adaptive:
        raise RuntimeLivePathError(
            "live ModelRouter tool path has no adaptive retrieval owner"
        )
    if not route_integrity:
        raise RuntimeLivePathError(
            "live ModelRouter tool path has no coder route-integrity wrapper"
        )

    # adaptive_retrieval deliberately replaces the old loop and does not delegate to
    # its ``__wrapped__`` callable. Any compaction layer below it is therefore dead.
    # The live compaction wrapper must be outside the adaptive owner.
    if min(compaction) > min(adaptive):
        raise RuntimeLivePathError(
            "context compaction exists only below the adaptive replacement loop and "
            "is therefore not on the executable model path"
        )

    # Route integrity delegates to its captured current callable, so either order is
    # executable, but finalization intentionally places compaction outermost to make
    # every causal/tool turn pass through context fitting before decode.
    if min(compaction) > min(route_integrity):
        raise RuntimeLivePathError(
            "context compaction is not the final owner around coder route integrity"
        )


__all__ = ["RuntimeLivePathError", "run_runtime_live_path_preflight"]
