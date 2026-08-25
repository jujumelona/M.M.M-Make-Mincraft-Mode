from __future__ import annotations

"""Fail-safe runtime defaults for bounded local-model production.

These defaults keep the live Qwen/llama.cpp request safely below the physical slot
boundary and prevent adaptive retrieval from becoming an unbounded conversation.
Operators can still override the bounded context and tool-round values explicitly.
MTP is intentionally opt-in until its production tool-transport path is proven stable.
"""

import os

_DEFAULT_AGENT_TOOL_ROUNDS = "12"
_DEFAULT_SMALL_AGENT_CONTEXT_BYTES = str(24 * 1024)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in _TRUE_VALUES


def install_runtime_stability_defaults() -> None:
    """Install bounded defaults before model/runtime contracts are imported."""

    os.environ.setdefault("MMM_AGENT_TOOL_ROUNDS", _DEFAULT_AGENT_TOOL_ROUNDS)
    os.environ.setdefault(
        "MMM_SMALL_AGENT_CONTEXT_BYTES",
        _DEFAULT_SMALL_AGENT_CONTEXT_BYTES,
    )

    # The observed production failure already occurred with spec=none, so MTP is not
    # treated as the root cause. It is nevertheless disabled by default as requested;
    # a deliberate operator opt-in can restore the autotuner candidate widths later.
    if not _enabled("MMM_LLAMA_ENABLE_MTP"):
        os.environ["MMM_LLAMA_MTP_WIDTHS"] = ""
        os.environ["MMM_LLAMA_MTP_CONFIDENCE_WIDTHS"] = ""


__all__ = ["install_runtime_stability_defaults"]
