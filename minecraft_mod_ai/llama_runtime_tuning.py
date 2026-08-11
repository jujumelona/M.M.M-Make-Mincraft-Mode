from __future__ import annotations

import os
from functools import wraps
from typing import Any


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _tuned_constructor_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(kwargs)
    logical_batch = _positive_env_int("MMM_LLAMA_BATCH", 2048)
    micro_batch = _positive_env_int("MMM_LLAMA_UBATCH", 512)

    # llama-cpp-python distinguishes logical prompt batching (n_batch) from the
    # physical microbatch (n_ubatch). Increasing only the former improves long
    # prompt prefill throughput while retaining the existing 512-token physical
    # allocation ceiling.
    current_batch = int(tuned.get("n_batch", 512))
    if current_batch <= 512:
        tuned["n_batch"] = logical_batch
    current_ubatch = int(tuned.get("n_ubatch", 512))
    tuned["n_ubatch"] = min(current_ubatch, micro_batch)

    if os.environ.get("MMM_LLAMA_VERBOSE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        tuned["verbose"] = False
    return tuned


def install() -> None:
    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    original = LlamaCppAdapter.generate
    if getattr(original, "_mmm_prefill_tuned", False):
        return

    @wraps(original)
    def tuned_generate(self: Any, request: Any) -> str:
        # The configured llama-server path does not need llama-cpp-python in this
        # process. If the package is absent, preserve that remote/local-server path.
        try:
            import llama_cpp
        except ImportError:
            return original(self, request)

        original_ctor = llama_cpp.Llama

        @wraps(original_ctor)
        def tuned_ctor(*args: Any, **kwargs: Any):
            return original_ctor(*args, **_tuned_constructor_kwargs(kwargs))

        # LLM work is already serialized by ModelRouter and the dedicated LLM lane,
        # so this short constructor substitution cannot race another local text load.
        llama_cpp.Llama = tuned_ctor
        try:
            return original(self, request)
        finally:
            llama_cpp.Llama = original_ctor

    tuned_generate._mmm_prefill_tuned = True
    LlamaCppAdapter.generate = tuned_generate
