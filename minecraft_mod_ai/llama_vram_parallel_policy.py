from __future__ import annotations

"""VRAM-first admission and bounded cold-search policy for managed llama-server.

Normal inference never needs an exhaustive benchmark.  The default ``fast`` search
keeps the live VRAM/resource validation contract but avoids repeatedly reloading the
same model for speculative, ubatch, and every intermediate parallel candidate.
Operators can opt into ``balanced`` or ``full`` search when they explicitly want a
wider offline sweep.
"""

import json
import os
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any

_POLICY_VERSION = 5
_MIB = 1024 * 1024
_RESOURCE_MARKER = "_mmm_vram_parallel_resource_policy_v5"
_SELECTION_MARKER = "_mmm_vram_parallel_selection_policy_v5"
_CANDIDATE_MARKER = "_mmm_vram_parallel_bounded_candidates_v5"
_SEARCH_MARKER = "_mmm_llama_bounded_cold_search_v5"
_LEGACY_RESOURCE_MARKERS = (
    "_mmm_vram_parallel_resource_policy_v1",
    "_mmm_vram_parallel_resource_policy_v2",
    "_mmm_vram_parallel_resource_policy_v3",
    "_mmm_vram_parallel_resource_policy_v4",
)
_LEGACY_SELECTION_MARKERS = (
    "_mmm_vram_parallel_selection_policy_v1",
    "_mmm_vram_parallel_selection_policy_v2",
    "_mmm_vram_parallel_selection_policy_v3",
    "_mmm_vram_parallel_selection_policy_v4",
)
_RUNTIME_RECEIPT_SCHEMA = "mmm/llama-runtime-receipt-v1"


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _search_mode() -> str:
    raw = os.environ.get("MMM_LLAMA_AUTOTUNE_SEARCH", "fast").strip().lower()
    return raw if raw in {"fast", "balanced", "full"} else "fast"


def _configure_benchmark_defaults() -> None:
    """Keep cold tuning bounded without overriding explicit operator choices."""
    mode = _search_mode()
    if mode == "fast":
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_TOKENS", "16")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_MAX_SECONDS", "45")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS", "15")
    elif mode == "balanced":
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_TOKENS", "32")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_MAX_SECONDS", "120")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS", "30")


def _active_parallelism() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def validated_active_parallelism() -> int:
    """Return slots only when the managed runtime receipt proves they are live."""
    active = _active_parallelism()
    if active <= 1:
        return 1
    raw = os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", "").strip()
    if not raw:
        return 1
    try:
        receipt = json.loads(raw)
    except Exception:
        return 1
    if not isinstance(receipt, dict):
        return 1
    if str(receipt.get("schema_version", "")) != _RUNTIME_RECEIPT_SCHEMA:
        return 1
    try:
        receipt_slots = max(1, min(8, int(receipt.get("slots", 1))))
    except (TypeError, ValueError):
        return 1
    return active if receipt_slots == active else 1


def _unwrap_marked(current: Callable[..., Any], markers: tuple[str, ...]) -> Callable[..., Any]:
    while any(bool(getattr(current, marker, False)) for marker in markers):
        previous = getattr(current, "__wrapped__", None)
        if not callable(previous):
            break
        current = previous
    return current


def _shortlist_parallel(values: Iterable[int], mode: str | None = None) -> tuple[int, ...]:
    """Keep p1 as the exact-output reference and benchmark only useful high widths."""
    ordered = tuple(sorted({max(1, int(value)) for value in values}))
    if not ordered:
        return (1,)
    active_mode = mode or _search_mode()
    if active_mode == "full":
        return ordered

    high = [value for value in ordered if value > 1]
    if not high:
        return (1,)
    if active_mode == "fast":
        return (1, high[-1])

    # Balanced mode keeps the two highest feasible contenders.  This still samples
    # the saturation edge without returning to an O(max_slots) server-reload sweep.
    contenders = high[-2:]
    return tuple(dict.fromkeys((1, *contenders)))


def _shortlist_ubatches(values: Iterable[int], mode: str | None = None) -> tuple[int, ...]:
    ordered = tuple(dict.fromkeys(max(1, int(value)) for value in values))
    if not ordered:
        return ()
    active_mode = mode or _search_mode()
    if active_mode == "full":
        return ordered
    if active_mode == "fast":
        return (ordered[0],)
    largest = max(ordered)
    return tuple(dict.fromkeys((ordered[0], largest)))


def _shortlist_variants(values: Iterable[Any], mode: str | None = None) -> tuple[Any, ...]:
    variants = tuple(values)
    if not variants:
        return variants
    active_mode = mode or _search_mode()
    if active_mode == "full":
        return variants

    baseline = next(
        (value for value in variants if str(getattr(value, "spec_type", "none")) == "none"),
        variants[0],
    )
    if active_mode == "fast":
        return (baseline,)

    mtp = [value for value in variants if str(getattr(value, "spec_type", "")) == "draft-mtp"]
    ngram = [
        value
        for value in variants
        if str(getattr(value, "spec_type", "")).startswith("ngram-")
    ]
    chosen: list[Any] = [baseline]
    if mtp:
        # Prefer a moderate draft width; very wide drafts are expensive to validate
        # and are better left to explicit full/offline tuning.
        chosen.append(min(mtp, key=lambda item: abs(int(getattr(item, "draft_n_max", 1)) - 4)))
    if ngram:
        preferred = next(
            (item for item in ngram if str(getattr(item, "spec_type", "")) == "ngram-map-k"),
            ngram[0],
        )
        chosen.append(preferred)
    return tuple(dict.fromkeys(chosen))


def _install_resource_admission(runtime_tuning: Any) -> None:
    current = runtime_tuning._parallel_resource_feasible
    if getattr(current, _RESOURCE_MARKER, False):
        return
    current = _unwrap_marked(current, _LEGACY_RESOURCE_MARKERS)

    @wraps(current)
    def vram_first_parallel_feasible(
        slots: int,
        config: Any,
        model_path: str | None,
        resources: Any,
    ) -> bool:
        if current(slots, config, model_path, resources):
            return True

        slots = max(1, int(slots))
        if slots <= 1 or not _env_enabled("MMM_LLAMA_VRAM_PARALLEL", True):
            return False

        try:
            context = runtime_tuning._per_request_context(config)
            total_context = runtime_tuning._total_context(context, slots)
            model_bytes = runtime_tuning._model_size(model_path)
            gpu_free = max(0, int(resources.gpu_free_bytes))
            ram_available = max(0, int(resources.ram_available_bytes))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

        if not model_bytes or not gpu_free or not ram_available:
            return False

        gpu_required = (
            int(model_bytes * 1.07)
            + total_context * runtime_tuning._kv_bytes_per_token()
            + 1280 * _MIB
        )
        host_runtime_required = (512 + 256 * slots) * _MIB
        return bool(
            gpu_required <= int(gpu_free * 0.97)
            and host_runtime_required <= int(ram_available * 0.94)
        )

    setattr(vram_first_parallel_feasible, _RESOURCE_MARKER, True)
    runtime_tuning._parallel_resource_feasible = vram_first_parallel_feasible


def _install_bounded_parallel_candidates(runtime_tuning: Any) -> None:
    current = getattr(runtime_tuning, "_parallel_candidates", None)
    if not callable(current):
        return
    if getattr(current, _CANDIDATE_MARKER, False):
        return

    @wraps(current)
    def bounded_parallel_candidates(
        config: Any | None = None,
        model_path: str | None = None,
        resources: Any | None = None,
    ) -> tuple[int, ...]:
        if runtime_tuning._explicit_parallel() is not None:
            return current(config, model_path, resources)
        if not _env_enabled("MMM_LLAMA_MAXIMIZE_PARALLEL", True):
            return current(config, model_path, resources)
        if config is None:
            return current(config, model_path, resources)

        explicit_concurrency = os.environ.get("MMM_LLAMA_CONCURRENT_REQUESTS", "").strip()
        maximum = int(getattr(runtime_tuning, "_MAX_PARALLEL", 8) or 8)
        target = runtime_tuning._parallel_target() if explicit_concurrency else maximum
        target = max(1, min(maximum, int(target)))
        snapshot = resources or runtime_tuning._runtime_resources()
        feasible = tuple(
            slots
            for slots in range(1, target + 1)
            if runtime_tuning._parallel_resource_feasible(slots, config, model_path, snapshot)
        )
        return _shortlist_parallel(feasible or (1,))

    setattr(bounded_parallel_candidates, _CANDIDATE_MARKER, True)
    runtime_tuning._parallel_candidates = bounded_parallel_candidates


def _install_bounded_cold_search(runtime_tuning: Any) -> None:
    """Bound restart-heavy search stages; full mode preserves the original sweep."""
    variants = getattr(runtime_tuning, "_candidate_variants", None)
    ubatches = getattr(runtime_tuning, "_ubatch_candidates", None)
    if callable(variants) and not getattr(variants, _SEARCH_MARKER, False):
        current_variants = variants

        @wraps(current_variants)
        def bounded_variants(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            return _shortlist_variants(current_variants(*args, **kwargs))

        setattr(bounded_variants, _SEARCH_MARKER, True)
        runtime_tuning._candidate_variants = bounded_variants

    if callable(ubatches) and not getattr(ubatches, _SEARCH_MARKER, False):
        current_ubatches = ubatches

        @wraps(current_ubatches)
        def bounded_ubatches(*args: Any, **kwargs: Any) -> tuple[int, ...]:
            return _shortlist_ubatches(current_ubatches(*args, **kwargs))

        setattr(bounded_ubatches, _SEARCH_MARKER, True)
        runtime_tuning._ubatch_candidates = bounded_ubatches


def _install_selection_version(runtime_tuning: Any) -> None:
    current = runtime_tuning._selection_inputs
    if getattr(current, _SELECTION_MARKER, False):
        return
    current = _unwrap_marked(current, _LEGACY_SELECTION_MARKERS)

    @wraps(current)
    def selection_inputs(config: Any) -> dict[str, Any]:
        payload = dict(current(config))
        payload["vram_parallel_policy_version"] = _POLICY_VERSION
        payload["maximize_parallel"] = _env_enabled("MMM_LLAMA_MAXIMIZE_PARALLEL", True)
        payload["autotune_search"] = _search_mode()
        return payload

    setattr(selection_inputs, _SELECTION_MARKER, True)
    runtime_tuning._selection_inputs = selection_inputs


def install(runtime_tuning: Any) -> None:
    """Install safe resource admission plus a bounded, cacheable cold search."""
    _configure_benchmark_defaults()
    _install_resource_admission(runtime_tuning)
    _install_bounded_cold_search(runtime_tuning)
    _install_bounded_parallel_candidates(runtime_tuning)
    _install_selection_version(runtime_tuning)


__all__ = ["install", "validated_active_parallelism"]
