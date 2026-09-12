from __future__ import annotations

"""VRAM-first admission and no-reload fast-start policy for managed llama-server.

Normal inference must not benchmark by repeatedly reloading the same GGUF.  The default
``fast`` mode chooses a safe parallel width from live VRAM/RAM before the single managed
server launch and disables optional kernel/KV cold sweeps.  ``balanced`` and ``full`` stay
available as explicit offline tuning modes.
"""

import json
import os
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any

_POLICY_VERSION = 6
_MIB = 1024 * 1024
_RESOURCE_MARKER = "_mmm_vram_parallel_resource_policy_v6"
_SELECTION_MARKER = "_mmm_vram_parallel_selection_policy_v6"
_CANDIDATE_MARKER = "_mmm_vram_parallel_bounded_candidates_v6"
_SEARCH_MARKER = "_mmm_llama_bounded_cold_search_v6"
_FAST_START_MARKER = "_mmm_llama_no_reload_fast_start_v6"
_LEGACY_RESOURCE_MARKERS = (
    "_mmm_vram_parallel_resource_policy_v1",
    "_mmm_vram_parallel_resource_policy_v2",
    "_mmm_vram_parallel_resource_policy_v3",
    "_mmm_vram_parallel_resource_policy_v4",
    "_mmm_vram_parallel_resource_policy_v5",
)
_LEGACY_SELECTION_MARKERS = (
    "_mmm_vram_parallel_selection_policy_v1",
    "_mmm_vram_parallel_selection_policy_v2",
    "_mmm_vram_parallel_selection_policy_v3",
    "_mmm_vram_parallel_selection_policy_v4",
    "_mmm_vram_parallel_selection_policy_v5",
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
    """Default request startup to zero cold-search reloads.

    Server startup itself keeps its independent readiness timeout.  These values only apply
    when an operator explicitly invokes inline/offline autotuning.
    """

    mode = _search_mode()
    if mode == "fast":
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_TOKENS", "8")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_MAX_SECONDS", "30")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS", "15")
        # These wrappers previously ignored the base owner's non-inline default and launched
        # several extra full model processes on the first request.
        os.environ.setdefault("MMM_LLAMA_KERNEL_AUTOTUNE", "0")
        os.environ.setdefault("MMM_LLAMA_KV_AUTOTUNE", "0")
    elif mode == "balanced":
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_TOKENS", "24")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_MAX_SECONDS", "120")
        os.environ.setdefault("MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS", "30")
    # full mode deliberately leaves operator/offline search settings untouched.


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
    """Bound explicit benchmark widths; fast runtime uses heuristic launch selection."""

    ordered = tuple(sorted({max(1, int(value)) for value in values}))
    if not ordered:
        return (1,)
    active_mode = mode or _search_mode()
    if active_mode == "full":
        return ordered
    if active_mode == "fast":
        # The fast path chooses its actual live width before launch, without a p1/pN
        # benchmark pair.  If an explicit benchmark is requested, keep one reference only.
        return (1,)

    high = [value for value in ordered if value > 1]
    if not high:
        return (1,)
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


def _shortlist_cache_reuse(values: Iterable[int], mode: str | None = None) -> tuple[int, ...]:
    ordered = tuple(dict.fromkeys(max(0, int(value)) for value in values))
    active_mode = mode or _search_mode()
    if active_mode == "full":
        return ordered
    if active_mode == "fast":
        # Request-scoped cache reuse used to create another fully loaded temporary server.
        # Keep it disabled unless an operator asks for balanced/full offline tuning.
        return ()
    if not ordered:
        return ()
    return tuple(dict.fromkeys((ordered[0], ordered[-1])))


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
        # Speculative decoding is never enabled without exact-output validation.
        return (baseline,)

    mtp = [value for value in variants if str(getattr(value, "spec_type", "")) == "draft-mtp"]
    ngram = [
        value
        for value in variants
        if str(getattr(value, "spec_type", "")).startswith("ngram-")
    ]
    chosen: list[Any] = [baseline]
    if mtp:
        chosen.append(
            min(
                mtp,
                key=lambda item: abs(int(getattr(item, "draft_n_max", 1)) - 4),
            )
        )
    if ngram:
        preferred = next(
            (
                item
                for item in ngram
                if str(getattr(item, "spec_type", "")) == "ngram-map-k"
            ),
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
    if not callable(current) or getattr(current, _CANDIDATE_MARKER, False):
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
            if runtime_tuning._parallel_resource_feasible(
                slots, config, model_path, snapshot
            )
        )
        return _shortlist_parallel(feasible or (1,))

    setattr(bounded_parallel_candidates, _CANDIDATE_MARKER, True)
    runtime_tuning._parallel_candidates = bounded_parallel_candidates


def _install_bounded_cold_search(runtime_tuning: Any) -> None:
    """Bound restart-heavy explicit tuning stages; full mode preserves the sweep."""

    variants = getattr(runtime_tuning, "_candidate_variants", None)
    ubatches = getattr(runtime_tuning, "_ubatch_candidates", None)
    cache_reuse = getattr(runtime_tuning, "_cache_reuse_candidates", None)

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

    if callable(cache_reuse) and not getattr(cache_reuse, _SEARCH_MARKER, False):
        current_cache_reuse = cache_reuse

        @wraps(current_cache_reuse)
        def bounded_cache_reuse(*args: Any, **kwargs: Any) -> tuple[int, ...]:
            return _shortlist_cache_reuse(
                current_cache_reuse(*args, **kwargs)
            )

        setattr(bounded_cache_reuse, _SEARCH_MARKER, True)
        runtime_tuning._cache_reuse_candidates = bounded_cache_reuse


def _recommended_parallel(
    runtime_tuning: Any,
    config: Any,
    model_path: str,
) -> int:
    explicit = runtime_tuning._explicit_parallel()
    if explicit is not None:
        return int(explicit)
    if runtime_tuning._performance_mode() == "latency" and not os.environ.get(
        "MMM_LLAMA_CONCURRENT_REQUESTS", ""
    ).strip():
        return 1
    if not _env_enabled("MMM_LLAMA_MAXIMIZE_PARALLEL", True):
        return 1

    resources = runtime_tuning._runtime_resources()
    maximum = int(getattr(runtime_tuning, "_MAX_PARALLEL", 8) or 8)
    target = runtime_tuning._parallel_target()
    target = max(1, min(maximum, int(target)))
    for slots in range(target, 0, -1):
        if runtime_tuning._parallel_resource_feasible(
            slots, config, model_path, resources
        ):
            return slots
    return 1


def _install_fast_start_profile(runtime_tuning: Any, autotune: Any) -> None:
    """Choose live server width once, before launch, with no benchmark process."""

    current = autotune.ensure_tuned_server
    if getattr(current, _FAST_START_MARKER, False):
        return

    @wraps(current)
    def ensure_fast_start(config: Any, request: Any) -> str:
        if _search_mode() != "fast":
            return current(config, request)

        # Reuse an already-running managed server before touching model metadata or
        # recomputing launch width.  The base owner validates the process/URL pair and
        # returns it without any model-path or health-HTTP work.
        managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
        managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
        if managed_process is not None and managed_url:
            try:
                if managed_process.poll() is None:
                    return current(config, request)
            except Exception:
                pass

        # An operator-set value is authoritative.  Otherwise persist the auto-selected
        # width in-process so fingerprints, stale-runtime checks and the launch receipt all
        # observe the same selection on every subsequent request.
        if not os.environ.get("MMM_LLAMA_PARALLEL", "").strip():
            model_path = autotune._resolve_model_path(config)
            os.environ["MMM_LLAMA_PARALLEL"] = str(
                _recommended_parallel(runtime_tuning, config, model_path)
            )

        return current(config, request)

    setattr(ensure_fast_start, _FAST_START_MARKER, True)
    autotune.ensure_tuned_server = ensure_fast_start


def _install_selection_version(runtime_tuning: Any) -> None:
    current = runtime_tuning._selection_inputs
    if getattr(current, _SELECTION_MARKER, False):
        return
    current = _unwrap_marked(current, _LEGACY_SELECTION_MARKERS)

    @wraps(current)
    def selection_inputs(config: Any) -> dict[str, Any]:
        payload = dict(current(config))
        payload["vram_parallel_policy_version"] = _POLICY_VERSION
        payload["maximize_parallel"] = _env_enabled(
            "MMM_LLAMA_MAXIMIZE_PARALLEL", True
        )
        payload["autotune_search"] = _search_mode()
        payload["cold_reload_policy"] = "none-fast"
        return payload

    setattr(selection_inputs, _SELECTION_MARKER, True)
    runtime_tuning._selection_inputs = selection_inputs


def install(runtime_tuning: Any, autotune: Any | None = None) -> None:
    """Install safe admission, bounded explicit tuning, and zero-reload fast startup."""

    if autotune is None:
        from . import llama_server_autotune as autotune

    _configure_benchmark_defaults()
    _install_resource_admission(runtime_tuning)
    _install_bounded_cold_search(runtime_tuning)
    _install_bounded_parallel_candidates(runtime_tuning)
    _install_selection_version(runtime_tuning)
    _install_fast_start_profile(runtime_tuning, autotune)


__all__ = [
    "_recommended_parallel",
    "_shortlist_cache_reuse",
    "_shortlist_parallel",
    "_shortlist_ubatches",
    "_shortlist_variants",
    "install",
    "validated_active_parallelism",
]
