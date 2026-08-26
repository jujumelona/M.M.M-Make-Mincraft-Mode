from __future__ import annotations

"""Bounded real-hardware tuning for llama-server kernel/runtime axes.

This module deliberately does not replace the existing MTP, ubatch, parallel, or
request-cache tuners. It runs first on a cold managed server, chooses the fastest
verified Flash-Attention / logical-batch / K,V-cache configuration, then lets the
existing staged tuner optimize the remaining axes on top of that winner.

Kernel tuning is an optional optimizer. A failed synthetic kernel baseline must never
turn a model that can run with the canonical llama-server configuration into a backend
failure; in that case the native canonical launch remains authoritative.
"""

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any

_SCHEMA = "mmm/llama-kernel-autotune-v1"
_LOCK = threading.RLock()
_ALLOWED_FLASH = ("auto", "on", "off")
_ALLOWED_KV = ("q4_0", "q8_0", "f16")
_BYPASS_ENV = "MMM_LLAMA_KERNEL_BYPASS"


@dataclass(frozen=True)
class KernelConfig:
    flash_attn: str
    batch: int
    cache_type_k: str
    cache_type_v: str


@dataclass(frozen=True)
class KernelProbe:
    config: KernelConfig
    ok: bool
    output_sha256: str
    predicted_tps: float
    prompt_tps: float
    elapsed_seconds: float
    error: str = ""


def _dedupe(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _valid_flash(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in _ALLOWED_FLASH else "on"


def _valid_kv(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in _ALLOWED_KV else "q4_0"


def _int(value: str, default: int, *, minimum: int = 1, maximum: int = 16384) -> int:
    try:
        parsed = int(value.strip()) if value.strip() else int(default)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(maximum, parsed))


def _operator_batch() -> int | None:
    raw = os.environ.get("MMM_LLAMA_BATCH", "").strip()
    return _int(raw, 2048) if raw else None


def _baseline_config() -> KernelConfig:
    generic_kv = _valid_kv(os.environ.get("MMM_KV_CACHE_QUANT", "q4_0"))
    return KernelConfig(
        flash_attn=_valid_flash(os.environ.get("MMM_LLAMA_FLASH_ATTN", "on")),
        batch=_operator_batch() or 2048,
        cache_type_k=_valid_kv(os.environ.get("MMM_LLAMA_CACHE_TYPE_K", generic_kv)),
        cache_type_v=_valid_kv(os.environ.get("MMM_LLAMA_CACHE_TYPE_V", generic_kv)),
    )


def _parse_ints(raw: str, *, maximum: int = 16384) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if 1 <= value <= maximum and value not in values:
            values.append(value)
    return tuple(values)


def _flash_candidates(current: str) -> tuple[str, ...]:
    explicit = os.environ.get("MMM_LLAMA_FLASH_ATTN", "").strip().lower()
    if explicit in _ALLOWED_FLASH:
        return (explicit,)
    raw = os.environ.get("MMM_LLAMA_FLASH_ATTN_CANDIDATES", "auto,on,off")
    parsed = [token.strip().lower() for token in raw.split(",")]
    return _dedupe((current, *(value for value in parsed if value in _ALLOWED_FLASH)))


def _batch_candidates(current: int, hardware: str) -> tuple[int, ...]:
    explicit = _operator_batch()
    if explicit is not None:
        return (explicit,)
    default = "256,512,1024,2048,4096" if "t4" in hardware.casefold() else "512,1024,2048"
    parsed = _parse_ints(os.environ.get("MMM_LLAMA_BATCH_CANDIDATES", default))
    return _dedupe((current, *parsed))


def _cache_candidates(current: KernelConfig, hardware: str) -> tuple[tuple[str, str], ...]:
    explicit_k = os.environ.get("MMM_LLAMA_CACHE_TYPE_K", "").strip().lower()
    explicit_v = os.environ.get("MMM_LLAMA_CACHE_TYPE_V", "").strip().lower()
    generic = os.environ.get("MMM_KV_CACHE_QUANT", "").strip().lower()
    fixed_k = explicit_k if explicit_k in _ALLOWED_KV else (generic if generic in _ALLOWED_KV else "")
    fixed_v = explicit_v if explicit_v in _ALLOWED_KV else (generic if generic in _ALLOWED_KV else "")
    if fixed_k and fixed_v:
        return ((fixed_k, fixed_v),)

    default = (
        "q4_0:q4_0,q8_0:q8_0,f16:f16,q4_0:q8_0,q8_0:f16"
        if "t4" in hardware.casefold()
        else "q4_0:q4_0,q8_0:q8_0,f16:f16"
    )
    values: list[tuple[str, str]] = [(current.cache_type_k, current.cache_type_v)]
    for token in os.environ.get("MMM_LLAMA_KV_PAIR_CANDIDATES", default).split(","):
        parts = token.strip().lower().split(":", 1)
        if len(parts) != 2 or parts[0] not in _ALLOWED_KV or parts[1] not in _ALLOWED_KV:
            continue
        k = fixed_k or parts[0]
        v = fixed_v or parts[1]
        pair = (k, v)
        if pair not in values:
            values.append(pair)
    return tuple(values)


def _replace_option(args: list[str], names: tuple[str, ...], value: str) -> None:
    for name in names:
        if name not in args:
            continue
        index = args.index(name)
        if index + 1 < len(args):
            args[index + 1] = value
            return
    args.extend([names[0], value])


def _active_config(fallback: KernelConfig | None = None) -> KernelConfig:
    base = fallback or _baseline_config()
    return KernelConfig(
        flash_attn=_valid_flash(os.environ.get("MMM_LLAMA_ACTIVE_FLASH_ATTN", base.flash_attn)),
        batch=_int(os.environ.get("MMM_LLAMA_ACTIVE_BATCH", ""), base.batch),
        cache_type_k=_valid_kv(os.environ.get("MMM_LLAMA_ACTIVE_CACHE_TYPE_K", base.cache_type_k)),
        cache_type_v=_valid_kv(os.environ.get("MMM_LLAMA_ACTIVE_CACHE_TYPE_V", base.cache_type_v)),
    )


def _apply_active(config: KernelConfig) -> None:
    os.environ["MMM_LLAMA_ACTIVE_FLASH_ATTN"] = config.flash_attn
    os.environ["MMM_LLAMA_ACTIVE_BATCH"] = str(config.batch)
    os.environ["MMM_LLAMA_ACTIVE_CACHE_TYPE_K"] = config.cache_type_k
    os.environ["MMM_LLAMA_ACTIVE_CACHE_TYPE_V"] = config.cache_type_v
    # The legacy KV tuner uses this as its already-tuned sentinel. Independent K/V
    # remain authoritative through the base-args wrapper below.
    os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] = config.cache_type_k


@contextmanager
def _temporary_server_env(config: KernelConfig) -> Iterator[None]:
    names = (
        "MMM_LLAMA_BATCH",
        "MMM_KV_CACHE_QUANT",
        "MMM_LLAMA_ACTIVE_FLASH_ATTN",
        "MMM_LLAMA_ACTIVE_BATCH",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_K",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_V",
        "MMM_LLAMA_ACTIVE_KV_CACHE",
    )
    old = {name: os.environ.get(name) for name in names}
    try:
        os.environ["MMM_LLAMA_BATCH"] = str(config.batch)
        os.environ["MMM_KV_CACHE_QUANT"] = config.cache_type_k
        _apply_active(config)
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _temporary_kernel_bypass() -> Iterator[None]:
    old = os.environ.get(_BYPASS_ENV)
    try:
        os.environ[_BYPASS_ENV] = "1"
        yield
    finally:
        if old is None:
            os.environ.pop(_BYPASS_ENV, None)
        else:
            os.environ[_BYPASS_ENV] = old


def _score(probe: KernelProbe, baseline: KernelProbe) -> float:
    if not probe.ok or not baseline.ok or probe.output_sha256 != baseline.output_sha256:
        return 0.0
    decode_base = max(1e-9, baseline.predicted_tps)
    decode = probe.predicted_tps / decode_base
    if baseline.prompt_tps > 0 and probe.prompt_tps > 0:
        prompt = probe.prompt_tps / baseline.prompt_tps
        return 0.80 * decode + 0.20 * prompt
    return decode


def _select_stage(current: KernelProbe, candidates: Iterable[KernelProbe], *, minimum_gain: float) -> KernelProbe:
    valid = [probe for probe in candidates if probe.ok and probe.output_sha256 == current.output_sha256]
    if not valid:
        return current
    best = max(valid, key=lambda probe: _score(probe, current))
    return best if _score(best, current) >= max(1.0, minimum_gain) else current


def _cache_path(autotune: Any) -> Path:
    explicit = os.environ.get("MMM_LLAMA_KERNEL_AUTOTUNE_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (autotune._cache_path().parent / "llama-kernel-autotune.json").resolve()


def _fingerprint(autotune: Any, config: Any, binary: str, model_path: str, hardware: str) -> str:
    from .llama_server_efficiency_contract import _quick_file_signature

    path = Path(model_path).expanduser().resolve()
    stat = path.stat()
    baseline = _baseline_config()
    payload = {
        "schema": _SCHEMA,
        "model_id": str(getattr(config, "model_id", "")),
        "model_file": path.name,
        "model_size": int(stat.st_size),
        "model_signature": _quick_file_signature(path),
        "server": autotune._server_version(binary),
        "hardware": hardware,
        "context": int(getattr(config, "max_context", 0) or 0),
        "probe_tokens": min(
            int(getattr(config, "max_new_tokens", 256) or 256),
            autotune._env_int("MMM_LLAMA_AUTOTUNE_TOKENS", autotune._BENCHMARK_OUTPUT_TOKENS),
        ),
        "flash": _flash_candidates(baseline.flash_attn),
        "batch": _batch_candidates(baseline.batch, hardware),
        "kv_pairs": _cache_candidates(baseline, hardware),
        "operator": {
            "flash": os.environ.get("MMM_LLAMA_FLASH_ATTN", ""),
            "batch": os.environ.get("MMM_LLAMA_BATCH", ""),
            "k": os.environ.get("MMM_LLAMA_CACHE_TYPE_K", ""),
            "v": os.environ.get("MMM_LLAMA_CACHE_TYPE_V", ""),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load(autotune: Any, fingerprint: str) -> KernelConfig | None:
    try:
        payload = json.loads(_cache_path(autotune).read_text(encoding="utf-8"))
        if payload.get("schema") != _SCHEMA or payload.get("fingerprint") != fingerprint:
            return None
        raw = payload.get("selected") or {}
        selected = KernelConfig(**raw)
        if selected.flash_attn not in _ALLOWED_FLASH:
            return None
        if selected.cache_type_k not in _ALLOWED_KV or selected.cache_type_v not in _ALLOWED_KV:
            return None
        if selected.batch <= 0:
            return None
        return selected
    except Exception:
        return None


def _save(autotune: Any, fingerprint: str, selected: KernelConfig, probes: Iterable[KernelProbe]) -> None:
    path = _cache_path(autotune)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": _SCHEMA,
        "fingerprint": fingerprint,
        "selected": asdict(selected),
        "probes": [asdict(probe) for probe in probes],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_probe(
    autotune: Any,
    binary: str,
    model_path: str,
    model_config: Any,
    request: Any,
    kernel: KernelConfig,
    *,
    propagate_resource_failure: bool = False,
) -> KernelProbe:
    started = time.perf_counter()
    try:
        variant = autotune.ServerVariant(
            name=f"kernel|fa-{kernel.flash_attn}|b-{kernel.batch}|k-{kernel.cache_type_k}|v-{kernel.cache_type_v}"
        )
        with _temporary_server_env(kernel):
            probe_kwargs: dict[str, Any] = {
                "probe_tokens": min(
                    int(getattr(model_config, "max_new_tokens", 256) or 256),
                    autotune._env_int(
                        "MMM_LLAMA_AUTOTUNE_TOKENS",
                        autotune._BENCHMARK_OUTPUT_TOKENS,
                    ),
                )
            }
            if bool(
                getattr(
                    autotune._mmm_run_tuning_variant,
                    "_mmm_resource_failure_propagation",
                    False,
                )
            ):
                probe_kwargs["propagate_resource_failure"] = (
                    propagate_resource_failure
                )
            probe = autotune._mmm_run_tuning_variant(
                binary,
                model_path,
                model_config,
                autotune._compact_benchmark_request(request),
                variant,
                **probe_kwargs,
            )
        return KernelProbe(
            config=kernel,
            ok=bool(getattr(probe, "ok", False)),
            output_sha256=str(getattr(probe, "output_sha256", "")),
            predicted_tps=float(getattr(probe, "predicted_tps", 0.0)),
            prompt_tps=float(getattr(probe, "prompt_tps", 0.0)),
            elapsed_seconds=float(getattr(probe, "elapsed_seconds", 0.0)),
            error=str(getattr(probe, "error", "")),
        )
    except Exception as exc:
        if bool(getattr(exc, "_mmm_recoverable_resource_failure", False)):
            raise
        return KernelProbe(
            config=kernel,
            ok=False,
            output_sha256="",
            predicted_tps=0.0,
            prompt_tps=0.0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _benchmark(autotune: Any, binary: str, model_path: str, model_config: Any, request: Any, hardware: str) -> tuple[KernelConfig, tuple[KernelProbe, ...]]:
    minimum_gain = autotune._env_float("MMM_LLAMA_STAGE_MIN_GAIN", 1.01)
    current_config = _baseline_config()
    probes: list[KernelProbe] = []
    current = _run_probe(
        autotune,
        binary,
        model_path,
        model_config,
        request,
        current_config,
        propagate_resource_failure=True,
    )
    probes.append(current)
    if not current.ok or current.predicted_tps <= 0:
        detail = current.error.strip() or "probe returned no usable throughput"
        raise RuntimeError(
            "baseline llama kernel configuration failed: "
            f"fa={current_config.flash_attn} batch={current_config.batch} "
            f"kv={current_config.cache_type_k}/{current_config.cache_type_v}; "
            f"{detail}"
        )

    flash_probes: list[KernelProbe] = [current]
    for value in _flash_candidates(current_config.flash_attn):
        candidate = KernelConfig(value, current_config.batch, current_config.cache_type_k, current_config.cache_type_v)
        if candidate == current.config:
            continue
        probe = _run_probe(autotune, binary, model_path, model_config, request, candidate)
        probes.append(probe)
        flash_probes.append(probe)
    current = _select_stage(current, flash_probes, minimum_gain=minimum_gain)
    current_config = current.config

    batch_probes: list[KernelProbe] = [current]
    for value in _batch_candidates(current_config.batch, hardware):
        candidate = KernelConfig(current_config.flash_attn, value, current_config.cache_type_k, current_config.cache_type_v)
        if candidate == current.config:
            continue
        probe = _run_probe(autotune, binary, model_path, model_config, request, candidate)
        probes.append(probe)
        batch_probes.append(probe)
    current = _select_stage(current, batch_probes, minimum_gain=minimum_gain)
    current_config = current.config

    cache_probes: list[KernelProbe] = [current]
    for cache_k, cache_v in _cache_candidates(current_config, hardware):
        candidate = KernelConfig(current_config.flash_attn, current_config.batch, cache_k, cache_v)
        if candidate == current.config:
            continue
        probe = _run_probe(autotune, binary, model_path, model_config, request, candidate)
        probes.append(probe)
        cache_probes.append(probe)
    current = _select_stage(current, cache_probes, minimum_gain=minimum_gain)
    return current.config, tuple(probes)


def install(autotune: Any, runtime_tuning: Any) -> None:
    """Install the outer cold-start kernel tuner exactly once."""
    with _LOCK:
        if getattr(autotune, "_mmm_kernel_autotune_installed", False):
            return

        current_base = autotune._base_args

        @wraps(current_base)
        def kernel_base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
            if os.environ.get(_BYPASS_ENV, "").strip() == "1":
                return list(current_base(binary, model_path, config, port))
            args = list(current_base(binary, model_path, config, port))
            baseline = _baseline_config()
            active = _active_config(baseline)
            explicit_flash = os.environ.get("MMM_LLAMA_FLASH_ATTN", "").strip().lower()
            flash = explicit_flash if explicit_flash in _ALLOWED_FLASH else active.flash_attn
            batch = _operator_batch() or active.batch
            generic_kv = _valid_kv(os.environ.get("MMM_KV_CACHE_QUANT", "q4_0"))
            k = _valid_kv(os.environ.get("MMM_LLAMA_CACHE_TYPE_K", active.cache_type_k or generic_kv))
            v = _valid_kv(os.environ.get("MMM_LLAMA_CACHE_TYPE_V", active.cache_type_v or generic_kv))
            _replace_option(args, ("--flash-attn", "-fa"), flash)
            _replace_option(args, ("--batch-size", "-b"), str(batch))
            _replace_option(args, ("--cache-type-k", "-ctk"), k)
            _replace_option(args, ("--cache-type-v", "-ctv"), v)
            # Never leave physical batch above logical batch after an auto batch choice.
            for name in ("--ubatch-size", "-ub"):
                if name in args:
                    index = args.index(name)
                    if index + 1 < len(args):
                        args[index + 1] = str(min(batch, _int(args[index + 1], batch)))
                    break
            return args

        kernel_base_args._mmm_kernel_axes = True  # type: ignore[attr-defined]
        autotune._base_args = kernel_base_args

        # Expand the existing ubatch stage on T4; respect an explicit operator value.
        original_ubatch_candidates = runtime_tuning._ubatch_candidates

        def ubatch_candidates(autotune_module: Any) -> tuple[int, ...]:
            explicit = os.environ.get("MMM_LLAMA_UBATCH", "").strip()
            batch = autotune_module._env_int("MMM_LLAMA_BATCH", 2048)
            if explicit:
                return (min(batch, _int(explicit, 512)),)
            hardware = autotune_module._hardware_identity()
            default = "128,256,512,1024,2048" if "t4" in hardware.casefold() else "256,512,1024"
            raw = os.environ.get("MMM_LLAMA_UBATCH_CANDIDATES", default)
            values = tuple(value for value in _parse_ints(raw, maximum=batch) if value <= batch)
            current = min(batch, autotune_module._env_int("MMM_LLAMA_UBATCH", 512))
            return _dedupe((current, *values)) or original_ubatch_candidates(autotune_module)

        ubatch_candidates._mmm_t4_expanded_ubatch = True  # type: ignore[attr-defined]
        runtime_tuning._ubatch_candidates = ubatch_candidates

        current_fingerprint = autotune._fingerprint

        @wraps(current_fingerprint)
        def fingerprint(config: Any, binary: str, model_path: str) -> str:
            active = _active_config()
            payload = {
                "base": current_fingerprint(config, binary, model_path),
                "kernel_schema": _SCHEMA,
                "flash_attn": active.flash_attn,
                "batch": active.batch,
                "cache_type_k": active.cache_type_k,
                "cache_type_v": active.cache_type_v,
            }
            return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        fingerprint._mmm_kernel_fingerprint = True  # type: ignore[attr-defined]
        autotune._fingerprint = fingerprint

        current_ensure = autotune.ensure_tuned_server

        @wraps(current_ensure)
        def ensure_kernel_tuned(config: Any, request: Any) -> str:
            process = getattr(autotune, "_MANAGED_PROCESS", None)
            managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
            if process is not None and process.poll() is None and managed_url:
                return current_ensure(config, request)
            if not autotune._env_bool("MMM_LLAMA_KERNEL_AUTOTUNE", True):
                return current_ensure(config, request)
            if autotune._external_server_is_ready():
                return current_ensure(config, request)

            with _LOCK:
                binary = autotune._server_binary()
                if binary is None:
                    return current_ensure(config, request)
                model_path = autotune._resolve_model_path(config)
                hardware = autotune._hardware_identity()
                fp = _fingerprint(autotune, config, binary, model_path, hardware)
                selected = _load(autotune, fp)
                if selected is None:
                    try:
                        selected, probes = _benchmark(
                            autotune, binary, model_path, config, request, hardware
                        )
                    except Exception as exc:
                        # Kernel search is an optional optimization layer. Retry the
                        # canonical native owner with this layer physically bypassed;
                        # if the model/server is genuinely unusable, that canonical
                        # owner will surface its real launch error instead.
                        print(
                            "native llama-server: kernel autotune skipped; canonical baseline",
                            f" reason={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        with _temporary_kernel_bypass():
                            return current_ensure(config, request)
                    _save(autotune, fp, selected, probes)
                    print(
                        "native llama-server: kernel autotune selected",
                        f" fa={selected.flash_attn}",
                        f" batch={selected.batch}",
                        f" kv={selected.cache_type_k}/{selected.cache_type_v}",
                        flush=True,
                    )
                _apply_active(selected)
                # The inner MTP/ubatch/cache tuner must see the selected logical batch,
                # but the operator environment is restored after the persistent server
                # has launched.
                old_batch = os.environ.get("MMM_LLAMA_BATCH")
                old_kv = os.environ.get("MMM_KV_CACHE_QUANT")
                try:
                    os.environ["MMM_LLAMA_BATCH"] = str(selected.batch)
                    os.environ["MMM_KV_CACHE_QUANT"] = selected.cache_type_k
                    return current_ensure(config, request)
                finally:
                    if old_batch is None:
                        os.environ.pop("MMM_LLAMA_BATCH", None)
                    else:
                        os.environ["MMM_LLAMA_BATCH"] = old_batch
                    if old_kv is None:
                        os.environ.pop("MMM_KV_CACHE_QUANT", None)
                    else:
                        os.environ["MMM_KV_CACHE_QUANT"] = old_kv

        ensure_kernel_tuned._mmm_kernel_autotune = True  # type: ignore[attr-defined]
        autotune.ensure_tuned_server = ensure_kernel_tuned
        autotune._mmm_kernel_autotune_installed = True


__all__ = [
    "KernelConfig",
    "KernelProbe",
    "_batch_candidates",
    "_cache_candidates",
    "_flash_candidates",
    "_select_stage",
    "install",
]