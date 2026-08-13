from __future__ import annotations

"""Measured max-throughput policy for Qwen3.5-9B-MTP on Tesla T4."""

import hashlib
import json
import os
import time
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .qwen35_mtp_hotpath_contract import (
    _context_size,
    _is_qwen35_mtp,
    _reclaim_prior_mmm_server,
)
from .qwen35_t4_single_stream_tuning import (
    _EXPECTED_DIGEST,
    _EXPECTED_OBJECT,
    _EXPECTED_TEXT,
    _is_t4_runtime,
    _probe,
    _restore_kv_override,
    _set_kv_override,
    _valid,
    _variant,
)

_SCHEMA_VERSION = "mmm/qwen35-t4-max-speed-v2-lazy-kv"
_MARKER = "_mmm_qwen35_t4_max_speed_v2"
_DEFAULT_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16)
_DEFAULT_P_MIN = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
_DEFAULT_UBATCHES = (512, 1024, 2048)
_DEFAULT_KV = ("native-default", "q8_0", "q4_0")
_DEFAULT_CONTEXT_BUCKETS = (2048, 8192, 16384, 28672)
_KV_OVERRIDE_ENV = "MMM_QWEN35_T4_KV_OVERRIDE"
_CTX_OVERRIDE_ENV = "MMM_QWEN35_MTP_CTX"


def _enabled() -> bool:
    return os.environ.get("MMM_QWEN35_T4_MAX_SPEED", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _parse_ints(raw: str, defaults: tuple[int, ...], *, minimum: int, maximum: int) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if minimum <= value <= maximum and value not in values:
            values.append(value)
    return tuple(values or defaults)


def _widths() -> tuple[int, ...]:
    return _parse_ints(
        os.environ.get("MMM_QWEN35_T4_MAX_WIDTHS", ",".join(map(str, _DEFAULT_WIDTHS))),
        _DEFAULT_WIDTHS,
        minimum=1,
        maximum=32,
    )


def _ubatches() -> tuple[int, ...]:
    return _parse_ints(
        os.environ.get("MMM_QWEN35_T4_UBATCHES", ",".join(map(str, _DEFAULT_UBATCHES))),
        _DEFAULT_UBATCHES,
        minimum=64,
        maximum=4096,
    )


def _p_min_candidates() -> tuple[float, ...]:
    values: list[float] = []
    raw = os.environ.get("MMM_QWEN35_T4_MAX_P_MIN", ",".join(map(str, _DEFAULT_P_MIN)))
    for token in raw.split(","):
        try:
            value = round(float(token.strip()), 4)
        except ValueError:
            continue
        if 0.0 <= value < 1.0 and value not in values:
            values.append(value)
    if 0.0 not in values:
        values.insert(0, 0.0)
    return tuple(values or _DEFAULT_P_MIN)


def _kv_mode(config: Any) -> str:
    raw = os.environ.get("MMM_QWEN35_T4_KV_MODE", "").strip().lower()
    aliases = {
        "auto": "auto",
        "native": "native-default",
        "native-default": "native-default",
        "f16": "f16",
        "q8": "q8_0",
        "q8_0": "q8_0",
        "q4": "q4_0",
        "q4_0": "q4_0",
    }
    if raw:
        return aliases.get(raw, "auto")
    extra = getattr(config, "extra", {})
    if isinstance(extra, dict) and extra.get("kv_cache_autotune") is False:
        manual = str(extra.get("kv_cache_quant", "q4_0")).strip().lower()
        return aliases.get(manual, "q4_0")
    return "auto"


def _kv_candidates() -> tuple[str, ...]:
    aliases = {
        "native": "native-default",
        "native-default": "native-default",
        "f16": "f16",
        "q8": "q8_0",
        "q8_0": "q8_0",
        "q4": "q4_0",
        "q4_0": "q4_0",
    }
    values: list[str] = []
    raw = os.environ.get("MMM_QWEN35_T4_MAX_KV", ",".join(_DEFAULT_KV))
    for token in raw.split(","):
        value = aliases.get(token.strip().lower(), "")
        if value and value not in values:
            values.append(value)
    if "native-default" not in values:
        values.insert(0, "native-default")
    return tuple(values or _DEFAULT_KV)


def _context_buckets(config: Any) -> tuple[int, ...]:
    max_ctx = max(2048, _context_size(config))
    raw = os.environ.get(
        "MMM_QWEN35_T4_KV_CONTEXT_BUCKETS",
        ",".join(map(str, _DEFAULT_CONTEXT_BUCKETS)),
    )
    values = _parse_ints(raw, _DEFAULT_CONTEXT_BUCKETS, minimum=512, maximum=max_ctx)
    cap = max(512, max_ctx - min(1536, max_ctx // 4))
    clipped = sorted({min(value, cap) for value in values if min(value, cap) >= 512})
    return tuple(clipped or (min(2048, cap),))


def _minimum_gain() -> float:
    try:
        return max(1.0, float(os.environ.get("MMM_QWEN35_T4_MAX_MIN_GAIN", "1.005")))
    except ValueError:
        return 1.005


def _probe_tokens(autotune: Any, config: Any) -> int:
    value = autotune._env_int("MMM_QWEN35_T4_MAX_PROBE_TOKENS", 768, minimum=256)
    return min(int(getattr(config, "max_new_tokens", value) or value), value)


def _cache_path() -> Path:
    raw = os.environ.get("MMM_QWEN35_T4_MAX_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".cache" / "mmm" / "qwen35-t4-max-speed-v2.json").resolve()


def _fingerprint(autotune: Any, config: Any, binary: str, model_path: str) -> str:
    hardware = ""
    getter = getattr(autotune, "_hardware_identity", None)
    if callable(getter):
        try:
            hardware = str(getter())
        except Exception:
            pass
    payload = {
        "schema": _SCHEMA_VERSION,
        "base": autotune._fingerprint(config, binary, model_path),
        "hardware": hardware,
        "ctx": _context_size(config),
        "widths": list(_widths()),
        "p_min": list(_p_min_candidates()),
        "ubatches": list(_ubatches()),
        "kv_mode": _kv_mode(config),
        "kv": list(_kv_candidates()),
        "buckets": list(_context_buckets(config)),
        "probe_tokens": _probe_tokens(autotune, config),
        "expected": _EXPECTED_DIGEST,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tps(probe: Any) -> float:
    return float(getattr(probe, "predicted_tps", 0.0) or 0.0)


def _best(baseline: Any, probes: list[Any]) -> Any:
    eligible = [baseline] + [
        probe for probe in probes if probe is not baseline and _valid(probe, baseline)
    ]
    best = max(eligible, key=_tps)
    if best is not baseline and _tps(best) < _tps(baseline) * _minimum_gain():
        return baseline
    return best


def _log(probe: Any, *, context: int | None = None) -> None:
    detail = (
        f"{_tps(probe):.2f} tok/s"
        if bool(getattr(probe, "ok", False))
        else str(getattr(probe, "error", "failed"))
    )
    ctx = f" ctx~{context}" if context else ""
    print(
        "llama server: T4 max-speed probe",
        f"{probe.variant.name}",
        f"kv={getattr(probe, 'kv', 'native-default')}{ctx}",
        f"-> {detail}",
        flush=True,
    )


def _measure_core(
    autotune: Any, binary: str, model_path: str, config: Any
) -> tuple[Any, float, float]:
    batch = autotune._env_int("MMM_LLAMA_BATCH", 2048)
    ubatches = tuple(value for value in _ubatches() if value <= batch) or (min(batch, 512),)
    seed_ubatch = ubatches[0]
    baseline = _probe(
        autotune,
        binary,
        model_path,
        config,
        _variant(autotune, name="qwen35-t4-max-baseline", ubatch=seed_ubatch),
    )
    if not bool(getattr(baseline, "ok", False)) or _tps(baseline) <= 0:
        raise RuntimeError("T4 max-speed baseline probe failed")
    probes: list[Any] = [baseline]
    width_probes: list[Any] = []
    for width in _widths():
        probe = _probe(
            autotune,
            binary,
            model_path,
            config,
            _variant(
                autotune,
                name=f"qwen35-t4-max-mtp-{width}",
                ubatch=seed_ubatch,
                width=width,
            ),
        )
        probes.append(probe)
        if _valid(probe, baseline):
            width_probes.append(probe)

    for seed in sorted(width_probes, key=_tps, reverse=True)[:2]:
        width = int(seed.variant.draft_n_max)
        for p_min in _p_min_candidates():
            if p_min == 0.0:
                continue
            probes.append(
                _probe(
                    autotune,
                    binary,
                    model_path,
                    config,
                    _variant(
                        autotune,
                        name=f"qwen35-t4-max-mtp-{width}|pm{p_min:g}",
                        ubatch=seed_ubatch,
                        width=width,
                        p_min=p_min,
                    ),
                )
            )

    selected_probe = _best(baseline, probes)
    selected = selected_probe.variant
    ubatch_probes = [selected_probe]
    for ubatch in ubatches:
        if ubatch == int(selected.ubatch):
            continue
        probe = _probe(
            autotune,
            binary,
            model_path,
            config,
            _variant(
                autotune,
                name=f"{selected.name}|ub{ubatch}",
                ubatch=ubatch,
                width=int(selected.draft_n_max),
                p_min=float(selected.draft_p_min),
            ),
        )
        probes.append(probe)
        if _valid(probe, baseline):
            ubatch_probes.append(probe)
    winner = max(ubatch_probes, key=_tps)
    if _tps(winner) >= _tps(selected_probe) * _minimum_gain():
        selected_probe = winner
        selected = winner.variant

    for probe in probes:
        _log(probe)
    return selected, _tps(baseline), _tps(selected_probe)


def _request_tokens(request: Any) -> int:
    total = 0
    for message in getattr(request, "messages", ()) or ():
        if not isinstance(message, dict):
            continue
        value = message.get("content", "")
        if isinstance(value, str):
            total += len(value)
        else:
            try:
                total += len(json.dumps(value, ensure_ascii=False))
            except Exception:
                pass
    return max(1, total // 3)


def _bucket_for_request(config: Any, request: Any) -> int:
    tokens = _request_tokens(request)
    buckets = _context_buckets(config)
    for bucket in buckets:
        if tokens <= bucket:
            return bucket
    return buckets[-1]


def _context_benchmark(bucket: int) -> Any:
    filler = "x " * max(0, bucket - 1024)
    return SimpleNamespace(
        messages=(
            {
                "role": "system",
                "content": (
                    "Speed calibration only. Ignore filler. Thinking is disabled. "
                    "Return exactly the minified JSON payload after TARGET."
                ),
            },
            {"role": "user", "content": f"{filler}\nTARGET:{_EXPECTED_TEXT}"},
        )
    )


def _context_probe(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    variant: Any,
    *,
    kv: str,
    bucket: int,
) -> Any:
    import httpx

    process = None
    original_kv = _set_kv_override(kv)
    original_ctx = os.environ.get(_CTX_OVERRIDE_ENV)
    max_ctx = _context_size(config)
    os.environ[_CTX_OVERRIDE_ENV] = str(
        min(max_ctx, bucket + min(1536, max_ctx // 4))
    )
    started = time.perf_counter()
    try:
        port = autotune._free_port(autotune._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910))
        process = autotune._start_server(binary, model_path, config, variant, port)
        url = autotune._wait_ready(process, port)
        payload = {
            "model": "local",
            "messages": [dict(message) for message in _context_benchmark(bucket).messages],
            "max_tokens": _probe_tokens(autotune, config),
            "temperature": 0.0,
            "seed": 1234,
            "cache_prompt": False,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        response = httpx.post(
            f"{url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=autotune._env_int("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", 300),
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else {}
        message = message if isinstance(message, dict) else {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or "")
        try:
            parsed = json.loads(content.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        timings = data.get("timings") or {}
        usage = data.get("usage") or {}
        predicted_tokens = int(
            timings.get("predicted_n") or usage.get("completion_tokens") or 0
        )
        predicted_tps = float(timings.get("predicted_per_second") or 0.0)
        elapsed = time.perf_counter() - started
        if predicted_tps <= 0 and predicted_tokens > 0 and elapsed > 0:
            predicted_tps = predicted_tokens / elapsed
        ok = parsed == _EXPECTED_OBJECT and not reasoning.strip() and predicted_tokens > 0
        return SimpleNamespace(
            variant=variant,
            kv=kv,
            ok=ok,
            output_sha256=_EXPECTED_DIGEST if ok else "",
            predicted_tokens=predicted_tokens,
            predicted_tps=predicted_tps,
            prompt_tps=float(timings.get("prompt_per_second") or 0.0),
            elapsed_seconds=elapsed,
            error="" if ok else "context benchmark payload mismatch",
        )
    except Exception as exc:
        return SimpleNamespace(
            variant=variant,
            kv=kv,
            ok=False,
            output_sha256="",
            predicted_tokens=0,
            predicted_tps=0.0,
            prompt_tps=0.0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        autotune._stop_server(process)
        _restore_kv_override(original_kv)
        if original_ctx is None:
            os.environ.pop(_CTX_OVERRIDE_ENV, None)
        else:
            os.environ[_CTX_OVERRIDE_ENV] = original_ctx


def _measure_kv_bucket(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    selected: Any,
    bucket: int,
) -> tuple[str, dict[str, float]]:
    probes = [
        _context_probe(
            autotune, binary, model_path, config, selected, kv=kv, bucket=bucket
        )
        for kv in _kv_candidates()
    ]
    valid = [p for p in probes if bool(getattr(p, "ok", False)) and _tps(p) > 0]
    if not valid:
        raise RuntimeError(f"no valid KV candidate for context bucket {bucket}")
    native = next((p for p in valid if p.kv == "native-default"), valid[0])
    best = max(valid, key=_tps)
    if best is not native and _tps(best) < _tps(native) * _minimum_gain():
        best = native
    for probe in probes:
        _log(probe, context=bucket)
    return str(best.kv), {str(probe.kv): _tps(probe) for probe in valid}


def _load_cache(fingerprint: str, autotune: Any) -> dict[str, Any] | None:
    try:
        payload = json.loads(_cache_path().read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("schema") != _SCHEMA_VERSION or payload.get("fingerprint") != fingerprint:
        return None
    raw = payload.get("selected")
    if not isinstance(raw, dict):
        return None
    try:
        payload["selected_variant"] = autotune.ServerVariant(**raw)
        payload["kv_policy"] = {
            str(k): str(v) for k, v in dict(payload.get("kv_policy", {})).items()
        }
        return payload
    except (TypeError, ValueError):
        return None


def _save_cache(
    fingerprint: str,
    selected: Any,
    baseline_tps: float,
    selected_tps: float,
    kv_policy: dict[str, str],
    kv_measurements: dict[str, Any],
) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            {
                "schema": _SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "selected": asdict(selected),
                "baseline_tps": baseline_tps,
                "selected_tps": selected_tps,
                "kv_policy": kv_policy,
                "kv_measurements": kv_measurements,
            },
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _manual_policy(config: Any) -> dict[str, str]:
    mode = _kv_mode(config)
    if mode == "auto":
        return {}
    return {str(bucket): mode for bucket in _context_buckets(config)}


def _export(selected: Any, kv: str) -> None:
    os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = str(selected.spec_type)
    os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = str(selected.draft_n_max)
    os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "1"
    os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(selected.ubatch)
    os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = "0"
    os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] = f"{selected.draft_p_min:g}"
    os.environ["MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE"] = "single_stream_max"
    os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] = kv
    if kv == "native-default":
        os.environ.pop(_KV_OVERRIDE_ENV, None)
    else:
        os.environ[_KV_OVERRIDE_ENV] = kv


def _stop_managed(autotune: Any) -> None:
    shutdown = getattr(autotune, "_shutdown_managed_server", None)
    if callable(shutdown):
        shutdown()
    else:
        process = getattr(autotune, "_MANAGED_PROCESS", None)
        autotune._stop_server(process)
        autotune._MANAGED_PROCESS = None
        autotune._MANAGED_URL = ""


def install(autotune: Any) -> None:
    """Install exhaustive core tuning plus lazy context-aware KV selection."""
    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_max_speed(config: Any, request: Any) -> str:
        if not (_enabled() and _is_qwen35_mtp(config) and _is_t4_runtime(autotune)):
            return current(config, request)

        with autotune._AUTOTUNE_LOCK:
            binary = autotune._server_binary()
            if binary is None:
                return current(config, request)
            model_path = autotune._resolve_model_path(config)
            fingerprint = _fingerprint(autotune, config, binary, model_path)
            cached = _load_cache(fingerprint, autotune)
            if cached is None:
                try:
                    live = getattr(autotune, "_MANAGED_PROCESS", None)
                    if live is not None and live.poll() is None:
                        _stop_managed(autotune)
                    _reclaim_prior_mmm_server()
                    selected, baseline_tps, selected_tps = _measure_core(
                        autotune, binary, model_path, config
                    )
                    kv_policy = _manual_policy(config)
                    kv_measurements: dict[str, Any] = {}
                    source = "measured"
                except Exception as exc:
                    print(
                        "llama server: T4 max-speed calibration failed; "
                        "using conservative T4 tuner",
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    return current(config, request)
            else:
                selected = cached["selected_variant"]
                baseline_tps = float(cached.get("baseline_tps", 0.0) or 0.0)
                selected_tps = float(cached.get("selected_tps", 0.0) or 0.0)
                kv_policy = dict(cached.get("kv_policy", {}))
                kv_measurements = dict(cached.get("kv_measurements", {}))
                source = "cache"

            bucket = _bucket_for_request(config, request)
            if _kv_mode(config) == "auto" and str(bucket) not in kv_policy:
                try:
                    live = getattr(autotune, "_MANAGED_PROCESS", None)
                    if live is not None and live.poll() is None:
                        _stop_managed(autotune)
                    _reclaim_prior_mmm_server()
                    kv, measured = _measure_kv_bucket(
                        autotune, binary, model_path, config, selected, bucket
                    )
                    kv_policy[str(bucket)] = kv
                    kv_measurements[str(bucket)] = measured
                    source = "measured+kv"
                except Exception as exc:
                    print(
                        "llama server: T4 context-KV calibration failed; "
                        "using native KV for this bucket",
                        f"bucket={bucket}",
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    kv_policy[str(bucket)] = "native-default"

            _save_cache(
                fingerprint,
                selected,
                baseline_tps,
                selected_tps,
                kv_policy,
                kv_measurements,
            )
            kv = kv_policy.get(str(bucket), "native-default")
            active_key = (
                fingerprint,
                str(selected.spec_type),
                int(selected.draft_n_max),
                float(selected.draft_p_min),
                int(selected.ubatch),
                kv,
            )
            process = getattr(autotune, "_MANAGED_PROCESS", None)
            url = str(getattr(autotune, "_MANAGED_URL", "") or "")
            if (
                process is not None
                and process.poll() is None
                and url
                and getattr(autotune, "_mmm_t4_max_active_key", None) == active_key
            ):
                return url

            if process is not None and process.poll() is None:
                _stop_managed(autotune)
            _export(selected, kv)
            url = autotune._launch_selected(binary, model_path, config, selected)
            autotune._mmm_t4_max_active_key = active_key
            speedup = selected_tps / baseline_tps if baseline_tps > 0 else 1.0
            print(
                "llama server: T4 max-speed production",
                f"source={source}",
                f"spec={selected.spec_type}",
                f"n_max={selected.draft_n_max}",
                f"p_min={selected.draft_p_min:g}",
                f"ubatch={selected.ubatch}",
                f"kv={kv}",
                f"ctx_bucket={bucket}",
                f"baseline={baseline_tps:.2f}",
                f"selected={selected_tps:.2f}",
                f"speedup={speedup:.3f}x",
                flush=True,
            )
            return url

    setattr(ensure_max_speed, _MARKER, True)
    ensure_max_speed._mmm_qwen35_t4_max_speed = True  # type: ignore[attr-defined]
    autotune.ensure_tuned_server = ensure_max_speed


__all__ = [
    "_context_buckets",
    "_kv_candidates",
    "_kv_mode",
    "_p_min_candidates",
    "_ubatches",
    "_widths",
    "install",
]
