from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

from .qwen35_mtp_hotpath_contract import (
    _context_size,
    _is_qwen35_mtp,
    _reclaim_prior_mmm_server,
)

_SCHEMA_VERSION = "mmm/qwen35-t4-single-stream-tune-v1"
_MARKER = "_mmm_qwen35_t4_single_stream_tune_v1"
_DEFAULT_WIDTHS = (1, 2, 3, 4)
_DEFAULT_P_MIN = 0.8


def _enabled() -> bool:
    return os.environ.get("MMM_QWEN35_T4_TUNE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _hardware_identity(autotune: Any) -> str:
    getter = getattr(autotune, "_hardware_identity", None)
    if not callable(getter):
        return ""
    try:
        return str(getter())
    except Exception:
        return ""


def _is_t4_runtime(autotune: Any) -> bool:
    identity = _hardware_identity(autotune).casefold()
    return "tesla t4" in identity or "nvidia t4" in identity


def _widths() -> tuple[int, ...]:
    values: list[int] = []
    raw = os.environ.get("MMM_QWEN35_T4_WIDTHS", "1,2,3,4")
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if 1 <= value <= 8 and value not in values:
            values.append(value)
    return tuple(values or _DEFAULT_WIDTHS)


def _p_min() -> float:
    raw = os.environ.get("MMM_QWEN35_T4_P_MIN", str(_DEFAULT_P_MIN))
    try:
        value = float(raw)
    except ValueError:
        value = _DEFAULT_P_MIN
    return max(0.0, min(0.99, value))


def _probe_tokens(autotune: Any, config: Any) -> int:
    value = autotune._env_int("MMM_QWEN35_T4_PROBE_TOKENS", 256, minimum=64)
    return min(int(getattr(config, "max_new_tokens", value) or value), value)


def _cache_path() -> Path:
    explicit = os.environ.get("MMM_QWEN35_T4_TUNE_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path.home() / ".cache" / "mmm" / "qwen35-t4-single-stream.json").resolve()


def _fingerprint(
    autotune: Any,
    config: Any,
    binary: str,
    model_path: str,
    *,
    ubatch: int,
) -> str:
    base = autotune._fingerprint(config, binary, model_path)
    payload = {
        "schema": _SCHEMA_VERSION,
        "base": base,
        "hardware": _hardware_identity(autotune),
        "ctx": _context_size(),
        "ubatch": ubatch,
        "probe_tokens": _probe_tokens(autotune, config),
        "widths": list(_widths()),
        "p_min": _p_min(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _variant(
    autotune: Any,
    *,
    name: str,
    ubatch: int,
    width: int = 0,
    p_min: float = 0.0,
) -> Any:
    return autotune.ServerVariant(
        name=name,
        spec_type="draft-mtp" if width else "none",
        draft_n_max=width,
        ubatch=ubatch,
        parallel=1,
        cache_reuse=0,
        draft_p_min=p_min if width else 0.0,
    )


def _probe(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    request: Any,
    candidate: Any,
) -> Any:
    process = None
    started = time.perf_counter()
    try:
        port = autotune._free_port(autotune._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910))
        process = autotune._start_server(binary, model_path, config, candidate, port)
        url = autotune._wait_ready(process, port)
        bench = autotune._compact_benchmark_request(request)
        autotune._probe_server(url, bench, max_tokens=1, variant=candidate)
        return autotune._probe_server(
            url,
            bench,
            max_tokens=_probe_tokens(autotune, config),
            variant=candidate,
        )
    except Exception as exc:
        return autotune.ProbeResult(
            variant=candidate,
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


def _valid(probe: Any, baseline: Any) -> bool:
    return (
        bool(getattr(probe, "ok", False))
        and float(getattr(probe, "predicted_tps", 0.0) or 0.0) > 0
        and str(getattr(probe, "output_sha256", ""))
        == str(getattr(baseline, "output_sha256", ""))
    )


def _select(baseline: Any, probes: list[Any]) -> tuple[Any, float, float]:
    baseline_tps = float(getattr(baseline, "predicted_tps", 0.0) or 0.0)
    if not bool(getattr(baseline, "ok", False)) or baseline_tps <= 0:
        raise RuntimeError("Qwen3.5 T4 baseline decode probe failed")
    eligible = [baseline] + [
        probe for probe in probes if probe is not baseline and _valid(probe, baseline)
    ]
    best = max(
        eligible,
        key=lambda probe: float(getattr(probe, "predicted_tps", 0.0) or 0.0),
    )
    best_tps = float(getattr(best, "predicted_tps", 0.0) or 0.0)
    try:
        min_gain = max(
            1.0,
            float(os.environ.get("MMM_QWEN35_T4_MIN_GAIN", "1.01")),
        )
    except ValueError:
        min_gain = 1.01
    if best is not baseline and best_tps < baseline_tps * min_gain:
        return baseline.variant, baseline_tps, baseline_tps
    return best.variant, baseline_tps, best_tps


def _measure(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    request: Any,
    *,
    ubatch: int,
) -> tuple[Any, float, float, list[Any]]:
    baseline = _probe(
        autotune,
        binary,
        model_path,
        config,
        request,
        _variant(autotune, name="qwen35-t4-baseline", ubatch=ubatch),
    )
    probes = [baseline]
    mtp: list[Any] = []
    for width in _widths():
        probe = _probe(
            autotune,
            binary,
            model_path,
            config,
            request,
            _variant(
                autotune,
                name=f"qwen35-t4-mtp-{width}",
                ubatch=ubatch,
                width=width,
            ),
        )
        probes.append(probe)
        if _valid(probe, baseline):
            mtp.append(probe)

    # One confidence probe on the fastest zero-threshold width catches the common
    # low-acceptance case without multiplying every width by every threshold.
    if mtp and _p_min() > 0:
        fastest_mtp = max(
            mtp,
            key=lambda probe: float(getattr(probe, "predicted_tps", 0.0) or 0.0),
        )
        width = int(getattr(fastest_mtp.variant, "draft_n_max", 0) or 0)
        probes.append(
            _probe(
                autotune,
                binary,
                model_path,
                config,
                request,
                _variant(
                    autotune,
                    name=f"qwen35-t4-mtp-{width}|pm{_p_min():g}",
                    ubatch=ubatch,
                    width=width,
                    p_min=_p_min(),
                ),
            )
        )

    selected, baseline_tps, selected_tps = _select(baseline, probes)
    for probe in probes:
        tps = float(getattr(probe, "predicted_tps", 0.0) or 0.0)
        detail = f"{tps:.2f} tok/s" if getattr(probe, "ok", False) else str(
            getattr(probe, "error", "failed")
        )
        print(
            "llama server: T4 single-stream probe",
            f"{probe.variant.name} -> {detail}",
            flush=True,
        )
    return selected, baseline_tps, selected_tps, probes


def _load_cached(autotune: Any, fingerprint: str) -> tuple[Any, float, float] | None:
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
        selected = autotune.ServerVariant(**raw)
        return (
            selected,
            float(payload.get("baseline_tps", 0.0) or 0.0),
            float(payload.get("selected_tps", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        return None


def _save_cached(
    fingerprint: str,
    selected: Any,
    baseline_tps: float,
    selected_tps: float,
) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": _SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "selected": asdict(selected),
                "baseline_tps": baseline_tps,
                "selected_tps": selected_tps,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _export(selected: Any, ubatch: int) -> None:
    os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = str(selected.spec_type)
    os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = str(selected.draft_n_max)
    os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "1"
    os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(ubatch)
    os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = "0"
    os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] = f"{selected.draft_p_min:g}"
    os.environ["MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE"] = "single_stream"
    os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] = "native-default"


def install(autotune: Any) -> None:
    """Replace fixed MTP-3 with measured single-stream selection on Tesla T4 only."""
    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_t4_fastest(config: Any, request: Any) -> str:
        if not (_enabled() and _is_qwen35_mtp(config) and _is_t4_runtime(autotune)):
            return current(config, request)

        process = getattr(autotune, "_MANAGED_PROCESS", None)
        url = str(getattr(autotune, "_MANAGED_URL", "") or "")
        if process is not None and process.poll() is None and url:
            return url

        _reclaim_prior_mmm_server()
        if os.environ.get("LLAMA_SERVER_URL", "").strip():
            return current(config, request)

        with autotune._AUTOTUNE_LOCK:
            process = getattr(autotune, "_MANAGED_PROCESS", None)
            url = str(getattr(autotune, "_MANAGED_URL", "") or "")
            if process is not None and process.poll() is None and url:
                return url

            binary = autotune._server_binary()
            if binary is None:
                raise RuntimeError("native llama-server binary is unavailable")
            model_path = autotune._resolve_model_path(config)
            batch = autotune._env_int("MMM_LLAMA_BATCH", 2048)
            ubatch = min(batch, autotune._env_int("MMM_LLAMA_UBATCH", 512))
            fingerprint = _fingerprint(
                autotune, config, binary, model_path, ubatch=ubatch
            )
            cached = _load_cached(autotune, fingerprint)
            if cached is None:
                try:
                    selected, baseline_tps, selected_tps, _ = _measure(
                        autotune,
                        binary,
                        model_path,
                        config,
                        request,
                        ubatch=ubatch,
                    )
                    _save_cached(
                        fingerprint,
                        selected,
                        baseline_tps,
                        selected_tps,
                    )
                    source = "measured"
                except Exception as exc:
                    print(
                        "llama server: T4 speed calibration failed; using fixed hotpath",
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    return current(config, request)
            else:
                selected, baseline_tps, selected_tps = cached
                source = "cache"

            url = autotune._launch_selected(binary, model_path, config, selected)
            _export(selected, ubatch)
            speedup = selected_tps / baseline_tps if baseline_tps > 0 else 1.0
            print(
                "llama server: T4 single-stream production profile",
                f"source={source}",
                f"spec={selected.spec_type}",
                f"n_max={selected.draft_n_max}",
                f"p_min={selected.draft_p_min:g}",
                f"baseline={baseline_tps:.2f}",
                f"selected={selected_tps:.2f}",
                f"speedup={speedup:.3f}x",
                flush=True,
            )
            return url

    setattr(ensure_t4_fastest, _MARKER, True)
    ensure_t4_fastest._mmm_qwen35_t4_single_stream_tune = True  # type: ignore[attr-defined]
    autotune.ensure_tuned_server = ensure_t4_fastest


__all__ = ["_is_t4_runtime", "_select", "_widths", "install"]
