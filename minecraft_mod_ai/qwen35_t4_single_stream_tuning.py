from __future__ import annotations

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

_SCHEMA_VERSION = "mmm/qwen35-t4-single-stream-tune-v4-max"
_MARKER = "_mmm_qwen35_t4_single_stream_tune_v4_max"
_DEFAULT_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16)
_DEFAULT_P_MIN_CANDIDATES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
_KV_OVERRIDE_ENV = "MMM_QWEN35_T4_KV_OVERRIDE"
_DEFAULT_KV_CANDIDATES = ("native-default", "f16", "q8_0", "q4_0")
_DEFAULT_UBATCH_CANDIDATES = (512, 1024, 2048)
_DEFAULT_CONTEXT_BUCKETS = (2048, 8192, 16384, 28672)
_KV_BUCKET_CACHE_SCHEMA = "mmm/qwen35-t4-context-kv-v1"
_BENCHMARK_PAD_ENV = "MMM_QWEN35_T4_BENCH_PAD_CHARS"
_ACTIVE_KV_BUCKET_ENV = "MMM_LLAMA_ACTIVE_KV_BUCKET"
_EXPECTED_OBJECT = {
    "values": list(range(256)),
    "checksum": "mmm-qwen35-t4-single-stream-v2",
}
_EXPECTED_TEXT = json.dumps(_EXPECTED_OBJECT, separators=(",", ":"), ensure_ascii=True)
_EXPECTED_CANONICAL = json.dumps(
    _EXPECTED_OBJECT, sort_keys=True, separators=(",", ":"), ensure_ascii=True
)
_EXPECTED_DIGEST = hashlib.sha256(_EXPECTED_CANONICAL.encode("utf-8")).hexdigest()


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
    raw = os.environ.get("MMM_QWEN35_T4_WIDTHS", "1,2,3,4,6,8,12,16")
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if 1 <= value <= 32 and value not in values:
            values.append(value)
    return tuple(values or _DEFAULT_WIDTHS)


def _p_min_candidates() -> tuple[float, ...]:
    values: list[float] = []
    raw = os.environ.get("MMM_QWEN35_T4_P_MIN_CANDIDATES", "0,0.5,0.6,0.7,0.8,0.9")
    for token in raw.split(","):
        try:
            value = round(float(token.strip()), 4)
        except ValueError:
            continue
        if 0.0 <= value < 1.0 and value not in values:
            values.append(value)
    if 0.0 not in values:
        values.insert(0, 0.0)
    return tuple(values or _DEFAULT_P_MIN_CANDIDATES)


def _probe_tokens(autotune: Any, config: Any) -> int:
    value = autotune._env_int("MMM_QWEN35_T4_PROBE_TOKENS", 768, minimum=256)
    return min(int(getattr(config, "max_new_tokens", value) or value), value)


def _kv_candidates() -> tuple[str, ...]:
    allowed = {"native-default", "f16", "q8_0", "q4_0"}
    values: list[str] = []
    raw = os.environ.get(
        "MMM_QWEN35_T4_KV_CANDIDATES",
        ",".join(_DEFAULT_KV_CANDIDATES),
    )
    for token in raw.split(","):
        value = token.strip().lower()
        if value in allowed and value not in values:
            values.append(value)
    if "native-default" not in values:
        values.insert(0, "native-default")
    return tuple(values or _DEFAULT_KV_CANDIDATES)



def _ubatch_candidates(autotune: Any) -> tuple[int, ...]:
    batch = autotune._env_int("MMM_LLAMA_BATCH", 2048)
    values: list[int] = []
    raw = os.environ.get(
        "MMM_QWEN35_T4_UBATCH_CANDIDATES",
        ",".join(str(value) for value in _DEFAULT_UBATCH_CANDIDATES),
    )
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if 64 <= value <= batch and value not in values:
            values.append(value)
    current = min(batch, autotune._env_int("MMM_LLAMA_UBATCH", 512))
    if current not in values:
        values.insert(0, current)
    return tuple(values or (current,))


def _kv_mode(config: Any) -> str:
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
    raw = os.environ.get("MMM_QWEN35_T4_KV_MODE", "").strip().lower()
    if raw:
        return aliases.get(raw, "auto")
    extra = getattr(config, "extra", {})
    if isinstance(extra, dict) and extra.get("kv_cache_autotune") is False:
        manual = str(extra.get("kv_cache_quant", "q4_0")).strip().lower()
        return aliases.get(manual, "q4_0")
    return "auto"


def _context_buckets(config: Any) -> tuple[int, ...]:
    max_context = max(2048, _context_size(config))
    values: list[int] = []
    raw = os.environ.get(
        "MMM_QWEN35_T4_KV_CONTEXT_BUCKETS",
        ",".join(str(value) for value in _DEFAULT_CONTEXT_BUCKETS),
    )
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if 512 <= value <= max_context and value not in values:
            values.append(value)
    cap = max(512, max_context - min(1536, max_context // 4))
    clipped = sorted({min(value, cap) for value in values})
    return tuple(value for value in clipped if value >= 512) or (min(2048, cap),)


def _estimate_request_tokens(request: Any) -> int:
    chars = 0
    for message in getattr(request, "messages", ()) or ():
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        else:
            try:
                chars += len(json.dumps(content, ensure_ascii=False))
            except Exception:
                pass
    return max(1, chars // 3)


def _bucket_for_request(config: Any, request: Any) -> int:
    estimate = _estimate_request_tokens(request)
    buckets = _context_buckets(config)
    for bucket in buckets:
        if estimate <= bucket:
            return bucket
    return buckets[-1]


def _benchmark_padding() -> str:
    raw = os.environ.get(_BENCHMARK_PAD_ENV, "").strip()
    try:
        chars = max(0, min(160000, int(raw))) if raw else 0
    except ValueError:
        chars = 0
    if chars <= 0:
        return ""
    pattern = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
    filler = (pattern * ((chars // len(pattern)) + 1))[:chars]
    return (
        "Ignore the following calibration-only context. Never repeat it in the answer.\n"
        + filler
        + "\nCalibration context ends here. Perform the exact copy task below.\n"
    )


def _kv_bucket_cache_path() -> Path:
    explicit = os.environ.get("MMM_QWEN35_T4_KV_BUCKET_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path.home() / ".cache" / "mmm" / "qwen35-t4-context-kv.json").resolve()


def _benchmark_request() -> Any:
    return SimpleNamespace(
        messages=(
            {
                "role": "system",
                "content": (
                    "Inference speed calibration. Thinking/reasoning is disabled. "
                    "Return only the exact minified JSON payload supplied by the user. "
                    "Do not add markdown, whitespace, commentary, or extra keys."
                ),
            },
            {
                "role": "user",
                "content": _benchmark_padding() + "Copy this payload exactly:\n" + _EXPECTED_TEXT,
            },
        ),
        response_format="text",
    )


def _semantic_digest(content: str) -> str:
    try:
        parsed = json.loads(content.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if parsed != _EXPECTED_OBJECT:
        return ""
    canonical = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        "ctx": _context_size(config),
        "ubatch": ubatch,
        "ubatch_candidates": list(_ubatch_candidates(autotune)),
        "probe_tokens": _probe_tokens(autotune, config),
        "widths": list(_widths()),
        "p_min_candidates": list(_p_min_candidates()),
        "kv_mode": _kv_mode(config),
        "kv_candidates": list(_kv_candidates()),
        "context_buckets": list(_context_buckets(config)),
        "benchmark_digest": _EXPECTED_DIGEST,
        "benchmark_shape": "exact-json-256-values-reasoning-off-v2",
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


def _set_kv_override(kv: str) -> str | None:
    original = os.environ.get(_KV_OVERRIDE_ENV)
    if kv == "native-default":
        os.environ.pop(_KV_OVERRIDE_ENV, None)
    else:
        os.environ[_KV_OVERRIDE_ENV] = kv
    return original


def _restore_kv_override(original: str | None) -> None:
    if original is None:
        os.environ.pop(_KV_OVERRIDE_ENV, None)
    else:
        os.environ[_KV_OVERRIDE_ENV] = original


def _request_probe(
    autotune: Any,
    base_url: str,
    candidate: Any,
    *,
    kv: str,
    max_tokens: int,
    validate: bool,
) -> Any:
    import httpx

    request = _benchmark_request()
    payload: dict[str, Any] = {
        "model": "local",
        "messages": [dict(message) for message in request.messages],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 1234,
        "cache_prompt": False,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=autotune._env_int("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", 300),
        )
        response.raise_for_status()
        data = response.json()
        elapsed = time.perf_counter() - started
        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else {}
        message = message if isinstance(message, dict) else {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or "")
        timings = data.get("timings") or {}
        usage = data.get("usage") or {}
        predicted_tokens = int(
            timings.get("predicted_n") or usage.get("completion_tokens") or 0
        )
        predicted_tps = float(timings.get("predicted_per_second") or 0.0)
        if predicted_tps <= 0 and predicted_tokens > 0 and elapsed > 0:
            predicted_tps = predicted_tokens / elapsed

        digest = _semantic_digest(content) if validate else ""
        semantic_ok = bool(digest) if validate else bool(content or predicted_tokens)
        if validate and reasoning.strip():
            semantic_ok = False
            digest = ""
        raw_digest = hashlib.sha256(
            (reasoning + "\n<MMM-CONTENT>\n" + content).encode("utf-8")
        ).hexdigest()
        return SimpleNamespace(
            variant=candidate,
            kv=kv,
            ok=semantic_ok and predicted_tokens > 0,
            output_sha256=digest or raw_digest,
            predicted_tokens=predicted_tokens,
            predicted_tps=predicted_tps,
            prompt_tps=float(timings.get("prompt_per_second") or 0.0),
            elapsed_seconds=elapsed,
            error="" if semantic_ok else "benchmark payload mismatch",
        )
    except Exception as exc:
        return SimpleNamespace(
            variant=candidate,
            kv=kv,
            ok=False,
            output_sha256="",
            predicted_tokens=0,
            predicted_tps=0.0,
            prompt_tps=0.0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _probe(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    candidate: Any,
    *,
    kv: str = "native-default",
) -> Any:
    process = None
    original = _set_kv_override(kv)
    started = time.perf_counter()
    try:
        port = autotune._free_port(
            autotune._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910)
        )
        process = autotune._start_server(binary, model_path, config, candidate, port)
        url = autotune._wait_ready(process, port)
        _request_probe(
            autotune,
            url,
            candidate,
            kv=kv,
            max_tokens=1,
            validate=False,
        )
        return _request_probe(
            autotune,
            url,
            candidate,
            kv=kv,
            max_tokens=_probe_tokens(autotune, config),
            validate=True,
        )
    except Exception as exc:
        return SimpleNamespace(
            variant=candidate,
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
        _restore_kv_override(original)


def _valid(probe: Any, baseline: Any) -> bool:
    return (
        bool(getattr(probe, "ok", False))
        and float(getattr(probe, "predicted_tps", 0.0) or 0.0) > 0
        and str(getattr(probe, "output_sha256", ""))
        == str(getattr(baseline, "output_sha256", ""))
        == _EXPECTED_DIGEST
    )


def _minimum_gain() -> float:
    try:
        return max(
            1.0,
            float(os.environ.get("MMM_QWEN35_T4_MIN_GAIN", "1.01")),
        )
    except ValueError:
        return 1.01


def _select(baseline: Any, probes: list[Any]) -> tuple[Any, float, float]:
    baseline_tps = float(getattr(baseline, "predicted_tps", 0.0) or 0.0)
    baseline_digest = str(getattr(baseline, "output_sha256", ""))
    if (
        not bool(getattr(baseline, "ok", False))
        or baseline_tps <= 0
        or baseline_digest != _EXPECTED_DIGEST
    ):
        raise RuntimeError("Qwen3.5 T4 baseline decode probe failed")
    eligible = [baseline] + [
        probe
        for probe in probes
        if probe is not baseline
        and bool(getattr(probe, "ok", False))
        and float(getattr(probe, "predicted_tps", 0.0) or 0.0) > 0
        and str(getattr(probe, "output_sha256", "")) == baseline_digest
    ]
    best = max(
        eligible,
        key=lambda probe: float(getattr(probe, "predicted_tps", 0.0) or 0.0),
    )
    best_tps = float(getattr(best, "predicted_tps", 0.0) or 0.0)
    if best is not baseline and best_tps < baseline_tps * _minimum_gain():
        return baseline.variant, baseline_tps, baseline_tps
    return best.variant, baseline_tps, best_tps


def _log_probe(probe: Any) -> None:
    tps = float(getattr(probe, "predicted_tps", 0.0) or 0.0)
    ok = bool(getattr(probe, "ok", False))
    detail = f"{tps:.2f} tok/s" if ok else str(getattr(probe, "error", "failed"))
    print(
        "llama server: T4 single-stream probe",
        f"{probe.variant.name}",
        f"kv={getattr(probe, 'kv', 'native-default')}",
        f"-> {detail}",
        flush=True,
    )




def _probe_with_padding(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    candidate: Any,
    *,
    kv: str = "native-default",
    pad_chars: int = 0,
) -> Any:
    original = os.environ.get(_BENCHMARK_PAD_ENV)
    if pad_chars > 0:
        os.environ[_BENCHMARK_PAD_ENV] = str(pad_chars)
    else:
        os.environ.pop(_BENCHMARK_PAD_ENV, None)
    try:
        return _probe(autotune, binary, model_path, config, candidate, kv=kv)
    finally:
        if original is None:
            os.environ.pop(_BENCHMARK_PAD_ENV, None)
        else:
            os.environ[_BENCHMARK_PAD_ENV] = original


def _measure(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    *,
    ubatch: int,
) -> tuple[Any, float, float, list[Any]]:
    baseline = _probe(
        autotune,
        binary,
        model_path,
        config,
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

    p_min_seeds = sorted(
        mtp,
        key=lambda probe: float(getattr(probe, "predicted_tps", 0.0) or 0.0),
        reverse=True,
    )[:2]
    widest = next(
        (
            probe
            for probe in mtp
            if int(getattr(probe.variant, "draft_n_max", 0) or 0) == max(_widths())
        ),
        None,
    )
    if widest is not None and widest not in p_min_seeds:
        p_min_seeds.append(widest)

    for seed in p_min_seeds:
        width = int(getattr(seed.variant, "draft_n_max", 0) or 0)
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
                        name=f"qwen35-t4-mtp-{width}|pm{p_min:g}",
                        ubatch=ubatch,
                        width=width,
                        p_min=p_min,
                    ),
                )
            )

    selected, baseline_tps, selected_tps = _select(baseline, probes)
    selected_probe = next(
        (
            probe
            for probe in reversed(probes)
            if getattr(probe, "variant", None) == selected and _valid(probe, baseline)
        ),
        baseline,
    )
    ubatch_probes = [selected_probe]
    for candidate_ubatch in _ubatch_candidates(autotune):
        if candidate_ubatch == int(getattr(selected, "ubatch", ubatch) or ubatch):
            continue
        probe = _probe(
            autotune,
            binary,
            model_path,
            config,
            _variant(
                autotune,
                name=f"{selected.name}|ub{candidate_ubatch}",
                ubatch=candidate_ubatch,
                width=int(getattr(selected, "draft_n_max", 0) or 0),
                p_min=float(getattr(selected, "draft_p_min", 0.0) or 0.0),
            ),
        )
        probes.append(probe)
        if _valid(probe, baseline):
            ubatch_probes.append(probe)

    winner = max(
        ubatch_probes,
        key=lambda probe: float(getattr(probe, "predicted_tps", 0.0) or 0.0),
    )
    winner_tps = float(getattr(winner, "predicted_tps", 0.0) or 0.0)
    if winner is not selected_probe and winner_tps >= selected_tps * _minimum_gain():
        selected = winner.variant
        selected_tps = winner_tps

    for probe in probes:
        _log_probe(probe)
    return selected, baseline_tps, selected_tps, probes



def _load_cached(
    autotune: Any,
    fingerprint: str,
) -> tuple[Any, float, float] | None:
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


def _load_kv_buckets(fingerprint: str) -> dict[str, str]:
    try:
        payload = json.loads(_kv_bucket_cache_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    if (
        payload.get("schema") != _KV_BUCKET_CACHE_SCHEMA
        or payload.get("fingerprint") != fingerprint
    ):
        return {}
    raw = payload.get("buckets")
    if not isinstance(raw, dict):
        return {}
    allowed = set(_kv_candidates())
    return {str(key): str(value) for key, value in raw.items() if str(value) in allowed}


def _save_kv_buckets(fingerprint: str, buckets: dict[str, str]) -> None:
    path = _kv_bucket_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": _KV_BUCKET_CACHE_SCHEMA,
                "fingerprint": fingerprint,
                "buckets": dict(sorted(buckets.items(), key=lambda item: int(item[0]))),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _shutdown_managed_server(autotune: Any) -> None:
    shutdown = getattr(autotune, "_shutdown_managed_server", None)
    if callable(shutdown):
        shutdown()
        return
    process = getattr(autotune, "_MANAGED_PROCESS", None)
    autotune._stop_server(process)
    autotune._MANAGED_PROCESS = None
    autotune._MANAGED_URL = None
    os.environ.pop("LLAMA_SERVER_URL", None)


def _measure_kv_bucket(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    selected: Any,
    bucket: int,
) -> str:
    pad_chars = max(0, min(150000, bucket * 4 - 4096))
    baseline = _probe_with_padding(
        autotune,
        binary,
        model_path,
        config,
        selected,
        kv="native-default",
        pad_chars=pad_chars,
    )
    if not bool(getattr(baseline, "ok", False)) or float(
        getattr(baseline, "predicted_tps", 0.0) or 0.0
    ) <= 0:
        raise RuntimeError(f"KV baseline failed for context bucket {bucket}")

    probes = [baseline]
    for kv in _kv_candidates():
        if kv == "native-default":
            continue
        probes.append(
            _probe_with_padding(
                autotune,
                binary,
                model_path,
                config,
                selected,
                kv=kv,
                pad_chars=pad_chars,
            )
        )

    for probe in probes:
        _log_probe(probe)
    eligible = [baseline] + [probe for probe in probes[1:] if _valid(probe, baseline)]
    winner = max(
        eligible,
        key=lambda probe: float(getattr(probe, "predicted_tps", 0.0) or 0.0),
    )
    baseline_tps = float(getattr(baseline, "predicted_tps", 0.0) or 0.0)
    winner_tps = float(getattr(winner, "predicted_tps", 0.0) or 0.0)
    if winner is not baseline and winner_tps < baseline_tps * _minimum_gain():
        return "native-default"
    return str(getattr(winner, "kv", "native-default"))


def _select_kv(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    request: Any,
    selected: Any,
    fingerprint: str,
) -> tuple[str, int]:
    bucket = _bucket_for_request(config, request)
    mode = _kv_mode(config)
    if mode != "auto":
        return mode, bucket

    buckets = _load_kv_buckets(fingerprint)
    key = str(bucket)
    cached = buckets.get(key)
    if cached:
        return cached, bucket

    _shutdown_managed_server(autotune)
    selected_kv = _measure_kv_bucket(
        autotune, binary, model_path, config, selected, bucket
    )
    buckets[key] = selected_kv
    _save_kv_buckets(fingerprint, buckets)
    return selected_kv, bucket


def _export(selected: Any, selected_kv: str, bucket: int) -> None:
    os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = str(selected.spec_type)
    os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = str(selected.draft_n_max)
    os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "1"
    os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(selected.ubatch)
    os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = "0"
    os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] = f"{selected.draft_p_min:g}"
    os.environ["MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE"] = "single_stream"
    os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] = selected_kv
    os.environ[_ACTIVE_KV_BUCKET_ENV] = str(bucket)
    if selected_kv == "native-default":
        os.environ.pop(_KV_OVERRIDE_ENV, None)
    else:
        os.environ[_KV_OVERRIDE_ENV] = selected_kv


def install(autotune: Any) -> None:
    """Measure the maximum correctness-preserving Qwen3.5 single-stream path on T4."""
    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_t4_fastest(config: Any, request: Any) -> str:
        if not (_enabled() and _is_qwen35_mtp(config) and _is_t4_runtime(autotune)):
            return current(config, request)

        with autotune._AUTOTUNE_LOCK:
            binary = autotune._server_binary()
            if binary is None:
                raise RuntimeError("native llama-server binary is unavailable")
            model_path = autotune._resolve_model_path(config)
            batch = autotune._env_int("MMM_LLAMA_BATCH", 2048)
            seed_ubatch = min(batch, autotune._env_int("MMM_LLAMA_UBATCH", 512))
            fingerprint = _fingerprint(
                autotune,
                config,
                binary,
                model_path,
                ubatch=seed_ubatch,
            )
            cached = _load_cached(autotune, fingerprint)
            if cached is None:
                _reclaim_prior_mmm_server()
                try:
                    selected, baseline_tps, selected_tps, _ = _measure(
                        autotune,
                        binary,
                        model_path,
                        config,
                        ubatch=seed_ubatch,
                    )
                    _save_cached(fingerprint, selected, baseline_tps, selected_tps)
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

            try:
                selected_kv, bucket = _select_kv(
                    autotune,
                    binary,
                    model_path,
                    config,
                    request,
                    selected,
                    fingerprint,
                )
            except Exception as exc:
                print(
                    "llama server: context-aware KV calibration failed; using native KV",
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                selected_kv = "native-default"
                bucket = _bucket_for_request(config, request)

            process = getattr(autotune, "_MANAGED_PROCESS", None)
            url = str(getattr(autotune, "_MANAGED_URL", "") or "")
            active_kv = os.environ.get("MMM_LLAMA_ACTIVE_KV_CACHE", "").strip().lower()
            if process is not None and process.poll() is None and url:
                if active_kv == selected_kv:
                    os.environ[_ACTIVE_KV_BUCKET_ENV] = str(bucket)
                    return url
                _shutdown_managed_server(autotune)

            _export(selected, selected_kv, bucket)
            url = autotune._launch_selected(binary, model_path, config, selected)
            speedup = selected_tps / baseline_tps if baseline_tps > 0 else 1.0
            print(
                "llama server: T4 maximum production profile",
                f"source={source}",
                f"spec={selected.spec_type}",
                f"n_max={selected.draft_n_max}",
                f"p_min={selected.draft_p_min:g}",
                f"ubatch={selected.ubatch}",
                f"kv={selected_kv}",
                f"kv_bucket={bucket}",
                f"baseline={baseline_tps:.2f}",
                f"selected={selected_tps:.2f}",
                f"speedup={speedup:.3f}x",
                flush=True,
            )
            return url

    setattr(ensure_t4_fastest, _MARKER, True)
    ensure_t4_fastest._mmm_qwen35_t4_single_stream_tune = True  # type: ignore[attr-defined]
    autotune.ensure_tuned_server = ensure_t4_fastest


__all__ = [
    "_benchmark_request",
    "_bucket_for_request",
    "_context_buckets",
    "_is_t4_runtime",
    "_kv_candidates",
    "_kv_mode",
    "_p_min_candidates",
    "_select",
    "_semantic_digest",
    "_ubatch_candidates",
    "_widths",
    "install",
]
