from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
from dataclasses import replace
from functools import wraps
from typing import Any


_TUNING_LOCK = threading.RLock()


def _probe_request_cache_reuse(
    autotune_module: Any,
    base_url: str,
    *,
    cache_reuse: int,
    variant: Any,
    max_tokens: int,
) -> Any:
    """Measure one request-scoped cache-reuse value without reloading the model."""
    import httpx

    prefix = (
        f"MMM cache reuse candidate {cache_reuse}. "
        "Minecraft Fabric deterministic repair context. Preserve package names, "
        "imports, registry identifiers, JSON keys, and API contracts. "
    ) * 24
    messages_a = (
        {"role": "system", "content": prefix},
        {"role": "user", "content": "Return exactly: CACHE-A"},
    )
    messages_b = (
        {"role": "system", "content": prefix},
        {"role": "user", "content": "Return exactly: CACHE-B"},
    )

    def request(messages: tuple[dict[str, str], ...]) -> tuple[str, int, float, float]:
        payload = {
            "model": "local",
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 1234,
            "cache_prompt": True,
            "n_cache_reuse": int(cache_reuse),
            "stream": False,
        }
        started = time.perf_counter()
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=autotune_module._env_int(
                "MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT",
                300,
            ),
        )
        response.raise_for_status()
        data = response.json()
        elapsed = time.perf_counter() - started
        output = autotune_module._assistant_output(data)
        timings = data.get("timings") or {}
        usage = data.get("usage") or {}
        predicted = int(
            timings.get("predicted_n")
            or usage.get("completion_tokens")
            or 0
        )
        prompt_tps = float(timings.get("prompt_per_second") or 0.0)
        return output, predicted, prompt_tps, elapsed

    started = time.perf_counter()
    try:
        request(messages_a)
        output, predicted, prompt_tps, reuse_elapsed = request(messages_b)
        return autotune_module.ProbeResult(
            variant=variant,
            ok=bool(output) and reuse_elapsed > 0,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            predicted_tokens=predicted,
            predicted_tps=(predicted / reuse_elapsed if predicted > 0 else 0.0),
            prompt_tps=prompt_tps,
            elapsed_seconds=reuse_elapsed,
        )
    except Exception as exc:
        return autotune_module.ProbeResult(
            variant=variant,
            ok=False,
            output_sha256="",
            predicted_tokens=0,
            predicted_tps=0.0,
            prompt_tps=0.0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _choose_cache_probe(probes: list[Any], *, minimum_gain: float) -> Any | None:
    if not probes:
        return None
    baseline = probes[0]
    if not getattr(baseline, "ok", False) or baseline.elapsed_seconds <= 0:
        return None
    valid = [
        probe
        for probe in probes
        if getattr(probe, "ok", False)
        and getattr(probe, "output_sha256", "")
        == getattr(baseline, "output_sha256", "")
        and float(getattr(probe, "elapsed_seconds", 0.0)) > 0
    ]
    if not valid:
        return baseline
    best = min(valid, key=lambda probe: float(probe.elapsed_seconds))
    if best is baseline:
        return baseline
    gain = float(baseline.elapsed_seconds) / max(1e-9, float(best.elapsed_seconds))
    return best if gain >= max(1.0, minimum_gain) else baseline


def install(
    autotune_module: Any,
    hardware_policy_module: Any,
    runtime_tuning_module: Any,
) -> None:
    """Tune request-scoped KV cache reuse on one already-loaded llama server."""
    if getattr(autotune_module, "_mmm_request_cache_reuse_tuning", False):
        return

    current_start = autotune_module._start_server
    if not getattr(current_start, "_mmm_request_scoped_cache_reuse", False):

        @wraps(current_start)
        def start_without_global_cache_reuse(
            binary: str,
            model_path: str,
            config: Any,
            variant: Any,
            port: int,
        ) -> subprocess.Popen[bytes]:
            return current_start(
                binary,
                model_path,
                config,
                replace(variant, cache_reuse=0),
                port,
            )

        start_without_global_cache_reuse._mmm_request_scoped_cache_reuse = True  # type: ignore[attr-defined]
        autotune_module._start_server = start_without_global_cache_reuse

    current_payload = hardware_policy_module._server_payload
    if not getattr(current_payload, "_mmm_active_cache_reuse", False):

        @wraps(current_payload)
        def payload_with_active_cache_reuse(adapter: Any, request: Any) -> dict[str, Any]:
            payload = current_payload(adapter, request)
            raw = os.environ.get("MMM_LLAMA_ACTIVE_CACHE_REUSE", "").strip()
            if raw:
                try:
                    value = int(raw)
                except ValueError:
                    value = 0
                payload["n_cache_reuse"] = max(0, min(8192, value))
            return payload

        payload_with_active_cache_reuse._mmm_active_cache_reuse = True  # type: ignore[attr-defined]
        hardware_policy_module._server_payload = payload_with_active_cache_reuse

    current_benchmark = autotune_module._benchmark

    @wraps(current_benchmark)
    def benchmark_with_single_server_cache_stage(
        binary: str,
        model_path: str,
        config: Any,
        request: Any,
        fingerprint: str,
    ) -> Any | None:
        candidates = tuple(runtime_tuning_module._cache_reuse_candidates())

        # The runtime benchmark owns speculative, ubatch and parallel selection.
        # Temporarily disable only its cache-reuse stage, then benchmark every
        # request-scoped reuse value against one already-loaded selected server.
        with _TUNING_LOCK:
            original_candidates = runtime_tuning_module._cache_reuse_candidates
            runtime_tuning_module._cache_reuse_candidates = lambda: ()
            try:
                decision = current_benchmark(
                    binary,
                    model_path,
                    config,
                    request,
                    fingerprint,
                )
            finally:
                runtime_tuning_module._cache_reuse_candidates = original_candidates

        if decision is None or not candidates:
            return decision

        selected = decision.selected
        port = autotune_module._free_port(
            autotune_module._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910)
        )
        process = None
        probes: list[Any] = []
        probe_tokens = min(
            8,
            int(config.max_new_tokens),
            autotune_module._env_int(
                "MMM_LLAMA_AUTOTUNE_TOKENS",
                autotune_module._BENCHMARK_OUTPUT_TOKENS,
            ),
        )
        try:
            process = autotune_module._start_server(
                binary,
                model_path,
                config,
                replace(selected, cache_reuse=0),
                port,
            )
            url = autotune_module._wait_ready(process, port)
            autotune_module._probe_server(
                url,
                autotune_module._compact_benchmark_request(request),
                max_tokens=1,
                variant=selected,
            )
            for value in candidates:
                variant = replace(
                    selected,
                    name=f"{selected.name.split('|cr', 1)[0]}|cr{value}",
                    cache_reuse=int(value),
                )
                probes.append(
                    _probe_request_cache_reuse(
                        autotune_module,
                        url,
                        cache_reuse=int(value),
                        variant=variant,
                        max_tokens=max(1, probe_tokens),
                    )
                )
        except Exception as exc:
            print(
                "native llama-server: request-scoped cache-reuse tuning unavailable; "
                f"keeping cache_reuse=0 ({type(exc).__name__})",
                flush=True,
            )
        finally:
            autotune_module._stop_server(process)

        if not probes:
            return decision
        chosen = _choose_cache_probe(
            probes,
            minimum_gain=autotune_module._env_float(
                "MMM_LLAMA_STAGE_MIN_GAIN",
                1.01,
            ),
        )
        if chosen is None:
            return replace(decision, probes=tuple(decision.probes) + tuple(probes))
        return replace(
            decision,
            selected=chosen.variant,
            probes=tuple(decision.probes) + tuple(probes),
        )

    benchmark_with_single_server_cache_stage._mmm_single_server_cache_stage = True  # type: ignore[attr-defined]
    autotune_module._benchmark = benchmark_with_single_server_cache_stage
    autotune_module._mmm_request_cache_reuse_tuning = True


__all__ = ["install"]
