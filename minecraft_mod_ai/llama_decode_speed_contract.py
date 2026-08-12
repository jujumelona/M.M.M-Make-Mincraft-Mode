from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, replace
from functools import wraps
from types import SimpleNamespace
from typing import Any

_SCHEMA_VERSION = "mmm/llama-decode-speed-v1"
_INSTALL_LOCK = threading.RLock()
_BENCHMARK_LOCK = threading.RLock()


@dataclass(frozen=True)
class SpeedServerVariant:
    name: str
    spec_type: str = "none"
    draft_n_max: int = 0
    ubatch: int = 0
    parallel: int = 1
    cache_reuse: int = 0
    draft_p_min: float = 0.0


def _as_speed_variant(value: Any, *, draft_p_min: float | None = None) -> SpeedServerVariant:
    p_min = (
        float(getattr(value, "draft_p_min", 0.0) or 0.0)
        if draft_p_min is None
        else float(draft_p_min)
    )
    return SpeedServerVariant(
        name=str(getattr(value, "name", "baseline")),
        spec_type=str(getattr(value, "spec_type", "none")),
        draft_n_max=max(0, int(getattr(value, "draft_n_max", 0) or 0)),
        ubatch=max(0, int(getattr(value, "ubatch", 0) or 0)),
        parallel=max(1, int(getattr(value, "parallel", 1) or 1)),
        cache_reuse=max(0, int(getattr(value, "cache_reuse", 0) or 0)),
        draft_p_min=max(0.0, min(0.999, p_min)),
    )


def _mtp_p_min_candidates() -> tuple[float, ...]:
    values: list[float] = [0.0]
    for token in os.environ.get(
        "MMM_LLAMA_MTP_P_MIN_CANDIDATES", "0,0.6,0.8,0.9"
    ).split(","):
        try:
            value = round(float(token.strip()), 4)
        except ValueError:
            continue
        if 0.0 <= value < 1.0 and value not in values:
            values.append(value)
    return tuple(values)


def _tuning_objective() -> str:
    raw = os.environ.get("MMM_LLAMA_TUNING_OBJECTIVE", "single_stream").strip().lower()
    return "throughput" if raw in {"throughput", "aggregate", "concurrent"} else "single_stream"


def _explicit_parallel_requested() -> bool:
    for name in ("MMM_LLAMA_PARALLEL", "MMM_LLAMA_CONCURRENT_REQUESTS"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            if int(raw) > 1:
                return True
        except ValueError:
            pass
    return False


def _decode_ratio(probe: Any, baseline: Any) -> float:
    base = max(1e-9, float(getattr(baseline, "predicted_tps", 0.0)))
    return max(0.0, float(getattr(probe, "predicted_tps", 0.0))) / base


def _representative_benchmark_request(request: Any) -> Any:
    del request
    return SimpleNamespace(
        messages=(
            {
                "role": "system",
                "content": "Deterministic benchmark. No reasoning. Return only minified JSON.",
            },
            {
                "role": "user",
                "content": (
                    "Emit keys title,modules,constraints. modules has exactly 12 objects m00..m11; "
                    "each has id,kind,depends_on,registry_key,priority. Cycle kind through block,item,"
                    "entity,event,network,ui; use distinct registry keys and a valid mixed dependency "
                    "DAG. constraints has deterministic,fabric_api,server_safe,client_safe all true."
                ),
            },
        ),
        response_format="text",
    )


def _probe_p_min(autotune: Any, binary: str, model_path: str, config: Any, request: Any, selected: Any):
    if str(getattr(selected, "spec_type", "none")) != "draft-mtp":
        return _as_speed_variant(selected), ()
    candidates = _mtp_p_min_candidates()
    if len(candidates) <= 1:
        return _as_speed_variant(selected), ()

    bench_request = autotune._compact_benchmark_request(request)
    probe_tokens = min(
        int(config.max_new_tokens),
        autotune._env_int(
            "MMM_LLAMA_AUTOTUNE_TOKENS", autotune._BENCHMARK_OUTPUT_TOKENS
        ),
    )
    probes = []
    preferred_port = autotune._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910)
    for p_min in candidates:
        base = _as_speed_variant(selected, draft_p_min=p_min)
        root_name = base.name.split("|pm", 1)[0]
        variant = replace(base, name=root_name if p_min == 0 else f"{root_name}|pm{p_min:g}")
        process = None
        started = time.perf_counter()
        try:
            port = autotune._free_port(preferred_port)
            process = autotune._start_server(binary, model_path, config, variant, port)
            url = autotune._wait_ready(process, port)
            autotune._probe_server(url, bench_request, max_tokens=1, variant=variant)
            probe = autotune._probe_server(
                url, bench_request, max_tokens=probe_tokens, variant=variant
            )
        except Exception as exc:
            probe = autotune.ProbeResult(
                variant=variant,
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
        probes.append(probe)

    baseline = probes[0]
    if not bool(getattr(baseline, "ok", False)):
        return _as_speed_variant(selected), tuple(probes)
    valid = [
        p for p in probes
        if bool(getattr(p, "ok", False))
        and str(getattr(p, "output_sha256", "")) == str(getattr(baseline, "output_sha256", ""))
        and float(getattr(p, "predicted_tps", 0.0)) > 0
    ]
    if not valid:
        return _as_speed_variant(selected), tuple(probes)
    best = max(valid, key=lambda p: float(getattr(p, "predicted_tps", 0.0)))
    minimum_gain = autotune._env_float("MMM_LLAMA_STAGE_MIN_GAIN", 1.01)
    if best is not baseline and _decode_ratio(best, baseline) < max(1.0, minimum_gain):
        best = baseline
    return _as_speed_variant(best.variant), tuple(probes)


def install(autotune: Any, runtime_tuning: Any) -> None:
    """Optimize normal MMM local inference for one-response llama TG tok/s."""
    with _INSTALL_LOCK:
        if getattr(autotune, "_mmm_decode_speed_contract_installed", False):
            return

        autotune.ServerVariant = SpeedServerVariant
        runtime_tuning.ServerVariant = SpeedServerVariant
        autotune._BENCHMARK_SCHEMA_VERSION = _SCHEMA_VERSION
        runtime_tuning._SCHEMA_VERSION = _SCHEMA_VERSION
        autotune._BENCHMARK_OUTPUT_TOKENS = max(
            256, int(getattr(autotune, "_BENCHMARK_OUTPUT_TOKENS", 96))
        )

        current_compact = autotune._compact_benchmark_request
        _representative_benchmark_request.__wrapped__ = current_compact
        _representative_benchmark_request._mmm_representative_decode_probe = True
        autotune._compact_benchmark_request = _representative_benchmark_request

        current_variant_args = autotune._variant_args

        @wraps(current_variant_args)
        def variant_args(variant: Any) -> list[str]:
            args = list(current_variant_args(variant))
            p_min = float(getattr(variant, "draft_p_min", 0.0) or 0.0)
            if str(getattr(variant, "spec_type", "none")) == "draft-mtp" and p_min > 0:
                args.extend(["--spec-draft-p-min", f"{p_min:g}"])
            return args

        for tag in ("_mmm_auto_draft_layers", "_mmm_ngram_speculation"):
            if getattr(current_variant_args, tag, False):
                setattr(variant_args, tag, True)
        variant_args._mmm_mtp_p_min_tuning = True
        autotune._variant_args = variant_args

        current_fingerprint = autotune._fingerprint

        @wraps(current_fingerprint)
        def fingerprint(config: Any, binary: str, model_path: str) -> str:
            payload = {
                "schema": _SCHEMA_VERSION,
                "base": current_fingerprint(config, binary, model_path),
                "objective": _tuning_objective(),
                "explicit_parallel": _explicit_parallel_requested(),
                "mtp_p_min": _mtp_p_min_candidates(),
                "probe_tokens": int(autotune._BENCHMARK_OUTPUT_TOKENS),
                "probe_shape": "varied-planner-json-v1",
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

        for tag in ("_mmm_stable_model_signature", "_mmm_runtime_tuning_fingerprint"):
            if getattr(current_fingerprint, tag, False):
                setattr(fingerprint, tag, True)
        fingerprint._mmm_decode_objective_fingerprint = True
        autotune._fingerprint = fingerprint

        current_benchmark = autotune._benchmark

        @wraps(current_benchmark)
        def benchmark(binary: str, model_path: str, config: Any, request: Any, fingerprint_value: str):
            with _BENCHMARK_LOCK:
                old_target = runtime_tuning._parallel_target
                old_candidates = runtime_tuning._parallel_candidates
                old_score = runtime_tuning._balanced_score
                try:
                    if _tuning_objective() == "single_stream" and not _explicit_parallel_requested():
                        runtime_tuning._parallel_target = lambda: 1
                        runtime_tuning._parallel_candidates = lambda: ()
                    runtime_tuning._balanced_score = _decode_ratio
                    decision = current_benchmark(
                        binary, model_path, config, request, fingerprint_value
                    )
                finally:
                    runtime_tuning._parallel_target = old_target
                    runtime_tuning._parallel_candidates = old_candidates
                    runtime_tuning._balanced_score = old_score

            if decision is None:
                return None
            selected, extra_probes = _probe_p_min(
                autotune, binary, model_path, config, request, decision.selected
            )
            selected_probe = next(
                (
                    p for p in reversed(extra_probes)
                    if getattr(p, "variant", None) == selected and bool(getattr(p, "ok", False))
                ),
                None,
            )
            selected_tps = (
                float(getattr(selected_probe, "predicted_tps", 0.0))
                if selected_probe is not None
                else float(getattr(decision, "selected_tps", 0.0))
            )
            baseline_tps = float(getattr(decision, "baseline_tps", 0.0))
            return replace(
                decision,
                selected=selected,
                selected_tps=selected_tps,
                speedup=selected_tps / baseline_tps if baseline_tps > 0 else 1.0,
                probes=tuple(decision.probes) + tuple(extra_probes),
            )

        for tag in (
            "_mmm_staged_runtime_tuning",
            "_mmm_model_eligible_speculation",
            "_mmm_single_server_cache_stage",
        ):
            if getattr(current_benchmark, tag, False):
                setattr(benchmark, tag, True)
        benchmark._mmm_single_stream_decode_objective = True
        benchmark._mmm_mtp_p_min_stage = True
        autotune._benchmark = benchmark

        current_launch = autotune._launch_selected

        @wraps(current_launch)
        def launch(binary: str, model_path: str, config: Any, selected: Any) -> str:
            url = current_launch(binary, model_path, config, selected)
            os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] = f"{float(getattr(selected, 'draft_p_min', 0.0) or 0.0):g}"
            os.environ["MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE"] = _tuning_objective()
            return url

        if getattr(current_launch, "_mmm_exports_active_runtime", False):
            launch._mmm_exports_active_runtime = True
        launch._mmm_exports_decode_speed_state = True
        autotune._launch_selected = launch
        autotune._mmm_decode_speed_contract_installed = True


__all__ = [
    "SpeedServerVariant",
    "_as_speed_variant",
    "_decode_ratio",
    "_explicit_parallel_requested",
    "_mtp_p_min_candidates",
    "_representative_benchmark_request",
    "_tuning_objective",
    "install",
]
