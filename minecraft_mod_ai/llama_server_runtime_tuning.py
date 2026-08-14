from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import wraps
from typing import Any, Iterable

_SCHEMA_VERSION = "mmm/llama-server-autotune-v6-adaptive-joint-mtp"
_INSTALL_LOCK = threading.RLock()


@dataclass(frozen=True)
class ServerVariant:
    name: str
    spec_type: str = "none"
    draft_n_max: int = 0
    ubatch: int = 0
    parallel: int = 1
    cache_reuse: int = 0
    draft_p_min: float = 0.0


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


def _parse_int_candidates(raw: str, *, minimum: int, maximum: int, required: Iterable[int] = ()) -> tuple[int, ...]:
    values: list[int] = []
    for value in required:
        value = int(value)
        if minimum <= value <= maximum and value not in values:
            values.append(value)
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if minimum <= value <= maximum and value not in values:
            values.append(value)
    return tuple(values)


def _replace_option(args: list[str], names: tuple[str, ...], value: str) -> None:
    for name in names:
        try:
            index = args.index(name)
        except ValueError:
            continue
        if index + 1 < len(args):
            args[index + 1] = value
            return
    args.extend([names[0], value])


def _remove_option(args: list[str], names: tuple[str, ...], *, takes_value: bool = True) -> None:
    for name in names:
        while name in args:
            index = args.index(name)
            del args[index]
            if takes_value and index < len(args):
                del args[index]


def _ubatch_candidates(autotune_module: Any) -> tuple[int, ...]:
    batch = autotune_module._env_int("MMM_LLAMA_BATCH", 2048)
    current = min(batch, autotune_module._env_int("MMM_LLAMA_UBATCH", 512))
    return _parse_int_candidates(
        os.environ.get("MMM_LLAMA_UBATCH_CANDIDATES", "512,1024,2048"),
        minimum=64,
        maximum=max(64, batch),
        required=(current,),
    )


def _cache_reuse_candidates() -> tuple[int, ...]:
    return _parse_int_candidates(
        os.environ.get("MMM_LLAMA_CACHE_REUSE_CANDIDATES", "0,64,256"),
        minimum=0,
        maximum=8192,
        required=(0,),
    )


def _parallel_target() -> int:
    return _env_int("MMM_LLAMA_CONCURRENT_REQUESTS", 2, minimum=1, maximum=8)


def _parallel_candidates() -> tuple[int, ...]:
    explicit = os.environ.get("MMM_LLAMA_PARALLEL", "").strip()
    if explicit:
        try:
            return (max(1, min(8, int(explicit))),)
        except ValueError:
            return (1,)
    target = _parallel_target()
    values = [1]
    for value in (2, 4, 8):
        if value <= target:
            values.append(value)
    if target not in values:
        values.append(target)
    return tuple(sorted(set(values)))


def _candidate_variants(autotune_module: Any) -> tuple[ServerVariant, ...]:
    del autotune_module
    base_widths = _parse_int_candidates(
        os.environ.get("MMM_LLAMA_MTP_WIDTHS", "1,2,3"), minimum=1, maximum=32
    )
    confidence_widths = _parse_int_candidates(
        os.environ.get("MMM_LLAMA_MTP_CONFIDENCE_WIDTHS", "2,4,8,16"), minimum=1, maximum=32
    )
    try:
        confidence_p_min = float(os.environ.get("MMM_LLAMA_MTP_SEED_P_MIN", "0.8"))
    except ValueError:
        confidence_p_min = 0.8
    confidence_p_min = max(0.0, min(0.999, confidence_p_min))

    values = [ServerVariant("baseline")]
    values.extend(ServerVariant(f"mtp-{w}", "draft-mtp", w) for w in base_widths)
    if confidence_p_min > 0:
        for width in confidence_widths:
            candidate = ServerVariant(
                f"mtp-{width}|pm{confidence_p_min:g}", "draft-mtp", width,
                draft_p_min=confidence_p_min,
            )
            if all(
                not (
                    item.spec_type == "draft-mtp"
                    and item.draft_n_max == candidate.draft_n_max
                    and float(getattr(item, "draft_p_min", 0.0)) == candidate.draft_p_min
                )
                for item in values
            ):
                values.append(candidate)

    allowed = {"ngram-simple", "ngram-mod", "ngram-map-k", "ngram-map-k4v", "ngram-cache"}
    for token in os.environ.get("MMM_LLAMA_NGRAM_SPEC_TYPES", "ngram-simple,ngram-mod,ngram-map-k").split(","):
        spec_type = token.strip()
        if spec_type in allowed and all(item.spec_type != spec_type for item in values):
            values.append(ServerVariant(spec_type, spec_type))
    return tuple(values)


def _model_supports_mtp(config: Any) -> bool:
    extra = getattr(config, "extra", {})
    if isinstance(extra, dict) and "supports_mtp" in extra:
        return bool(extra.get("supports_mtp"))
    model_id = str(getattr(config, "model_id", "")).upper()
    filename = str(extra.get("gguf_filename", "")).upper() if isinstance(extra, dict) else ""
    return "-MTP-GGUF" in model_id or "-MTP-" in filename


def _candidate_variants_for_config(autotune_module: Any, config: Any) -> tuple[ServerVariant, ...]:
    values = _candidate_variants(autotune_module)
    if _model_supports_mtp(config):
        return values
    return tuple(value for value in values if value.spec_type != "draft-mtp")


def _eligible(probe: Any, baseline: Any) -> bool:
    return bool(getattr(probe, "ok", False)) and str(getattr(probe, "output_sha256", "")) == str(getattr(baseline, "output_sha256", "")) and float(getattr(probe, "predicted_tps", 0.0)) > 0


def _balanced_score(probe: Any, baseline: Any) -> float:
    decode_base = max(1e-9, float(getattr(baseline, "predicted_tps", 0.0)))
    decode = max(0.0, float(getattr(probe, "predicted_tps", 0.0))) / decode_base
    prompt_base = float(getattr(baseline, "prompt_tps", 0.0))
    prompt_value = float(getattr(probe, "prompt_tps", 0.0))
    prompt = prompt_value / prompt_base if prompt_base > 0 and prompt_value > 0 else 1.0
    return 0.70 * decode + 0.30 * prompt


def _select_probe(probes: list[Any], *, balanced: bool, minimum_gain: float) -> Any | None:
    if not probes:
        return None
    baseline = probes[0]
    if not getattr(baseline, "ok", False) or float(getattr(baseline, "predicted_tps", 0.0)) <= 0:
        return None
    valid = [baseline] + [probe for probe in probes[1:] if _eligible(probe, baseline)]
    score = (lambda probe: _balanced_score(probe, baseline)) if balanced else (lambda probe: float(getattr(probe, "predicted_tps", 0.0)) / max(1e-9, float(getattr(baseline, "predicted_tps", 0.0))))
    best = max(valid, key=score)
    return best if best is baseline or score(best) >= max(1.0, minimum_gain) else baseline


def _parallel_probe(autotune_module: Any, base_url: str, request: Any, *, max_tokens: int, variant: ServerVariant, concurrency: int) -> Any:
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="mmm_llama_slot_probe") as pool:
            values = [future.result() for future in [pool.submit(autotune_module._probe_server, base_url, request, max_tokens=max_tokens, variant=variant) for _ in range(concurrency)]]
        elapsed = time.perf_counter() - started
        first = values[0]
        if not all(getattr(value, "ok", False) for value in values) or any(getattr(value, "output_sha256", "") != getattr(first, "output_sha256", "") for value in values[1:]):
            raise RuntimeError("parallel slot probe produced non-identical or failed outputs")
        tokens = sum(int(getattr(value, "predicted_tokens", 0)) for value in values)
        prompt_values = [float(getattr(value, "prompt_tps", 0.0)) for value in values if float(getattr(value, "prompt_tps", 0.0)) > 0]
        return autotune_module.ProbeResult(variant=variant, ok=tokens > 0 and elapsed > 0, output_sha256=str(getattr(first, "output_sha256", "")), predicted_tokens=tokens, predicted_tps=(tokens / elapsed if elapsed > 0 else 0.0), prompt_tps=(sum(prompt_values) / len(prompt_values) if prompt_values else 0.0), elapsed_seconds=elapsed)
    except Exception as exc:
        return autotune_module.ProbeResult(variant=variant, ok=False, output_sha256="", predicted_tokens=0, predicted_tps=0.0, prompt_tps=0.0, elapsed_seconds=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")


def install(autotune_module: Any) -> None:
    """Install compile-free adaptive staged tuning over verified native llama-server."""
    with _INSTALL_LOCK:
        if getattr(autotune_module, "_mmm_runtime_tuning_installed", False):
            return
        autotune_module.ServerVariant = ServerVariant
        autotune_module._BENCHMARK_SCHEMA_VERSION = _SCHEMA_VERSION

        def candidate_variants() -> tuple[ServerVariant, ...]:
            return _candidate_variants(autotune_module)
        candidate_variants._mmm_mtp_ngram_autotune = True
        autotune_module._candidate_variants = candidate_variants

        current_base = autotune_module._base_args
        @wraps(current_base)
        def tuned_base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
            args = list(current_base(binary, model_path, config, port))
            _replace_option(args, ("--load-mode", "-lm"), "auto")
            if "--cache-prompt" not in args:
                args.append("--cache-prompt")
            return args
        for tag in ("_mmm_auto_gpu_layers", "_mmm_single_decode_slot", "_mmm_native_telemetry_endpoints"):
            if getattr(current_base, tag, False):
                setattr(tuned_base_args, tag, True)
        tuned_base_args._mmm_load_mode_auto = True
        autotune_module._base_args = tuned_base_args

        current_variant_args = autotune_module._variant_args
        @wraps(current_variant_args)
        def tuned_variant_args(variant: ServerVariant) -> list[str]:
            if variant.spec_type.startswith("ngram-"):
                return ["--spec-type", variant.spec_type]
            return current_variant_args(variant)
        if getattr(current_variant_args, "_mmm_auto_draft_layers", False):
            tuned_variant_args._mmm_auto_draft_layers = True
        tuned_variant_args._mmm_ngram_speculation = True
        autotune_module._variant_args = tuned_variant_args

        def start_server(binary: str, model_path: str, config: Any, variant: ServerVariant, port: int) -> subprocess.Popen[bytes]:
            debug = autotune_module._env_bool("MMM_LLAMA_AUTOTUNE_DEBUG", False)
            stream = None if debug else subprocess.DEVNULL
            args = list(autotune_module._base_args(binary, model_path, config, port))
            if variant.ubatch > 0:
                _replace_option(args, ("--ubatch-size", "-ub"), str(variant.ubatch))
            _replace_option(args, ("--parallel", "-np"), str(max(1, variant.parallel)))
            if variant.parallel > 1:
                if "--cont-batching" not in args and "-cb" not in args:
                    args.append("--cont-batching")
                if "--kv-unified" not in args and "-kvu" not in args:
                    args.append("--kv-unified")
            _remove_option(args, ("--cache-reuse",), takes_value=True)
            if variant.cache_reuse > 0:
                args.extend(["--cache-reuse", str(variant.cache_reuse)])
            args.extend(autotune_module._variant_args(variant))
            return subprocess.Popen(args, stdout=stream, stderr=stream)
        start_server._mmm_staged_runtime_tuning = True
        autotune_module._start_server = start_server

        current_fingerprint = autotune_module._fingerprint
        @wraps(current_fingerprint)
        def tuning_fingerprint(config: Any, binary: str, model_path: str) -> str:
            payload = {"schema": _SCHEMA_VERSION, "base": current_fingerprint(config, binary, model_path), "mtp_supported": _model_supports_mtp(config), "load_mode": "auto", "spec_variants": [asdict(value) for value in _candidate_variants_for_config(autotune_module, config)], "ubatch_candidates": _ubatch_candidates(autotune_module), "cache_reuse_candidates": _cache_reuse_candidates(), "parallel_candidates": _parallel_candidates(), "concurrent_requests": _parallel_target(), "search": "adaptive-joint-mtp-seeds+ngram-fallback+ubatch-exhaustive"}
            return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if getattr(current_fingerprint, "_mmm_stable_model_signature", False):
            tuning_fingerprint._mmm_stable_model_signature = True
        tuning_fingerprint._mmm_runtime_tuning_fingerprint = True
        autotune_module._fingerprint = tuning_fingerprint

        def run_variant(binary: str, model_path: str, config: Any, benchmark_request: Any, variant: ServerVariant, *, probe_tokens: int, parallel_probe: bool = False, concurrency: int = 1) -> Any:
            port = autotune_module._free_port(autotune_module._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910))
            process = None
            try:
                process = autotune_module._start_server(binary, model_path, config, variant, port)
                url = autotune_module._wait_ready(process, port)
                autotune_module._probe_server(url, benchmark_request, max_tokens=1, variant=variant)
                if parallel_probe:
                    return _parallel_probe(autotune_module, url, benchmark_request, max_tokens=probe_tokens, variant=variant, concurrency=concurrency)
                return autotune_module._probe_server(url, benchmark_request, max_tokens=probe_tokens, variant=variant)
            except Exception as exc:
                return autotune_module.ProbeResult(variant=variant, ok=False, output_sha256="", predicted_tokens=0, predicted_tps=0.0, prompt_tps=0.0, elapsed_seconds=0.0, error=f"{type(exc).__name__}: {exc}")
            finally:
                autotune_module._stop_server(process)
        autotune_module._mmm_run_tuning_variant = run_variant

        def benchmark(binary: str, model_path: str, config: Any, request: Any, fingerprint: str) -> Any | None:
            benchmark_request = autotune_module._compact_benchmark_request(request)
            probe_tokens = min(int(config.max_new_tokens), autotune_module._env_int("MMM_LLAMA_AUTOTUNE_TOKENS", autotune_module._BENCHMARK_OUTPUT_TOKENS))
            stage_gain = autotune_module._env_float("MMM_LLAMA_STAGE_MIN_GAIN", 1.01)
            spec_gain = max(stage_gain, autotune_module._env_float("MMM_LLAMA_AUTOTUNE_MIN_SPEEDUP", 1.01))
            probes: list[Any] = []
            values = _candidate_variants_for_config(autotune_module, config)
            baseline_variant = next(value for value in values if value.spec_type == "none")
            baseline = run_variant(binary, model_path, config, benchmark_request, baseline_variant, probe_tokens=probe_tokens)
            probes.append(baseline)
            if not getattr(baseline, "ok", False) or float(getattr(baseline, "predicted_tps", 0.0)) <= 0:
                return None

            mtp_values = [value for value in values if value.spec_type == "draft-mtp"]
            ngram_values = [value for value in values if value.spec_type.startswith("ngram-")]
            primary = mtp_values if _model_supports_mtp(config) else ngram_values
            primary_probes = [baseline]
            for variant in primary:
                probe = run_variant(binary, model_path, config, benchmark_request, variant, probe_tokens=probe_tokens)
                probes.append(probe)
                primary_probes.append(probe)
            spec = _select_probe(primary_probes, balanced=False, minimum_gain=spec_gain) or baseline

            # MTP-capable Qwen models do not pay for ngram probes when native MTP wins.
            if _model_supports_mtp(config) and spec is baseline and ngram_values:
                fallback_probes = [baseline]
                for variant in ngram_values:
                    probe = run_variant(binary, model_path, config, benchmark_request, variant, probe_tokens=probe_tokens)
                    probes.append(probe)
                    fallback_probes.append(probe)
                spec = _select_probe(fallback_probes, balanced=False, minimum_gain=spec_gain) or baseline

            selected = replace(spec.variant, ubatch=min(autotune_module._env_int("MMM_LLAMA_BATCH", 2048), autotune_module._env_int("MMM_LLAMA_UBATCH", 512)))
            final_probe = spec
            base_ubatch = selected.ubatch
            # Every ubatch candidate is independent: a local dip must not hide a faster later setting.
            for value in sorted((v for v in _ubatch_candidates(autotune_module) if v != base_ubatch)):
                variant = replace(selected, name=f"{selected.name.split('|ub', 1)[0]}|ub{value}", ubatch=value)
                probe = run_variant(binary, model_path, config, benchmark_request, variant, probe_tokens=probe_tokens)
                probes.append(probe)
                if not _eligible(probe, baseline):
                    continue
                if _balanced_score(probe, final_probe) >= max(1.0, stage_gain):
                    selected = variant
                    final_probe = probe

            concurrency = _parallel_target()
            for value in _parallel_candidates():
                if value == max(1, selected.parallel):
                    continue
                variant = replace(selected, name=f"{selected.name.split('|p', 1)[0]}|p{value}", parallel=value)
                probe = run_variant(binary, model_path, config, benchmark_request, variant, probe_tokens=probe_tokens, parallel_probe=True, concurrency=concurrency)
                probes.append(probe)
                if _eligible(probe, baseline) and float(getattr(probe, "predicted_tps", 0.0)) >= float(getattr(final_probe, "predicted_tps", 0.0)) * max(1.0, stage_gain):
                    selected = variant
                    final_probe = probe

            baseline_tps = float(getattr(baseline, "predicted_tps", 0.0))
            selected_tps = float(getattr(final_probe, "predicted_tps", 0.0))
            return autotune_module.AutotuneDecision(fingerprint=fingerprint, selected=selected, baseline_tps=baseline_tps, selected_tps=selected_tps, speedup=(selected_tps / baseline_tps if baseline_tps > 0 else 1.0), probes=tuple(probes))

        benchmark._mmm_staged_runtime_tuning = True
        benchmark._mmm_model_eligible_speculation = True
        benchmark._mmm_adaptive_cold_search = True
        benchmark._mmm_adaptive_joint_mtp_search = True
        benchmark._mmm_exhaustive_ubatch_search = True
        autotune_module._benchmark = benchmark

        current_launch = autotune_module._launch_selected
        @wraps(current_launch)
        def launch_selected(binary: str, model_path: str, config: Any, selected: ServerVariant) -> str:
            url = current_launch(binary, model_path, config, selected)
            os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = str(max(1, selected.parallel))
            os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(selected.ubatch or min(autotune_module._env_int("MMM_LLAMA_BATCH", 2048), autotune_module._env_int("MMM_LLAMA_UBATCH", 512)))
            os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = str(selected.cache_reuse)
            os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = selected.spec_type
            return url
        launch_selected._mmm_exports_active_runtime = True
        autotune_module._launch_selected = launch_selected
        autotune_module._mmm_runtime_tuning_installed = True


__all__ = ["ServerVariant", "_cache_reuse_candidates", "_candidate_variants_for_config", "_model_supports_mtp", "_parallel_candidates", "_parallel_target", "_parse_int_candidates", "_replace_option", "_ubatch_candidates", "install"]
