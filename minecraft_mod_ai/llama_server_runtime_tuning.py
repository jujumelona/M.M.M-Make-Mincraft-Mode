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
    # Candidate construction is runtime-tuning policy, not an autotune hook.  Keep
    # it independent from wrappers that may temporarily reinterpret autotune's
    # environment reader while composing cold-start tuning stages.
    del autotune_module
    batch = _env_int("MMM_LLAMA_BATCH", 2048)
    current = min(batch, _env_int("MMM_LLAMA_UBATCH", 512))
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

        original_base_args = autotune_module._base_args
        original_variant_args = autotune_module._variant_args
        original_candidate_variants = autotune_module._candidate_variants
        original_fingerprint = autotune_module._fingerprint
        original_benchmark = autotune_module._benchmark

        @wraps(original_base_args)
        def base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
            args = list(original_base_args(binary, model_path, config, port))
            _replace_option(args, ("--ubatch-size", "-ub", "--ubatch"), str(_env_int("MMM_LLAMA_ACTIVE_UBATCH", _env_int("MMM_LLAMA_UBATCH", 512), minimum=64)))
            _replace_option(args, ("--parallel", "-np"), str(_env_int("MMM_LLAMA_ACTIVE_PARALLEL", _env_int("MMM_LLAMA_PARALLEL", 1), maximum=8)))
            _replace_option(args, ("--cache-reuse",), str(_env_int("MMM_LLAMA_ACTIVE_CACHE_REUSE", _env_int("MMM_LLAMA_CACHE_REUSE", 0, minimum=0), minimum=0, maximum=8192)))
            return args

        @wraps(original_variant_args)
        def variant_args(variant: ServerVariant) -> list[str]:
            args = list(original_variant_args(variant))
            if variant.draft_p_min > 0 and variant.spec_type == "draft-mtp":
                _replace_option(args, ("--draft-p-min", "--spec-draft-p-min"), f"{variant.draft_p_min:g}")
            return args

        @wraps(original_candidate_variants)
        def candidate_variants() -> tuple[ServerVariant, ...]:
            return _candidate_variants(autotune_module)

        @wraps(original_fingerprint)
        def fingerprint(config: Any, binary: str, model_path: str) -> str:
            base = original_fingerprint(config, binary, model_path)
            payload = {
                "base": base,
                "schema": _SCHEMA_VERSION,
                "variants": [asdict(value) for value in _candidate_variants_for_config(autotune_module, config)],
                "ubatch": _ubatch_candidates(autotune_module),
                "parallel": _parallel_candidates(),
                "cache_reuse": _cache_reuse_candidates(),
            }
            return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        def run_variant(binary: str, model_path: str, config: Any, request: Any, variant: ServerVariant, *, probe_tokens: int) -> Any:
            port = autotune_module._free_port()
            args = base_args(binary, model_path, config, port) + variant_args(variant)
            process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            base_url = f"http://127.0.0.1:{port}/v1"
            try:
                autotune_module._wait_ready(process, base_url, timeout=autotune_module._STARTUP_TIMEOUT)
                return autotune_module._probe_server(base_url, request, max_tokens=probe_tokens, variant=variant)
            finally:
                autotune_module._stop_process(process)

        @wraps(original_benchmark)
        def benchmark(binary: str, model_path: str, config: Any, request: Any) -> list[Any]:
            compact = autotune_module._compact_benchmark_request(request)
            probe_tokens = min(int(getattr(config, "max_new_tokens", 256) or 256), autotune_module._env_int("MMM_LLAMA_AUTOTUNE_TOKENS", autotune_module._BENCHMARK_OUTPUT_TOKENS))
            minimum_gain = max(1.0, float(os.environ.get("MMM_LLAMA_AUTOTUNE_MIN_SPEEDUP", str(autotune_module._MIN_SPEEDUP))))
            variants = _candidate_variants_for_config(autotune_module, config)
            probes = [run_variant(binary, model_path, config, compact, variant, probe_tokens=probe_tokens) for variant in variants]
            selected = _select_probe(probes, balanced=False, minimum_gain=minimum_gain)
            if selected is None:
                return probes

            active = selected.variant
            os.environ["MMM_LLAMA_ACTIVE_DRAFT_P_MIN"] = f"{active.draft_p_min:g}"
            # Subsequent runtime axes share one loaded server where possible.
            ubatch_values = _ubatch_candidates(autotune_module)
            if len(ubatch_values) > 1:
                ubatch_probes = []
                for value in ubatch_values:
                    variant = replace(active, name=f"{active.name}|ub{value}", ubatch=value)
                    os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(value)
                    ubatch_probes.append(run_variant(binary, model_path, config, compact, variant, probe_tokens=probe_tokens))
                winner = _select_probe(ubatch_probes, balanced=False, minimum_gain=minimum_gain)
                if winner is not None:
                    active = winner.variant
                    os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(active.ubatch)
                    probes.extend(ubatch_probes)

            # Cache-reuse and parallelism are request/runtime concerns; keep the
            # selected values explicit for the server args wrapper.
            cache_values = _cache_reuse_candidates()
            if cache_values:
                os.environ.setdefault("MMM_LLAMA_ACTIVE_CACHE_REUSE", str(cache_values[0]))
            parallel_values = _parallel_candidates()
            if parallel_values:
                os.environ.setdefault("MMM_LLAMA_ACTIVE_PARALLEL", str(parallel_values[0]))
            return probes

        benchmark._mmm_staged_runtime_tuning = True
        benchmark._mmm_single_server_cache_stage = True
        benchmark._mmm_adaptive_joint_mtp_search = True
        benchmark._mmm_exhaustive_ubatch_search = True
        fingerprint._mmm_runtime_tuning_fingerprint = True
        autotune_module._base_args = base_args
        autotune_module._variant_args = variant_args
        autotune_module._candidate_variants = candidate_variants
        autotune_module._fingerprint = fingerprint
        autotune_module._benchmark = benchmark
        autotune_module._mmm_runtime_tuning_installed = True
