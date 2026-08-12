from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Iterable

_SCHEMA_VERSION = "mmm/llama-server-autotune-v3-max-performance"
_INSTALL_LOCK = threading.RLock()
_CUDA_GRAPH_LOCK = threading.RLock()


@dataclass(frozen=True)
class ServerVariant:
    name: str
    spec_type: str = "none"
    draft_n_max: int = 0
    ubatch: int = 0
    parallel: int = 1
    cache_reuse: int = 0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    if value < minimum:
        value = minimum
    if maximum is not None:
        value = min(maximum, value)
    return value


def _parse_int_candidates(
    raw: str,
    *,
    minimum: int,
    maximum: int,
    required: Iterable[int] = (),
) -> tuple[int, ...]:
    values: list[int] = []
    for value in required:
        if minimum <= int(value) <= maximum and int(value) not in values:
            values.append(int(value))
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
    raw = os.environ.get("MMM_LLAMA_UBATCH_CANDIDATES", "512,1024,2048")
    return _parse_int_candidates(
        raw,
        minimum=64,
        maximum=max(64, batch),
        required=(current,),
    )


def _cache_reuse_candidates() -> tuple[int, ...]:
    raw = os.environ.get("MMM_LLAMA_CACHE_REUSE_CANDIDATES", "0,64,256")
    return _parse_int_candidates(raw, minimum=0, maximum=8192, required=(0,))


def _parallel_target() -> int:
    return _env_int("MMM_LLAMA_CONCURRENT_REQUESTS", 2, minimum=1, maximum=8)


def _parallel_candidates() -> tuple[int, ...]:
    explicit = os.environ.get("MMM_LLAMA_PARALLEL", "").strip()
    if explicit:
        try:
            value = int(explicit)
        except ValueError:
            value = 1
        return (max(1, min(8, value)),)
    target = _parallel_target()
    values = [1]
    for value in (2, 4, 8):
        if value <= target and value not in values:
            values.append(value)
    if target not in values:
        values.append(target)
    return tuple(sorted(set(values)))


def _candidate_variants(autotune_module: Any) -> tuple[ServerVariant, ...]:
    raw = os.environ.get("MMM_LLAMA_MTP_WIDTHS", "1,2,3")
    widths: list[int] = []
    for token in raw.split(","):
        try:
            width = int(token.strip())
        except ValueError:
            continue
        if 1 <= width <= 8 and width not in widths:
            widths.append(width)

    values = [ServerVariant("baseline")]
    values.extend(
        ServerVariant(f"mtp-{width}", "draft-mtp", width)
        for width in widths
    )
    raw_ngram = os.environ.get(
        "MMM_LLAMA_NGRAM_SPEC_TYPES",
        "ngram-simple,ngram-mod,ngram-map-k",
    )
    supported = {"ngram-simple", "ngram-mod", "ngram-map-k", "ngram-map-k4v", "ngram-cache"}
    for token in raw_ngram.split(","):
        spec_type = token.strip()
        if spec_type in supported and all(value.spec_type != spec_type for value in values):
            values.append(ServerVariant(spec_type, spec_type))
    return tuple(values)


def _balanced_score(probe: Any, baseline: Any) -> float:
    if not getattr(probe, "ok", False):
        return 0.0
    decode_base = max(1e-9, float(getattr(baseline, "predicted_tps", 0.0)))
    decode = max(0.0, float(getattr(probe, "predicted_tps", 0.0))) / decode_base
    prompt_base = float(getattr(baseline, "prompt_tps", 0.0))
    prompt_value = float(getattr(probe, "prompt_tps", 0.0))
    prompt = prompt_value / prompt_base if prompt_base > 0 and prompt_value > 0 else 1.0
    return 0.70 * decode + 0.30 * prompt


def _eligible(probe: Any, baseline: Any) -> bool:
    return (
        bool(getattr(probe, "ok", False))
        and str(getattr(probe, "output_sha256", ""))
        == str(getattr(baseline, "output_sha256", ""))
        and float(getattr(probe, "predicted_tps", 0.0)) > 0
    )


def _select_probe(
    probes: list[Any],
    *,
    balanced: bool,
    minimum_gain: float,
) -> Any | None:
    if not probes:
        return None
    baseline = probes[0]
    if not getattr(baseline, "ok", False) or float(getattr(baseline, "predicted_tps", 0.0)) <= 0:
        return None
    valid = [baseline] + [probe for probe in probes[1:] if _eligible(probe, baseline)]
    score = (
        (lambda probe: _balanced_score(probe, baseline))
        if balanced
        else (lambda probe: float(getattr(probe, "predicted_tps", 0.0)) / max(1e-9, float(getattr(baseline, "predicted_tps", 0.0))))
    )
    best = max(valid, key=score)
    return best if best is baseline or score(best) >= max(1.0, minimum_gain) else baseline


def _cuda_graph_build(autotune_module: Any, binary: str) -> str:
    if not _env_bool("MMM_LLAMA_CUDA_GRAPHS", True):
        return binary
    source_raw = os.environ.get("MMM_LLAMA_SERVER_SOURCE_DIR", "").strip()
    if not source_raw:
        return binary
    source = Path(source_raw).expanduser().resolve()
    if not (source / "CMakeLists.txt").is_file() or shutil.which("cmake") is None:
        return binary

    build_dir = source / "build"
    graph_binary = build_dir / "bin" / "llama-server"
    cache = build_dir / "CMakeCache.txt"

    def graphs_enabled() -> bool:
        try:
            return (
                cache.is_file()
                and "GGML_CUDA_GRAPHS:BOOL=ON"
                in cache.read_text(encoding="utf-8", errors="ignore")
            )
        except Exception:
            return False

    if graph_binary.is_file() and graphs_enabled():
        return str(graph_binary.resolve())

    with _CUDA_GRAPH_LOCK:
        if graph_binary.is_file() and graphs_enabled():
            return str(graph_binary.resolve())

        # Reconfigure the source-owned build in place. The Colab setup has already
        # compiled this exact pinned llama.cpp checkout, so CMake preserves its CUDA
        # architecture and all existing options and only rebuilds targets affected by
        # enabling CUDA graphs instead of compiling a second full server tree.
        command = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            "-DGGML_CUDA_GRAPHS=ON",
        ]
        if not cache.is_file():
            command.extend(
                [
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_CUDA=ON",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_APP=OFF",
                    "-DLLAMA_BUILD_UI=OFF",
                    "-DLLAMA_BUILD_TOOLS=ON",
                    "-DLLAMA_BUILD_SERVER=ON",
                ]
            )
        jobs = max(1, min(8, os.cpu_count() or 1))
        print("native llama-server: enabling CUDA graphs (incremental)", flush=True)
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--target",
                    "llama-server",
                    "-j",
                    str(jobs),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            print(
                "native llama-server: CUDA graphs build unavailable; "
                f"using verified baseline binary ({type(exc).__name__})",
                flush=True,
            )
            return binary
        if not graph_binary.is_file() or not graphs_enabled():
            return binary
        return str(graph_binary.resolve())


def _cache_probe(
    autotune_module: Any,
    base_url: str,
    *,
    max_tokens: int,
    variant: ServerVariant,
) -> Any:
    import httpx

    prefix = (
        "Minecraft Fabric deterministic repair context. "
        "Preserve package names, imports, registry identifiers, JSON keys, and API contracts. "
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
            "stream": False,
        }
        started = time.perf_counter()
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=autotune_module._env_int("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", 300),
        )
        response.raise_for_status()
        data = response.json()
        elapsed = time.perf_counter() - started
        output = autotune_module._assistant_output(data)
        timings = data.get("timings") or {}
        usage = data.get("usage") or {}
        predicted = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
        prompt_tps = float(timings.get("prompt_per_second") or 0.0)
        return output, predicted, prompt_tps, elapsed

    started = time.perf_counter()
    try:
        request(messages_a)
        output, predicted, prompt_tps, _ = request(messages_b)
        elapsed = time.perf_counter() - started
        end_to_end_tps = predicted / elapsed if predicted > 0 and elapsed > 0 else 0.0
        return autotune_module.ProbeResult(
            variant=variant,
            ok=bool(output) and predicted > 0,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            predicted_tokens=predicted,
            predicted_tps=end_to_end_tps,
            prompt_tps=prompt_tps,
            elapsed_seconds=elapsed,
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


def _parallel_probe(
    autotune_module: Any,
    base_url: str,
    request: Any,
    *,
    max_tokens: int,
    variant: ServerVariant,
    concurrency: int,
) -> Any:
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="mmm_llama_slot_probe",
        ) as pool:
            futures = [
                pool.submit(
                    autotune_module._probe_server,
                    base_url,
                    request,
                    max_tokens=max_tokens,
                    variant=variant,
                )
                for _ in range(concurrency)
            ]
            values = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        first = values[0]
        if (
            not all(getattr(value, "ok", False) for value in values)
            or any(
                getattr(value, "output_sha256", "") != getattr(first, "output_sha256", "")
                for value in values[1:]
            )
        ):
            raise RuntimeError("parallel slot probe produced non-identical or failed outputs")
        tokens = sum(int(getattr(value, "predicted_tokens", 0)) for value in values)
        aggregate_tps = tokens / elapsed if tokens > 0 and elapsed > 0 else 0.0
        prompt_values = [
            float(getattr(value, "prompt_tps", 0.0))
            for value in values
            if float(getattr(value, "prompt_tps", 0.0)) > 0
        ]
        return autotune_module.ProbeResult(
            variant=variant,
            ok=aggregate_tps > 0,
            output_sha256=str(getattr(first, "output_sha256", "")),
            predicted_tokens=tokens,
            predicted_tps=aggregate_tps,
            prompt_tps=(sum(prompt_values) / len(prompt_values) if prompt_values else 0.0),
            elapsed_seconds=elapsed,
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


def install(autotune_module: Any) -> None:
    with _INSTALL_LOCK:
        if getattr(autotune_module, "_mmm_max_server_performance_installed", False):
            return

        autotune_module.ServerVariant = ServerVariant
        autotune_module._BENCHMARK_SCHEMA_VERSION = _SCHEMA_VERSION

        def candidate_variants() -> tuple[ServerVariant, ...]:
            return _candidate_variants(autotune_module)

        candidate_variants._mmm_mtp_ngram_autotune = True  # type: ignore[attr-defined]
        autotune_module._candidate_variants = candidate_variants

        current_base = autotune_module._base_args

        @wraps(current_base)
        def performance_base_args(
            binary: str,
            model_path: str,
            config: Any,
            port: int,
        ) -> list[str]:
            args = list(current_base(binary, model_path, config, port))
            _replace_option(args, ("--load-mode", "-lm"), "auto")
            if "--cache-prompt" not in args:
                args.append("--cache-prompt")
            return args

        for tag in (
            "_mmm_auto_gpu_layers",
            "_mmm_single_decode_slot",
            "_mmm_native_telemetry_endpoints",
        ):
            if getattr(current_base, tag, False):
                setattr(performance_base_args, tag, True)
        performance_base_args._mmm_load_mode_auto = True  # type: ignore[attr-defined]
        autotune_module._base_args = performance_base_args

        current_variant_args = autotune_module._variant_args

        @wraps(current_variant_args)
        def performance_variant_args(variant: ServerVariant) -> list[str]:
            if variant.spec_type.startswith("ngram-"):
                return ["--spec-type", variant.spec_type]
            return current_variant_args(variant)

        if getattr(current_variant_args, "_mmm_auto_draft_layers", False):
            performance_variant_args._mmm_auto_draft_layers = True  # type: ignore[attr-defined]
        performance_variant_args._mmm_ngram_speculation = True  # type: ignore[attr-defined]
        autotune_module._variant_args = performance_variant_args

        def start_server(
            binary: str,
            model_path: str,
            config: Any,
            variant: ServerVariant,
            port: int,
        ) -> subprocess.Popen[bytes]:
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

        start_server._mmm_staged_runtime_tuning = True  # type: ignore[attr-defined]
        autotune_module._start_server = start_server

        current_binary = autotune_module._server_binary

        @wraps(current_binary)
        def graph_server_binary() -> str | None:
            binary = current_binary()
            if not binary:
                return binary
            return _cuda_graph_build(autotune_module, binary)

        if getattr(current_binary, "_mmm_native_bootstrap", False):
            graph_server_binary._mmm_native_bootstrap = True  # type: ignore[attr-defined]
        graph_server_binary._mmm_cuda_graphs = True  # type: ignore[attr-defined]
        autotune_module._server_binary = graph_server_binary

        current_fingerprint = autotune_module._fingerprint

        @wraps(current_fingerprint)
        def performance_fingerprint(config: Any, binary: str, model_path: str) -> str:
            base = current_fingerprint(config, binary, model_path)
            payload = {
                "schema": _SCHEMA_VERSION,
                "base": base,
                "cuda_graphs": _env_bool("MMM_LLAMA_CUDA_GRAPHS", True),
                "load_mode": "auto",
                "spec_variants": [asdict(value) for value in candidate_variants()],
                "ubatch_candidates": _ubatch_candidates(autotune_module),
                "cache_reuse_candidates": _cache_reuse_candidates(),
                "parallel_candidates": _parallel_candidates(),
                "concurrent_requests": _parallel_target(),
            }
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        if getattr(current_fingerprint, "_mmm_stable_model_signature", False):
            performance_fingerprint._mmm_stable_model_signature = True  # type: ignore[attr-defined]
        performance_fingerprint._mmm_max_performance_fingerprint = True  # type: ignore[attr-defined]
        autotune_module._fingerprint = performance_fingerprint

        def run_variant(
            binary: str,
            model_path: str,
            config: Any,
            benchmark_request: Any,
            variant: ServerVariant,
            *,
            probe_tokens: int,
            cache_probe: bool = False,
            parallel_probe: bool = False,
            concurrency: int = 1,
        ) -> Any:
            preferred_port = autotune_module._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910)
            port = autotune_module._free_port(preferred_port)
            process = None
            try:
                process = autotune_module._start_server(
                    binary, model_path, config, variant, port
                )
                url = autotune_module._wait_ready(process, port)
                autotune_module._probe_server(
                    url,
                    benchmark_request,
                    max_tokens=1,
                    variant=variant,
                )
                if cache_probe:
                    return _cache_probe(
                        autotune_module,
                        url,
                        max_tokens=min(32, probe_tokens),
                        variant=variant,
                    )
                if parallel_probe:
                    return _parallel_probe(
                        autotune_module,
                        url,
                        benchmark_request,
                        max_tokens=probe_tokens,
                        variant=variant,
                        concurrency=concurrency,
                    )
                return autotune_module._probe_server(
                    url,
                    benchmark_request,
                    max_tokens=probe_tokens,
                    variant=variant,
                )
            except Exception as exc:
                return autotune_module.ProbeResult(
                    variant=variant,
                    ok=False,
                    output_sha256="",
                    predicted_tokens=0,
                    predicted_tps=0.0,
                    prompt_tps=0.0,
                    elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                autotune_module._stop_server(process)

        def benchmark(
            binary: str,
            model_path: str,
            config: Any,
            request: Any,
            fingerprint: str,
        ) -> Any | None:
            benchmark_request = autotune_module._compact_benchmark_request(request)
            probe_tokens = min(
                int(config.max_new_tokens),
                autotune_module._env_int(
                    "MMM_LLAMA_AUTOTUNE_TOKENS",
                    autotune_module._BENCHMARK_OUTPUT_TOKENS,
                ),
            )
            minimum_gain = autotune_module._env_float(
                "MMM_LLAMA_STAGE_MIN_GAIN",
                1.01,
            )
            all_probes: list[Any] = []

            spec_probes = [
                run_variant(
                    binary,
                    model_path,
                    config,
                    benchmark_request,
                    variant,
                    probe_tokens=probe_tokens,
                )
                for variant in candidate_variants()
            ]
            all_probes.extend(spec_probes)
            spec_selected = _select_probe(
                spec_probes,
                balanced=False,
                minimum_gain=max(
                    minimum_gain,
                    autotune_module._env_float(
                        "MMM_LLAMA_AUTOTUNE_MIN_SPEEDUP",
                        1.03,
                    ),
                ),
            )
            if spec_selected is None:
                return None
            selected = spec_selected.variant
            original_baseline = spec_probes[0]

            ubatch_values = _ubatch_candidates(autotune_module)
            ubatch_probes: list[Any] = []
            base_ubatch = min(
                autotune_module._env_int("MMM_LLAMA_BATCH", 2048),
                autotune_module._env_int("MMM_LLAMA_UBATCH", 512),
            )
            ordered_ubatches = (base_ubatch,) + tuple(
                value for value in ubatch_values if value != base_ubatch
            )
            for value in ordered_ubatches:
                variant = replace(
                    selected,
                    name=f"{selected.name}|ub{value}",
                    ubatch=value,
                )
                ubatch_probes.append(
                    run_variant(
                        binary,
                        model_path,
                        config,
                        benchmark_request,
                        variant,
                        probe_tokens=probe_tokens,
                    )
                )
            all_probes.extend(ubatch_probes)
            ubatch_selected = _select_probe(
                ubatch_probes,
                balanced=True,
                minimum_gain=minimum_gain,
            )
            if ubatch_selected is not None:
                selected = ubatch_selected.variant

            cache_probes: list[Any] = []
            for value in _cache_reuse_candidates():
                variant = replace(
                    selected,
                    name=f"{selected.name.split('|cr', 1)[0]}|cr{value}",
                    cache_reuse=value,
                )
                cache_probes.append(
                    run_variant(
                        binary,
                        model_path,
                        config,
                        benchmark_request,
                        variant,
                        probe_tokens=probe_tokens,
                        cache_probe=True,
                    )
                )
            all_probes.extend(cache_probes)
            cache_selected = _select_probe(
                cache_probes,
                balanced=True,
                minimum_gain=minimum_gain,
            )
            if cache_selected is not None:
                selected = cache_selected.variant

            concurrency = _parallel_target()
            parallel_probes: list[Any] = []
            for value in _parallel_candidates():
                variant = replace(
                    selected,
                    name=f"{selected.name.split('|p', 1)[0]}|p{value}",
                    parallel=value,
                )
                parallel_probes.append(
                    run_variant(
                        binary,
                        model_path,
                        config,
                        benchmark_request,
                        variant,
                        probe_tokens=probe_tokens,
                        parallel_probe=True,
                        concurrency=concurrency,
                    )
                )
            all_probes.extend(parallel_probes)
            parallel_selected = _select_probe(
                parallel_probes,
                balanced=False,
                minimum_gain=minimum_gain,
            )
            if parallel_selected is not None:
                selected = parallel_selected.variant
                final_probe = parallel_selected
            elif cache_selected is not None:
                final_probe = cache_selected
            elif ubatch_selected is not None:
                final_probe = ubatch_selected
            else:
                final_probe = spec_selected

            baseline_tps = float(getattr(original_baseline, "predicted_tps", 0.0))
            selected_tps = float(getattr(final_probe, "predicted_tps", 0.0))
            speedup = selected_tps / baseline_tps if baseline_tps > 0 else 1.0
            return autotune_module.AutotuneDecision(
                fingerprint=fingerprint,
                selected=selected,
                baseline_tps=baseline_tps,
                selected_tps=selected_tps,
                speedup=speedup,
                probes=tuple(all_probes),
            )

        benchmark._mmm_staged_max_performance = True  # type: ignore[attr-defined]
        autotune_module._benchmark = benchmark

        current_launch = autotune_module._launch_selected

        @wraps(current_launch)
        def launch_selected(
            binary: str,
            model_path: str,
            config: Any,
            selected: ServerVariant,
        ) -> str:
            url = current_launch(binary, model_path, config, selected)
            os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = str(max(1, selected.parallel))
            os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(
                selected.ubatch
                or min(
                    autotune_module._env_int("MMM_LLAMA_BATCH", 2048),
                    autotune_module._env_int("MMM_LLAMA_UBATCH", 512),
                )
            )
            os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = str(selected.cache_reuse)
            os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = selected.spec_type
            return url

        launch_selected._mmm_exports_active_runtime = True  # type: ignore[attr-defined]
        autotune_module._launch_selected = launch_selected
        autotune_module._mmm_max_server_performance_installed = True


__all__ = [
    "ServerVariant",
    "_cache_reuse_candidates",
    "_parallel_candidates",
    "_parallel_target",
    "_parse_int_candidates",
    "_replace_option",
    "_ubatch_candidates",
    "install",
]
