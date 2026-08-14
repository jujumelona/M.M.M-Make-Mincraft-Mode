from __future__ import annotations

import json
import inspect
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import llama_server_kernel_autotune as kernel_autotune
from minecraft_mod_ai import llama_server_runtime_tuning as runtime
from minecraft_mod_ai import long_run_resilience_contract as resilience


_GIB = 1024**3


def _config(*, context: int = 32768):
    return SimpleNamespace(
        model_id="example/model-GGUF",
        extra={"gguf_filename": "model.gguf"},
        max_context=context,
        max_new_tokens=8192,
    )


def _clear_parallel_env(monkeypatch) -> None:
    for name in (
        "MMM_PERFORMANCE_MODE",
        "MMM_LLAMA_TUNING_OBJECTIVE",
        "MMM_LLAMA_PARALLEL",
        "MMM_LLAMA_CONCURRENT_REQUESTS",
        "MMM_LLAMA_SERVER_CTX",
        "MMM_LLAMA_GPU_FREE_MIB",
        "MMM_LLAMA_GPU_TOTAL_MIB",
        "MMM_LLAMA_RAM_AVAILABLE_MIB",
        "MMM_LLAMA_KV_BYTES_PER_TOKEN",
        "MMM_LLAMA_RUNTIME_RECEIPT",
        "MMM_LLAMA_ACTIVE_PARALLEL",
        "MMM_LLAMA_ACTIVE_UBATCH",
        "MMM_LLAMA_ACTIVE_CACHE_REUSE",
        "MMM_LLAMA_ACTIVE_SPEC_TYPE",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_K",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_V",
    ):
        monkeypatch.delenv(name, raising=False)


def test_auto_mode_probes_p1_p2_p4_only_when_resources_fit(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setattr(runtime, "_model_size", lambda _path: 6 * _GIB)
    fit = runtime.RuntimeResources(15 * _GIB, 15 * _GIB, 8 * _GIB, 2)
    tighter = runtime.RuntimeResources(10 * _GIB, 15 * _GIB, 8 * _GIB, 2)

    assert runtime._performance_mode() == "auto"
    assert runtime._parallel_candidates(_config(), "model.gguf", fit) == (1, 2, 4)
    assert runtime._parallel_candidates(_config(), "model.gguf", tighter) == (1, 2)
    assert runtime._parallel_candidates(
        _config(), "model.gguf", runtime.RuntimeResources()
    ) == (1,)


def test_operator_modes_and_exact_parallel_take_priority(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    resources = runtime.RuntimeResources()
    monkeypatch.setenv("MMM_PERFORMANCE_MODE", "latency")
    monkeypatch.setenv("MMM_LLAMA_TUNING_OBJECTIVE", "throughput")
    assert runtime._performance_mode() == "latency"
    assert runtime._parallel_candidates(_config(), "missing.gguf", resources) == (1,)

    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "4")
    assert runtime._parallel_candidates(_config(), "missing.gguf", resources) == (4,)


def test_explicit_parallel_uses_one_canonical_value_everywhere(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "+4")
    assert runtime._explicit_parallel() == 4
    assert runtime._parallel_candidates() == (4,)

    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "9")
    assert runtime._explicit_parallel() == 8
    assert runtime._parallel_candidates() == (8,)

    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "many")
    with pytest.raises(ValueError, match="must be an integer"):
        runtime._parallel_candidates()


def test_live_zero_resource_measurement_does_not_use_stale_receipt(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv(
        "MMM_COLAB_SETUP_RECEIPT",
        json.dumps(
            {
                "gpu_free_mib": 14_000,
                "gpu_total_mib": 15_360,
                "ram_available_mib": 10_000,
            }
        ),
    )
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, 15360\n"),
    )
    monkeypatch.setattr(
        runtime.Path,
        "read_text",
        lambda _self, encoding=None: "MemAvailable: 0 kB\n",
    )

    resources = runtime._runtime_resources()

    assert resources.gpu_free_bytes == 0
    assert resources.gpu_total_bytes == 15_360 * 1024 * 1024
    assert resources.ram_available_bytes == 0


def test_native_start_scales_total_context_per_slot_and_bounds_cache_ram(
    monkeypatch,
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.delenv("MMM_LLAMA_CACHE_RAM_MIB", raising=False)
    seen: dict[str, list[str]] = {}

    def popen(args, **_kwargs):
        seen["args"] = list(args)
        return SimpleNamespace()

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    autotune._start_server(
        "llama-server",
        "model.gguf",
        _config(),
        runtime.ServerVariant("baseline|p4", parallel=4),
        8910,
    )
    args = seen["args"]
    assert args[args.index("--parallel") + 1] == "4"
    assert args[args.index("--ctx-size") + 1] == str(32768 * 4)
    assert args[args.index("--cache-ram") + 1] == "1024"
    assert "--cont-batching" in args
    assert "--kv-unified" in args


def test_qwen_hotpath_does_not_reenable_or_reserve_prompt_cache(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    seen: dict[str, list[str]] = {}

    def popen(args, **_kwargs):
        seen["args"] = list(args)
        return SimpleNamespace()

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    config = SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        max_context=32768,
        max_new_tokens=8192,
    )
    autotune._start_server(
        "llama-server",
        "model.gguf",
        config,
        runtime.ServerVariant("baseline"),
        8910,
    )
    assert "--cache-prompt" not in seen["args"]
    assert "--cache-ram" not in seen["args"]


def test_parallel_context_fails_closed_when_total_overflows(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must reject before process launch")
        ),
    )
    with pytest.raises(RuntimeError, match="exceeds supported maximum"):
        autotune._start_server(
            "llama-server",
            "model.gguf",
            _config(context=600_000_000),
            runtime.ServerVariant("baseline|p4", parallel=4),
            8910,
        )


def test_launch_receipt_reports_actual_downgraded_runtime(monkeypatch, tmp_path) -> None:
    _clear_parallel_env(monkeypatch)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(
        runtime,
        "_runtime_resources",
        lambda: runtime.RuntimeResources(15 * _GIB, 15 * _GIB, 8 * _GIB, 2),
    )
    attempts: list[int] = []

    def launch(_binary, _model, _config_value, selected):
        attempts.append(selected.parallel)
        if selected.parallel == 4:
            raise RuntimeError("synthetic p4 allocation failure")
        return "http://127.0.0.1:8910/v1"

    fake = SimpleNamespace(
        _base_args=lambda *_args: ["llama-server", "--ctx-size", "32768"],
        _variant_args=lambda _variant: [],
        _fingerprint=lambda *_args: "base",
        _benchmark=lambda *_args: None,
        _launch_selected=launch,
        ensure_tuned_server=lambda _config_value, _request: "http://127.0.0.1:8910/v1",
        _shutdown_managed_server=lambda: None,
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
        _env_bool=lambda _name, default: default,
        _env_int=lambda _name, default: default,
    )
    runtime.install(fake)
    url = fake._launch_selected(
        "llama-server",
        str(model),
        _config(),
        runtime.ServerVariant("mtp-2|p4", "draft-mtp", 2, ubatch=512, parallel=4),
    )

    assert url.endswith("/v1")
    assert attempts == [4, 2]
    receipt = json.loads(runtime.os.environ["MMM_LLAMA_RUNTIME_RECEIPT"])
    assert receipt["slots"] == 2
    assert receipt["context_per_slot"] == 32768
    assert receipt["context_total"] == 65536
    assert receipt["ubatch"] == 512
    assert receipt["spec_type"] == "draft-mtp"
    assert receipt["draft_n_max"] == 2
    assert len(receipt["selection_sha256"]) == 64


def test_parallel_probe_includes_short_and_medium_prefill(monkeypatch) -> None:
    lengths: list[int] = []

    def probe(_url, request, *, max_tokens, variant):
        lengths.append(sum(len(str(item.get("content", ""))) for item in request.messages))
        return autotune.ProbeResult(
            variant=variant,
            ok=True,
            output_sha256="same",
            predicted_tokens=max_tokens,
            predicted_tps=10.0,
            prompt_tps=20.0,
            elapsed_seconds=0.01,
        )

    fake = SimpleNamespace(_probe_server=probe, ProbeResult=autotune.ProbeResult)
    request = SimpleNamespace(
        messages=({"role": "user", "content": "short"},), response_format="text"
    )
    result = runtime._parallel_probe(
        fake,
        "http://127.0.0.1:8910/v1",
        request,
        max_tokens=8,
        variant=runtime.ServerVariant("baseline|p2", parallel=2),
        concurrency=2,
    )
    assert result.ok is True
    assert len(lengths) == 4
    assert max(lengths) > min(lengths) + 12_000


def test_parallel_stage_compares_same_e2e_metric_not_legacy_decode_tps() -> None:
    legacy_decode_only_p1 = SimpleNamespace(predicted_tps=50.0)
    rates = {1: 16.67, 2: 28.57, 4: 20.0}
    calls: list[int] = []

    def run_variant(
        _binary,
        _model_path,
        _config_value,
        _request,
        variant,
        *,
        probe_tokens,
        parallel_probe,
        concurrency,
        propagate_resource_failure=False,
    ):
        assert parallel_probe is True
        assert propagate_resource_failure is (concurrency == 1)
        calls.append(concurrency)
        return autotune.ProbeResult(
            variant=variant,
            ok=True,
            output_sha256="short+medium",
            predicted_tokens=probe_tokens * concurrency,
            predicted_tps=rates[concurrency],
            prompt_tps=20.0,
            elapsed_seconds=1.0,
        )

    selected, winner, p1, measured = runtime._run_parallel_stage(
        run_variant,
        binary="server",
        model_path="model.gguf",
        config=_config(),
        benchmark_request=object(),
        selected=runtime.ServerVariant("baseline"),
        probe_tokens=64,
        parallel_values=(1, 2, 4),
        minimum_gain=1.01,
    )
    assert p1.predicted_tps < legacy_decode_only_p1.predicted_tps
    assert calls == [1, 2, 4]
    assert len(measured) == 3
    assert winner.predicted_tps == 28.57
    assert selected.parallel == 2


def test_parallel_selection_chooses_global_best_independent_of_probe_order() -> None:
    def result(slots, rate):
        return autotune.ProbeResult(
            variant=runtime.ServerVariant(f"p{slots}", parallel=slots),
            ok=True,
            output_sha256="same",
            predicted_tokens=100,
            predicted_tps=rate,
            prompt_tps=20.0,
            elapsed_seconds=1.0,
        )

    p1, p2, p4 = result(1, 100.0), result(2, 101.1), result(4, 102.0)
    assert runtime._select_parallel_probe(p1, (p2, p4), minimum_gain=1.01) is p4
    assert runtime._select_parallel_probe(p1, (p4, p2), minimum_gain=1.01) is p4


def test_parallel_metric_hash_covers_both_short_and_medium_outputs() -> None:
    def probe(_url, request, *, max_tokens, variant):
        medium = sum(len(str(item.get("content", ""))) for item in request.messages) > 1000
        suffix = "different" if medium and variant.parallel == 2 else "same"
        return autotune.ProbeResult(
            variant=variant,
            ok=True,
            output_sha256=suffix,
            predicted_tokens=max_tokens,
            predicted_tps=10.0,
            prompt_tps=20.0,
            elapsed_seconds=0.01,
        )

    fake = SimpleNamespace(_probe_server=probe, ProbeResult=autotune.ProbeResult)
    request = SimpleNamespace(
        messages=({"role": "user", "content": "short"},), response_format="text"
    )
    p1 = runtime._parallel_probe(
        fake,
        "http://127.0.0.1:8910/v1",
        request,
        max_tokens=8,
        variant=runtime.ServerVariant("baseline|p1", parallel=1),
        concurrency=1,
    )
    p2 = runtime._parallel_probe(
        fake,
        "http://127.0.0.1:8910/v1",
        request,
        max_tokens=8,
        variant=runtime.ServerVariant("baseline|p2", parallel=2),
        concurrency=2,
    )
    assert p1.output_sha256 != p2.output_sha256
    assert runtime._select_parallel_probe(p1, (p2,), minimum_gain=1.0) is p1


@pytest.mark.parametrize(
    ("message", "recoverable"),
    (("CUDA out of memory", True), ("invalid GGUF magic", False)),
)
def test_actual_launch_classifier_only_retries_resource_failures(
    monkeypatch, message, recoverable
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "4")
    monkeypatch.setattr(runtime, "_model_size", lambda _path: 6 * _GIB)
    monkeypatch.setattr(
        runtime,
        "_runtime_resources",
        lambda: runtime.RuntimeResources(8 * _GIB, 15 * _GIB, 8 * _GIB, 2),
    )
    fake = SimpleNamespace(
        _base_args=lambda *_args: ["llama-server", "--ctx-size", "32768"],
        _variant_args=lambda _variant: [],
        _fingerprint=lambda *_args: "base",
        _benchmark=lambda *_args: None,
        _launch_selected=lambda *_args: (_ for _ in ()).throw(RuntimeError(message)),
        ensure_tuned_server=lambda *_args: "unused",
        _shutdown_managed_server=lambda: None,
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
        _env_bool=lambda _name, default: default,
        _env_int=lambda _name, default: default,
    )
    runtime.install(fake)
    expected = runtime.RecoverableResourceLaunchError if recoverable else RuntimeError
    with pytest.raises(expected) as raised:
        fake._launch_selected(
            "llama-server",
            "model.gguf",
            _config(),
            runtime.ServerVariant("baseline|p4", parallel=4),
        )
    assert isinstance(raised.value, runtime.RecoverableResourceLaunchError) is recoverable


def test_startup_log_capture_is_bounded_and_redacts_tokens() -> None:
    process = SimpleNamespace(
        stderr=io.BytesIO(
            (b"x" * 70_000)
            + b"\nCUDA out of memory\ntoken=supersecretvalue\n"
            + b"Authorization: Bearer TOPSECRET\npassword=PASSWORDSECRET\n"
        ),
        poll=lambda: 1,
    )
    runtime._attach_startup_log(process)
    tail = runtime._startup_log_tail(process)
    assert "resource_marker=out_of_memory" in tail
    assert "stderr_sha256=" in tail
    assert "supersecretvalue" not in tail
    assert "TOPSECRET" not in tail
    assert "PASSWORDSECRET" not in tail
    assert "Authorization" not in tail
    assert "password" not in tail
    assert len(tail) < 256


def test_empty_stderr_sigkill_is_still_a_recoverable_resource_exit(
    monkeypatch,
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setattr(runtime, "_model_size", lambda _path: 6 * _GIB)
    resources = runtime.RuntimeResources(7 * _GIB, 15 * _GIB, 8 * _GIB, 2)
    assert runtime._recoverable_resource_failure(
        ("RuntimeError: llama-server exited with code 137.",),
        slots=1,
        config=_config(),
        model_path="model.gguf",
        resources=resources,
    )


def test_exact_parallel_fails_closed_when_reference_or_target_cannot_run() -> None:
    def result(variant, *, ok=True):
        return autotune.ProbeResult(
            variant=variant,
            ok=ok,
            output_sha256="same" if ok else "",
            predicted_tokens=64 if ok else 0,
            predicted_tps=10.0 if ok else 0.0,
            prompt_tps=20.0 if ok else 0.0,
            elapsed_seconds=1.0,
        )

    def failed_reference(
        _binary,
        _model,
        _config_value,
        _request,
        variant,
        **_kwargs,
    ):
        return result(variant, ok=False)

    with pytest.raises(RuntimeError, match="could not validate the p1"):
        runtime._run_parallel_stage(
            failed_reference,
            binary="server",
            model_path="model.gguf",
            config=_config(),
            benchmark_request=object(),
            selected=runtime.ServerVariant("baseline"),
            probe_tokens=64,
            parallel_values=(4,),
            minimum_gain=1.01,
            forced_parallel=4,
        )

    def target_oom(
        _binary,
        _model,
        _config_value,
        _request,
        variant,
        *,
        concurrency,
        propagate_resource_failure=False,
        **_kwargs,
    ):
        if concurrency == 4:
            assert propagate_resource_failure is True
            raise runtime.RecoverableResourceLaunchError("resource_marker=out_of_memory")
        return result(variant)

    with pytest.raises(runtime.RecoverableResourceLaunchError):
        runtime._run_parallel_stage(
            target_oom,
            binary="server",
            model_path="model.gguf",
            config=_config(),
            benchmark_request=object(),
            selected=runtime.ServerVariant("baseline"),
            probe_tokens=64,
            parallel_values=(4,),
            minimum_gain=1.01,
            forced_parallel=4,
        )


def test_baseline_tuning_oom_releases_attempt_key(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    base_ensure = inspect.unwrap(autotune.ensure_tuned_server)
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.setattr(autotune, "_external_server_is_ready", lambda: False)
    monkeypatch.setattr(autotune, "_server_binary", lambda: "llama-server")
    monkeypatch.setattr(autotune, "_resolve_model_path", lambda _config: str(model))
    monkeypatch.setattr(autotune, "_fingerprint", lambda *_args: "oom-fingerprint")
    monkeypatch.setattr(autotune, "_load_cached_decision", lambda _key: None)
    monkeypatch.setattr(
        autotune,
        "_benchmark",
        lambda *_args: (_ for _ in ()).throw(
            runtime.RecoverableResourceLaunchError("resource_marker=out_of_memory")
        ),
    )
    monkeypatch.setattr(autotune, "_ATTEMPTED_KEYS", set())
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", None)
    monkeypatch.setattr(autotune, "_MANAGED_URL", None)

    with pytest.raises(runtime.RecoverableResourceLaunchError):
        base_ensure(_config(), object())
    assert "oom-fingerprint" not in autotune._ATTEMPTED_KEYS


def test_outer_kernel_baseline_preserves_typed_resource_failure() -> None:
    def run_variant(*_args, propagate_resource_failure=False, **_kwargs):
        assert propagate_resource_failure is True
        raise runtime.RecoverableResourceLaunchError(
            "resource_marker=out_of_memory"
        )

    run_variant._mmm_resource_failure_propagation = True
    fake = SimpleNamespace(
        _mmm_run_tuning_variant=run_variant,
        _compact_benchmark_request=lambda value: value,
        _env_int=lambda _name, default: default,
        _env_float=lambda _name, default: default,
        _BENCHMARK_OUTPUT_TOKENS=64,
        ServerVariant=runtime.ServerVariant,
    )
    with pytest.raises(runtime.RecoverableResourceLaunchError):
        kernel_autotune._benchmark(
            fake,
            "server",
            "model.gguf",
            _config(),
            object(),
            "Tesla T4",
        )


def test_managed_server_is_restarted_when_performance_mode_changes(
    monkeypatch, tmp_path
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(
        runtime,
        "_runtime_resources",
        lambda: runtime.RuntimeResources(15 * _GIB, 15 * _GIB, 8 * _GIB, 2),
    )

    class Alive:
        @staticmethod
        def poll():
            return None

    calls = {"ensure": 0, "shutdown": 0}
    fake = SimpleNamespace(
        _base_args=lambda *_args: ["llama-server", "--ctx-size", "32768"],
        _variant_args=lambda _variant: [],
        _fingerprint=lambda *_args: "base",
        _benchmark=lambda *_args: None,
        _launch_selected=lambda *_args: "http://127.0.0.1:8910/v1",
        _env_bool=lambda _name, default: default,
        _env_int=lambda _name, default: default,
        _AUTOTUNE_LOCK=threading.RLock(),
        _MANAGED_PROCESS=Alive(),
        _MANAGED_URL="http://127.0.0.1:8910/v1",
    )

    def ensure(_config_value, _request):
        calls["ensure"] += 1
        return "http://127.0.0.1:8911/v1"

    def shutdown():
        calls["shutdown"] += 1
        fake._MANAGED_PROCESS = None
        fake._MANAGED_URL = None

    fake.ensure_tuned_server = ensure
    fake._shutdown_managed_server = shutdown
    runtime.install(fake)
    fake._launch_selected(
        "llama-server",
        str(model),
        _config(),
        runtime.ServerVariant("baseline|p2", parallel=2),
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", fake._MANAGED_URL)
    monkeypatch.setenv("MMM_PERFORMANCE_MODE", "latency")

    assert fake.ensure_tuned_server(_config(), object()).endswith("8911/v1")
    assert calls == {"ensure": 1, "shutdown": 1}


def test_concurrent_mode_change_reuses_first_matching_replacement(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)

    class Process:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

    old_process = Process()
    replacement: list[Process] = []
    counts = {"launch": 0, "shutdown": 0}
    config = _config()
    old_url = "http://127.0.0.1:8910/v1"
    monkeypatch.setenv("LLAMA_SERVER_URL", old_url)
    auto_receipt = {
        "selection_inputs_sha256": runtime._json_fingerprint(
            runtime._selection_inputs(config)
        )
    }
    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", json.dumps(auto_receipt))

    fake = SimpleNamespace(
        _base_args=lambda *_args: ["llama-server", "--ctx-size", "32768"],
        _variant_args=lambda _variant: [],
        _fingerprint=lambda *_args: "base",
        _benchmark=lambda *_args: None,
        _launch_selected=lambda *_args: old_url,
        _env_bool=lambda _name, default: default,
        _env_int=lambda _name, default: default,
        _AUTOTUNE_LOCK=threading.RLock(),
        _MANAGED_PROCESS=old_process,
        _MANAGED_KEY="auto-key",
        _MANAGED_URL=old_url,
        _MMM_LLAMA_RUNTIME_RECEIPT=auto_receipt,
    )

    def ensure(_config_value, _request):
        if fake._MANAGED_PROCESS is None:
            counts["launch"] += 1
            process = Process()
            replacement.append(process)
            fake._MANAGED_PROCESS = process
            fake._MANAGED_KEY = "latency-key"
            fake._MANAGED_URL = old_url
            os.environ["LLAMA_SERVER_URL"] = old_url
            receipt = {
                "selection_inputs_sha256": runtime._json_fingerprint(
                    runtime._selection_inputs(config)
                )
            }
            fake._MMM_LLAMA_RUNTIME_RECEIPT = receipt
            os.environ["MMM_LLAMA_RUNTIME_RECEIPT"] = json.dumps(receipt)
        return old_url

    def shutdown():
        counts["shutdown"] += 1
        fake._MANAGED_PROCESS.terminate()
        fake._MANAGED_PROCESS = None
        fake._MANAGED_KEY = None
        fake._MANAGED_URL = None

    fake.ensure_tuned_server = ensure
    fake._shutdown_managed_server = shutdown
    runtime.install(fake)
    monkeypatch.setenv("MMM_PERFORMANCE_MODE", "latency")
    ready = threading.Barrier(2)

    def call(index):
        ready.wait(timeout=5)
        return fake.ensure_tuned_server(config, index)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(call, (1, 2)))

    assert results == [old_url, old_url]
    assert counts == {"launch": 1, "shutdown": 1}
    assert old_process.terminated is True
    assert len(replacement) == 1
    assert replacement[0].terminated is False
    assert fake._MANAGED_PROCESS is replacement[0]


def test_resource_failure_is_retryable_but_corrupt_launch_remains_poisoned() -> None:
    recoverable_key = "recoverable"
    corrupt_key = "corrupt"
    autotune._ATTEMPTED_KEYS.update({recoverable_key, corrupt_key})
    try:
        autotune._release_recoverable_attempt(
            recoverable_key, runtime.RecoverableResourceLaunchError("CUDA out of memory")
        )
        autotune._release_recoverable_attempt(corrupt_key, RuntimeError("invalid GGUF"))
        assert recoverable_key not in autotune._ATTEMPTED_KEYS
        assert corrupt_key in autotune._ATTEMPTED_KEYS
    finally:
        autotune._ATTEMPTED_KEYS.discard(recoverable_key)
        autotune._ATTEMPTED_KEYS.discard(corrupt_key)


def test_managed_key_allows_crash_rearm_and_auto_latency_auto_toggle(
    monkeypatch, tmp_path
) -> None:
    _clear_parallel_env(monkeypatch)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    class Process:
        def __init__(self):
            self.alive = True

        def poll(self):
            return None if self.alive else 1

        def terminate(self):
            self.alive = False

        def kill(self):
            self.alive = False

        def wait(self, timeout=None):
            del timeout
            return 0

    launches: list[str] = []
    fingerprint = {"value": "auto-fingerprint"}

    def launch(_binary, _model, _config_value, _selected):
        process = Process()
        autotune._MANAGED_PROCESS = process
        autotune._MANAGED_URL = "http://127.0.0.1:8910/v1"
        launches.append(fingerprint["value"])
        return autotune._MANAGED_URL

    base_ensure = inspect.unwrap(autotune.ensure_tuned_server)
    monkeypatch.setattr(autotune, "_external_server_is_ready", lambda: False)
    monkeypatch.setattr(autotune, "_server_binary", lambda: "llama-server")
    monkeypatch.setattr(autotune, "_resolve_model_path", lambda _config_value: str(model))
    monkeypatch.setattr(
        autotune, "_fingerprint", lambda *_args: fingerprint["value"]
    )
    monkeypatch.setattr(
        autotune,
        "_load_cached_decision",
        lambda value: autotune._baseline_decision(value),
    )
    monkeypatch.setattr(autotune, "_launch_selected", launch)
    monkeypatch.setattr(autotune, "_ATTEMPTED_KEYS", set())
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", None)
    monkeypatch.setattr(autotune, "_MANAGED_URL", None)
    monkeypatch.setattr(autotune, "_MANAGED_KEY", None)

    assert base_ensure(_config(), object()).endswith("/v1")
    assert autotune._MANAGED_KEY == "auto-fingerprint"
    assert "auto-fingerprint" in autotune._ATTEMPTED_KEYS

    autotune._MANAGED_PROCESS.alive = False
    assert resilience._rearm_managed_server(autotune, force=False) is True
    assert autotune._MANAGED_KEY is None
    assert "auto-fingerprint" not in autotune._ATTEMPTED_KEYS
    assert base_ensure(_config(), object()).endswith("/v1")

    autotune._shutdown_managed_server()
    fingerprint["value"] = "latency-fingerprint"
    assert base_ensure(_config(), object()).endswith("/v1")
    autotune._shutdown_managed_server()
    fingerprint["value"] = "auto-fingerprint"
    assert base_ensure(_config(), object()).endswith("/v1")
    assert launches == [
        "auto-fingerprint",
        "auto-fingerprint",
        "latency-fingerprint",
        "auto-fingerprint",
    ]
    autotune._shutdown_managed_server()
