from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_kernel_autotune as kernel


def test_kernel_probe_failure_uses_unmodified_canonical_server(monkeypatch) -> None:
    for name in (
        "MMM_LLAMA_KERNEL_BYPASS",
        "MMM_LLAMA_KERNEL_AUTOTUNE",
        "MMM_LLAMA_BATCH",
        "MMM_KV_CACHE_QUANT",
        "MMM_LLAMA_ACTIVE_FLASH_ATTN",
        "MMM_LLAMA_ACTIVE_BATCH",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_K",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_V",
        "MMM_LLAMA_ACTIVE_KV_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)

    observed_args: list[list[str]] = []
    holder: dict[str, object] = {}

    def base_args(_binary, _model_path, _config, _port):
        return [
            "llama-server",
            "--batch-size",
            "512",
            "--ubatch-size",
            "256",
        ]

    def canonical_ensure(_config, _request):
        autotune = holder["autotune"]
        observed_args.append(
            autotune._base_args("llama-server", "/tmp/model.gguf", object(), 8910)
        )
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        _base_args=base_args,
        _fingerprint=lambda *_args: "base",
        ensure_tuned_server=canonical_ensure,
        _hardware_identity=lambda: "NVIDIA Tesla T4",
        _env_int=lambda name, default: int(os.environ.get(name, default)),
        _env_bool=lambda name, default: (
            os.environ.get(name, "1" if default else "0").strip() == "1"
        ),
        _external_server_is_ready=lambda: False,
        _server_binary=lambda: "llama-server",
        _resolve_model_path=lambda _config: "/tmp/model.gguf",
        _MANAGED_PROCESS=None,
        _MANAGED_URL="",
    )
    holder["autotune"] = autotune
    runtime = SimpleNamespace(_ubatch_candidates=lambda _autotune: (256,))

    monkeypatch.setattr(kernel, "_fingerprint", lambda *_args: "kernel-fp")
    monkeypatch.setattr(kernel, "_load", lambda *_args: None)

    def fail_benchmark(*_args, **_kwargs):
        raise RuntimeError("baseline llama kernel configuration failed: synthetic")

    monkeypatch.setattr(kernel, "_benchmark", fail_benchmark)
    kernel.install(autotune, runtime)

    result = autotune.ensure_tuned_server(object(), object())

    assert result == "http://127.0.0.1:8910/v1"
    assert observed_args == [
        [
            "llama-server",
            "--batch-size",
            "512",
            "--ubatch-size",
            "256",
        ]
    ]
    assert "MMM_LLAMA_KERNEL_BYPASS" not in os.environ


def test_failed_kernel_baseline_error_keeps_probe_reason(monkeypatch) -> None:
    failed = kernel.KernelProbe(
        config=kernel.KernelConfig("on", 2048, "q4_0", "q4_0"),
        ok=False,
        output_sha256="",
        predicted_tps=0.0,
        prompt_tps=0.0,
        elapsed_seconds=0.1,
        error="server exited before health check",
    )
    monkeypatch.setattr(kernel, "_baseline_config", lambda: failed.config)
    monkeypatch.setattr(kernel, "_run_probe", lambda *_args, **_kwargs: failed)
    autotune = SimpleNamespace(_env_float=lambda _name, default: default)

    try:
        kernel._benchmark(
            autotune,
            "llama-server",
            "/tmp/model.gguf",
            object(),
            object(),
            "NVIDIA Tesla T4",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected failed kernel baseline")

    assert "fa=on batch=2048 kv=q4_0/q4_0" in message
    assert "server exited before health check" in message
