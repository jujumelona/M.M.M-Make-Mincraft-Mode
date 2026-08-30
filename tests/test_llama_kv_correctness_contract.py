from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import llama_decode_speed_contract as decode_speed


def test_precision_reference_order_prioritizes_high_precision() -> None:
    assert decode_speed._precision_reference_order(("q4_0", "f16", "q8_0")) == (
        "f16",
        "q8_0",
        "q4_0",
    )


def test_kv_schema_invalidates_pre_correctness_receipts() -> None:
    assert (
        decode_speed._KV_SCHEMA_VERSION
        == "mmm/llama-kv-decode-speed-v3-precision-reference"
    )


def test_kv_fingerprint_is_candidate_order_canonical(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test-model")
    autotune = SimpleNamespace(
        _server_version=lambda _binary: "server-v1",
        _hardware_identity=lambda: "hardware-v1",
        _env_int=lambda _name, default: default,
        _BENCHMARK_OUTPUT_TOKENS=8,
    )
    config = SimpleNamespace(
        model_id="model",
        extra={"gguf_filename": "model.gguf"},
        max_context=4096,
        max_new_tokens=8,
    )
    first = decode_speed._kv_fingerprint(
        autotune, config, "llama-server", str(model), ("q4_0", "f16", "q8_0")
    )
    second = decode_speed._kv_fingerprint(
        autotune, config, "llama-server", str(model), ("q8_0", "q4_0", "f16")
    )
    assert first == second


def test_kv_probe_uses_f16_as_semantic_reference_and_restores_env(monkeypatch) -> None:
    starts: list[str] = []

    class _Autotune:
        _BENCHMARK_OUTPUT_TOKENS = 8
        ProbeResult = SimpleNamespace

        @staticmethod
        def _compact_benchmark_request(request):
            return request

        @staticmethod
        def _env_int(_name: str, default: int) -> int:
            return default

        @staticmethod
        def _env_float(_name: str, default: float) -> float:
            return default

        @staticmethod
        def _free_port(port: int) -> int:
            return port

        @staticmethod
        def _start_server(_binary, _model_path, _config, _variant, _port):
            kv = os.environ["MMM_KV_CACHE_QUANT"]
            starts.append(kv)
            return SimpleNamespace(kv=kv)

        @staticmethod
        def _wait_ready(process, port: int) -> str:
            return f"http://127.0.0.1:{port}/{process.kv}"

        @staticmethod
        def _probe_server(_url, _request, *, max_tokens: int, variant):
            kv = os.environ["MMM_KV_CACHE_QUANT"]
            tps = {"f16": 10.0, "q8_0": 12.0, "q4_0": 14.0}[kv]
            return SimpleNamespace(
                variant=variant,
                ok=True,
                output_sha256="same-semantic-output",
                predicted_tokens=max_tokens,
                predicted_tps=tps,
                prompt_tps=100.0,
                elapsed_seconds=0.01,
                error="",
            )

        @staticmethod
        def _stop_server(_process) -> None:
            return None

    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "q4_0")
    config = SimpleNamespace(max_new_tokens=8)
    selected, probes = decode_speed._probe_kv_types(
        _Autotune(),
        "llama-server",
        "/tmp/model.gguf",
        config,
        SimpleNamespace(),
        ("q4_0", "f16", "q8_0"),
    )

    assert starts == ["f16", "q8_0", "q4_0"]
    assert [probe["kv"] for probe in probes] == starts
    assert selected == "q4_0"
    assert os.environ["MMM_KV_CACHE_QUANT"] == "q4_0"
