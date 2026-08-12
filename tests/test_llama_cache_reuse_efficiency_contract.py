from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import minecraft_mod_ai.llama_server_runtime_tuning as runtime_tuning
from minecraft_mod_ai import llama_cache_reuse_efficiency_contract as contract


@dataclass(frozen=True)
class _Probe:
    variant: runtime_tuning.ServerVariant
    ok: bool = True
    output_sha256: str = "same"
    predicted_tokens: int = 4
    predicted_tps: float = 1.0
    prompt_tps: float = 1.0
    elapsed_seconds: float = 1.0
    error: str = ""


@dataclass(frozen=True)
class _Decision:
    fingerprint: str
    selected: runtime_tuning.ServerVariant
    baseline_tps: float
    selected_tps: float
    speedup: float
    probes: tuple[_Probe, ...]


class _Process:
    pass


def test_cache_reuse_candidates_share_one_loaded_server(monkeypatch) -> None:
    benchmark_seen_candidates = []
    starts = []
    stops = []

    def base_benchmark(_binary, _model, _config, _request, fingerprint):
        benchmark_seen_candidates.append(tuple(runtime_tuning._cache_reuse_candidates()))
        selected = runtime_tuning.ServerVariant(
            "mtp-2",
            "draft-mtp",
            2,
            ubatch=512,
        )
        return _Decision(
            fingerprint=fingerprint,
            selected=selected,
            baseline_tps=10.0,
            selected_tps=12.0,
            speedup=1.2,
            probes=(),
        )

    def start(_binary, _model, _config, variant, _port):
        starts.append(variant)
        return _Process()

    autotune = SimpleNamespace(
        _start_server=start,
        _benchmark=base_benchmark,
        _env_int=lambda _name, default, **_kwargs: default,
        _env_float=lambda _name, default, **_kwargs: default,
        _free_port=lambda preferred: preferred,
        _wait_ready=lambda _process, port: f"http://127.0.0.1:{port}/v1",
        _stop_server=lambda process: stops.append(process),
        _probe_server=lambda _url, _request, *, max_tokens, variant: _Probe(variant),
        _compact_benchmark_request=lambda request: request,
        _BENCHMARK_OUTPUT_TOKENS=96,
    )
    hardware = SimpleNamespace(
        _server_payload=lambda _adapter, _request: {"cache_prompt": True}
    )

    monkeypatch.setattr(
        runtime_tuning,
        "_cache_reuse_candidates",
        lambda: (0, 64, 256),
    )

    def fake_probe(_autotune, _url, *, cache_reuse, variant, max_tokens):
        elapsed = {0: 1.0, 64: 0.5, 256: 0.8}[cache_reuse]
        return _Probe(variant=variant, elapsed_seconds=elapsed)

    monkeypatch.setattr(contract, "_probe_request_cache_reuse", fake_probe)
    contract.install(autotune, hardware, runtime_tuning)

    decision = autotune._benchmark(
        "server",
        "model.gguf",
        SimpleNamespace(max_new_tokens=8192),
        SimpleNamespace(),
        "fp",
    )

    # The original runtime benchmark sees its internal cache-reuse stage disabled,
    # while this contract keeps the same candidates and measures them on one server.
    assert benchmark_seen_candidates == [()]
    assert len(starts) == 1
    assert len(stops) == 1
    assert starts[0].cache_reuse == 0
    assert decision.selected.cache_reuse == 64
    assert len(decision.probes) == 3

    monkeypatch.setenv("MMM_LLAMA_ACTIVE_CACHE_REUSE", "64")
    payload = hardware._server_payload(SimpleNamespace(), SimpleNamespace())
    assert payload["n_cache_reuse"] == 64
