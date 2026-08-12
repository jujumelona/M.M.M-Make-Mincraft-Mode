from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_cache_reuse_efficiency_contract as cache_contract
from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import llama_server_hardware_policy as hardware
from minecraft_mod_ai.llama_server_runtime_tuning import ServerVariant


def _probe(*, elapsed: float, digest: str = "same", ok: bool = True):
    return autotune.ProbeResult(
        variant=ServerVariant("candidate"),
        ok=ok,
        output_sha256=digest,
        predicted_tokens=8 if ok else 0,
        predicted_tps=(8.0 / elapsed if ok and elapsed > 0 else 0.0),
        prompt_tps=100.0,
        elapsed_seconds=elapsed,
    )


def test_request_scoped_cache_reuse_tuning_is_installed() -> None:
    assert getattr(autotune, "_mmm_request_cache_reuse_tuning", False)
    assert getattr(autotune._benchmark, "_mmm_single_server_cache_stage", False)
    assert getattr(autotune._start_server, "_mmm_request_scoped_cache_reuse", False)
    assert getattr(hardware._server_payload, "_mmm_active_cache_reuse", False)


def test_cache_reuse_selection_requires_identical_output_and_measurable_gain() -> None:
    baseline = _probe(elapsed=1.0)
    different = _probe(elapsed=0.1, digest="different")
    faster = _probe(elapsed=0.5)
    selected = cache_contract._choose_cache_probe(
        [baseline, different, faster],
        minimum_gain=1.01,
    )
    assert selected is faster

    marginal = _probe(elapsed=0.995)
    selected = cache_contract._choose_cache_probe(
        [baseline, marginal],
        minimum_gain=1.01,
    )
    assert selected is baseline


def test_active_cache_reuse_is_forwarded_per_request(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_CACHE_REUSE", "256")
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=16))
    request = SimpleNamespace(
        messages=({"role": "user", "content": "x"},),
        response_format="text",
    )
    payload = hardware._server_payload(adapter, request)
    assert payload["cache_prompt"] is True
    assert payload["n_cache_reuse"] == 256
