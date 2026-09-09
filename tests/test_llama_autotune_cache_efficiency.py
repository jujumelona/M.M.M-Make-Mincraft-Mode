from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import parallel_runtime_contract as parallel_runtime


def _decision(fingerprint: str, name: str):
    variant = autotune.ServerVariant(name)
    probe = autotune.ProbeResult(
        variant=variant,
        ok=True,
        output_sha256="same",
        predicted_tokens=8,
        predicted_tps=8.0,
        prompt_tps=16.0,
        elapsed_seconds=1.0,
    )
    return autotune.AutotuneDecision(
        fingerprint=fingerprint,
        selected=variant,
        baseline_tps=8.0,
        selected_tps=8.0,
        speedup=1.0,
        probes=(probe,),
    )


def test_autotune_cache_keeps_multiple_model_fingerprints(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "autotune.json"
    monkeypatch.setenv("MMM_LLAMA_AUTOTUNE_CACHE", str(cache))

    autotune._save_decision(_decision("model-a", "baseline"))
    autotune._save_decision(_decision("model-b", "baseline"))

    first = autotune._load_cached_decision("model-a")
    second = autotune._load_cached_decision("model-b")
    assert first is not None and first.fingerprint == "model-a"
    assert second is not None and second.fingerprint == "model-b"

    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["store_schema"] == "mmm/llama-server-autotune-store-v1"
    assert set(payload["entries"]) == {"model-a", "model-b"}


def test_autotune_model_resolution_delegates_to_process_prefetch(monkeypatch) -> None:
    captured = {}

    def resolve_model_path(config, resolver):
        captured["config"] = config
        captured["resolver"] = resolver
        return "/tmp/model.gguf"

    monkeypatch.setattr(parallel_runtime, "resolve_model_path", resolve_model_path)
    config = SimpleNamespace(model_id="repo/model", extra={})

    assert Path(autotune._resolve_model_path(config)) == Path("/tmp/model.gguf").resolve()
    assert captured["config"] is config
    assert captured["resolver"] is autotune._resolve_model_path_direct
    assert isinstance(parallel_runtime._PREFETCH_FUTURES, dict)
    assert getattr(autotune._server_version, "_mmm_process_metadata_cache", False)
    assert getattr(autotune._hardware_identity, "_mmm_process_metadata_cache", False)
    assert getattr(autotune._load_cached_decision, "_mmm_multi_decision_store", False)
    assert getattr(autotune._save_decision, "_mmm_multi_decision_store", False)


def _base_ensure_tuned_server():
    from inspect import unwrap
    return unwrap(autotune.ensure_tuned_server)


def _stub_cold_runtime(monkeypatch, fingerprint="cold-fingerprint"):
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", None)
    monkeypatch.setattr(autotune, "_MANAGED_URL", None)
    monkeypatch.setattr(autotune, "_MANAGED_KEY", None)
    autotune._ATTEMPTED_KEYS.clear()
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_AUTOTUNE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_AUTOTUNE_INLINE", raising=False)
    monkeypatch.setattr(autotune, "_external_server_is_ready", lambda: False)
    monkeypatch.setattr(autotune, "_server_binary", lambda: "/tmp/llama-server")
    monkeypatch.setattr(autotune, "_resolve_model_path", lambda _config: "/tmp/model.gguf")
    monkeypatch.setattr(autotune, "_fingerprint", lambda _config, _binary, _path: fingerprint)
    return SimpleNamespace(model_id="repo/model", extra={}, max_context=8192)


def test_cold_request_cache_miss_does_not_benchmark(monkeypatch):
    config = _stub_cold_runtime(monkeypatch)
    monkeypatch.setattr(autotune, "_load_cached_decision", lambda _fp: None)
    monkeypatch.setattr(autotune, "_benchmark", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cold request benchmarked")))
    launched = {}
    monkeypatch.setattr(autotune, "_launch_selected", lambda _b, _p, _c, selected: launched.setdefault("selected", selected) and "http://127.0.0.1:8910/v1")
    assert _base_ensure_tuned_server()(config, object()).endswith("/v1")
    assert launched["selected"].name == "baseline"


def test_cold_request_uses_cached_validated_winner(monkeypatch):
    config = _stub_cold_runtime(monkeypatch, "cached")
    cached = _decision("cached", "mtp-3")
    monkeypatch.setattr(autotune, "_load_cached_decision", lambda _fp: cached)
    monkeypatch.setattr(autotune, "_benchmark", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cache hit benchmarked")))
    launched = {}
    monkeypatch.setattr(autotune, "_launch_selected", lambda _b, _p, _c, selected: launched.setdefault("selected", selected) and "http://127.0.0.1:8910/v1")
    _base_ensure_tuned_server()(config, object())
    assert launched["selected"].name == "mtp-3"


def test_explicit_tune_server_owns_benchmark_and_cache_write(monkeypatch):
    config = _stub_cold_runtime(monkeypatch, "explicit")
    decision = _decision("explicit", "mtp-2")
    monkeypatch.setattr(autotune, "_load_cached_decision", lambda _fp: None)
    calls = {"benchmark": 0, "saved": []}
    def benchmark(*_a, **_k):
        calls["benchmark"] += 1
        return decision
    monkeypatch.setattr(autotune, "_benchmark", benchmark)
    monkeypatch.setattr(autotune, "_save_decision", calls["saved"].append)
    assert autotune.tune_server(config, object()) is decision
    assert calls == {"benchmark": 1, "saved": [decision]}


def test_legacy_inline_autotune_requires_explicit_opt_in(monkeypatch):
    config = _stub_cold_runtime(monkeypatch, "inline")
    monkeypatch.setenv("MMM_LLAMA_SERVER_AUTOTUNE", "1")
    monkeypatch.setattr(autotune, "_load_cached_decision", lambda _fp: None)
    decision = _decision("inline", "mtp-1")
    calls = {"benchmark": 0}
    def benchmark(*_a, **_k):
        calls["benchmark"] += 1
        return decision
    monkeypatch.setattr(autotune, "_benchmark", benchmark)
    monkeypatch.setattr(autotune, "_save_decision", lambda _d: None)
    monkeypatch.setattr(autotune, "_launch_selected", lambda *_a, **_k: "http://127.0.0.1:8910/v1")
    _base_ensure_tuned_server()(config, object())
    assert calls["benchmark"] == 1
