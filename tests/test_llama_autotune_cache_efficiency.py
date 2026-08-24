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
