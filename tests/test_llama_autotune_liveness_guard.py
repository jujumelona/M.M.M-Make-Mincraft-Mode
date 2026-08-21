from __future__ import annotations

from types import SimpleNamespace

import pytest

import minecraft_mod_ai.llama_tuning_pipeline as pipeline_module
from minecraft_mod_ai.llama_tuning_pipeline import NativeLlamaTuningPipeline


def test_autotune_wall_budget_blocks_new_expensive_steps_and_restores_timeouts(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("MMM_LLAMA_AUTOTUNE_MAX_SECONDS", "30")
    monkeypatch.setenv("MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("MMM_LLAMA_SERVER_START_TIMEOUT", "250")
    monkeypatch.setenv("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", "240")

    autotune = SimpleNamespace()

    def original_start(*_args, **_kwargs):
        return "started"

    def original_probe(*_args, **_kwargs):
        return "probed"

    def original_benchmark(*_args, **_kwargs):
        assert pipeline_module.os.environ["MMM_LLAMA_SERVER_START_TIMEOUT"] == "12"
        assert pipeline_module.os.environ["MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT"] == "12"
        clock[0] += 31.0
        return autotune._start_server("bin", "model", object(), object(), 1)

    autotune._start_server = original_start
    autotune._probe_server = original_probe
    autotune._benchmark = original_benchmark
    pipeline = NativeLlamaTuningPipeline(
        autotune=autotune,
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    pipeline._install_autotune_liveness_guard()

    with pytest.raises(RuntimeError, match="wall-clock budget exhausted"):
        autotune._benchmark("bin", "model", object(), object(), "fingerprint")

    assert pipeline_module.os.environ["MMM_LLAMA_SERVER_START_TIMEOUT"] == "250"
    assert pipeline_module.os.environ["MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT"] == "240"
    assert pipeline_module._remaining_autotune_seconds() is None
