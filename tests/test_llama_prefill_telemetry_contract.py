from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai.llama_prefill_telemetry_contract import install


def test_prefill_telemetry_uses_native_prompt_counter_deltas(capsys) -> None:
    lock = threading.Lock()
    totals = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "generation_seconds": 0.0,
        "requests": 0,
    }

    def base(before, after):
        prompt = int(after["prompt_tokens_total"] - before["prompt_tokens_total"])
        with lock:
            totals["prompt_tokens"] += prompt
        return {"prompt_tokens": prompt}

    hardware = SimpleNamespace(
        _commit_metrics_delta=base,
        _TELEMETRY_LOCK=lock,
        _TELEMETRY_TOTALS=totals,
    )
    install(hardware)

    result = hardware._commit_metrics_delta(
        {"prompt_tokens_total": 100.0, "prompt_seconds_total": 2.0},
        {"prompt_tokens_total": 500.0, "prompt_seconds_total": 4.0},
    )

    assert result["prompt_tokens"] == 400
    assert result["prompt_seconds"] == 2.0
    assert result["prompt_tps"] == 200.0
    assert result["cumulative_prompt_tps"] == 200.0
    assert hardware._TELEMETRY_TOTALS["prompt_seconds"] == 2.0
    assert "prompt_tok_s=200.00" in capsys.readouterr().out


def test_runtime_bootstrap_installs_prefill_telemetry() -> None:
    from minecraft_mod_ai import llama_server_hardware_policy as hardware

    assert getattr(
        hardware._commit_metrics_delta,
        "_mmm_prompt_prefill_telemetry_v1",
        False,
    )
