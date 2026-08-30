from __future__ import annotations

from minecraft_mod_ai import llama_server_hardware_policy as hardware


def test_prefill_telemetry_uses_native_prompt_counter_deltas(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        hardware,
        "_TELEMETRY_TOTALS",
        {
            "prompt_tokens": 0,
            "prompt_seconds": 0.0,
            "output_tokens": 0,
            "generation_seconds": 0.0,
            "requests": 0,
        },
    )

    result = hardware._commit_metrics_delta(
        {
            "prompt_tokens_total": 100.0,
            "prompt_seconds_total": 2.0,
            "tokens_predicted_total": 10.0,
            "tokens_predicted_seconds_total": 1.0,
        },
        {
            "prompt_tokens_total": 500.0,
            "prompt_seconds_total": 4.0,
            "tokens_predicted_total": 30.0,
            "tokens_predicted_seconds_total": 2.0,
        },
    )

    assert result is not None
    assert result["prompt_tokens"] == 400
    assert result["prompt_seconds"] == 2.0
    assert result["prompt_tps"] == 200.0
    assert result["cumulative_prompt_tps"] == 200.0
    assert hardware._TELEMETRY_TOTALS["prompt_seconds"] == 2.0
    assert "prompt_tok_s=200.00" in capsys.readouterr().out


def test_prefill_telemetry_is_owned_directly_by_hardware_policy() -> None:
    assert not hasattr(hardware._commit_metrics_delta, "_mmm_prompt_prefill_telemetry_v1")
