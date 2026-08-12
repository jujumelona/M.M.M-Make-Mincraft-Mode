from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_hardware_policy as policy


def test_server_payload_explicitly_reuses_prompt_cache() -> None:
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=128))
    request = SimpleNamespace(
        messages=({"role": "user", "content": "x"},),
        response_format="text",
    )
    payload = policy._server_payload(adapter, request)
    assert payload["cache_prompt"] is True
    assert getattr(policy._server_payload, "_mmm_prompt_cache_reuse", False)


def test_prometheus_parser_reads_native_token_and_time_counters() -> None:
    text = """
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
llamacpp:prompt_tokens_total 1234
llamacpp:prompt_seconds_total 2.5
llamacpp:tokens_predicted_total 456
llamacpp:tokens_predicted_seconds_total 20.0
llamacpp:predicted_tokens_seconds 22.8
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} 10
"""
    values = policy._parse_prometheus_metrics(text)
    assert values["prompt_tokens_total"] == 1234
    assert values["tokens_predicted_total"] == 456
    assert values["tokens_predicted_seconds_total"] == 20.0
    assert values["predicted_tokens_seconds"] == 22.8
    assert "spec_decode_num_accepted_tokens_per_pos_total" not in values


def test_slot_snapshot_reads_counters_without_tokenizing() -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return [
                {
                    "id": 0,
                    "is_processing": True,
                    "n_prompt_tokens": 901,
                    "n_prompt_tokens_processed": 701,
                    "n_prompt_tokens_cache": 200,
                    "next_token": {"n_decoded": 321},
                }
            ]

    calls: list[tuple[str, float]] = []

    def get(url: str, timeout: float):
        calls.append((url, timeout))
        return Response()

    snapshot = policy._slot_snapshot(SimpleNamespace(get=get), "http://127.0.0.1:8910/v1")
    assert snapshot == {
        "prompt_tokens": 901,
        "prompt_processed": 701,
        "prompt_cached": 200,
        "output_tokens": 321,
    }
    assert calls == [("http://127.0.0.1:8910/slots", 0.75)]


def test_metrics_delta_accumulates_exact_native_counters(monkeypatch) -> None:
    monkeypatch.setattr(
        policy,
        "_TELEMETRY_TOTALS",
        {
            "prompt_tokens": 10,
            "output_tokens": 20,
            "generation_seconds": 2.0,
            "requests": 1,
        },
    )
    before = {
        "prompt_tokens_total": 100.0,
        "tokens_predicted_total": 200.0,
        "tokens_predicted_seconds_total": 8.0,
    }
    after = {
        "prompt_tokens_total": 150.0,
        "tokens_predicted_total": 280.0,
        "tokens_predicted_seconds_total": 12.0,
    }
    usage = policy._commit_metrics_delta(before, after)
    assert usage is not None
    assert usage["prompt_tokens"] == 50
    assert usage["output_tokens"] == 80
    assert usage["generation_seconds"] == 4.0
    assert usage["cumulative_prompt_tokens"] == 60
    assert usage["cumulative_output_tokens"] == 100
    assert usage["cumulative_generation_seconds"] == 6.0
    assert usage["cumulative_requests"] == 2
