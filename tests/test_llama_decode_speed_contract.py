from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import llama_server_hardware_policy as hardware_policy
from minecraft_mod_ai.llama_decode_speed_contract import (
    SpeedServerVariant,
    _decode_ratio,
    _explicit_parallel_requested,
    _kv_autotune_enabled,
    _kv_candidates,
    _kv_fingerprint,
    _latest_probe_for_shape,
    _mtp_p_min_candidates,
    _probe_p_min,
    _representative_benchmark_request,
    _tuning_objective,
)


def test_decode_speed_contract_is_installed() -> None:
    assert autotune._mmm_decode_speed_contract_installed is True
    assert getattr(autotune._benchmark, "_mmm_single_stream_decode_objective", False)
    assert getattr(autotune._benchmark, "_mmm_mtp_p_min_stage", False)
    assert getattr(autotune._variant_args, "_mmm_mtp_p_min_tuning", False)
    assert getattr(autotune._fingerprint, "_mmm_decode_objective_fingerprint", False)
    assert getattr(autotune.ensure_tuned_server, "_mmm_kv_decode_autotune", False)
    assert autotune._BENCHMARK_OUTPUT_TOKENS >= 256


def test_default_objective_is_auto(monkeypatch) -> None:
    monkeypatch.delenv("MMM_PERFORMANCE_MODE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_PARALLEL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_CONCURRENT_REQUESTS", raising=False)
    assert _tuning_objective() == "auto"
    assert _explicit_parallel_requested() is False


def test_explicit_throughput_mode_is_preserved(monkeypatch) -> None:
    monkeypatch.delenv("MMM_PERFORMANCE_MODE", raising=False)
    monkeypatch.setenv("MMM_LLAMA_TUNING_OBJECTIVE", "throughput")
    monkeypatch.setenv("MMM_LLAMA_CONCURRENT_REQUESTS", "4")
    assert _tuning_objective() == "throughput"
    assert _explicit_parallel_requested() is True


def test_notebook_performance_mode_has_priority_over_legacy_objective(monkeypatch) -> None:
    monkeypatch.setenv("MMM_PERFORMANCE_MODE", "latency")
    monkeypatch.setenv("MMM_LLAMA_TUNING_OBJECTIVE", "throughput")
    assert _tuning_objective() == "latency"


def test_mtp_p_min_candidates_are_bounded_and_include_baseline(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_P_MIN_CANDIDATES", "0,0.5,0.8,0.95,1.0,-1,bad")
    assert _mtp_p_min_candidates() == (0.0, 0.5, 0.8, 0.95)


def test_mtp_p_min_reaches_native_server_args() -> None:
    args = autotune._variant_args(
        SpeedServerVariant("mtp-2|pm0.8", "draft-mtp", 2, draft_p_min=0.8)
    )
    assert args[args.index("--spec-draft-n-max") + 1] == "2"
    assert args[args.index("--spec-draft-p-min") + 1] == "0.8"


def test_representative_probe_does_not_reuse_real_prompt() -> None:
    secret = "REAL-MMM-WORKFLOW-PROMPT-MUST-NOT-ENTER-AUTOTUNE"
    request = SimpleNamespace(
        messages=({"role": "user", "content": secret},),
        response_format="json",
    )
    probe = _representative_benchmark_request(request)
    rendered = "\n".join(str(message["content"]) for message in probe.messages)
    assert secret not in rendered
    assert "modules" in rendered
    assert "depends_on" in rendered
    assert "0,1,2,...,63" not in rendered
    assert probe.response_format == "text"


def test_decode_ratio_ignores_prompt_prefill_speed() -> None:
    baseline = SimpleNamespace(predicted_tps=20.0, prompt_tps=1000.0)
    faster_decode_slower_prefill = SimpleNamespace(predicted_tps=24.0, prompt_tps=1.0)
    assert _decode_ratio(faster_decode_slower_prefill, baseline) == 1.2


def test_parallel_probe_lookup_requires_exact_execution_shape() -> None:
    selected = SpeedServerVariant(
        "mtp-2|p1", "draft-mtp", 2, ubatch=512, parallel=1, cache_reuse=64
    )
    p1 = SimpleNamespace(ok=True, variant=selected)
    p4 = SimpleNamespace(
        ok=True,
        variant=SpeedServerVariant(
            "mtp-2|p4", "draft-mtp", 2, ubatch=512, parallel=4, cache_reuse=64
        ),
    )
    assert _latest_probe_for_shape((p1, p4), selected) is p1


def test_p_min_latency_stage_remeasures_same_combined_workload(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_P_MIN_CANDIDATES", "0,0.8")
    calls: list[tuple[float, int]] = []

    def aggregate_probe(_autotune, _url, _request, *, max_tokens, variant, concurrency):
        p_min = float(variant.draft_p_min)
        calls.append((p_min, concurrency))
        return autotune.ProbeResult(
            variant=variant,
            ok=True,
            output_sha256="combined-short-medium",
            predicted_tokens=max_tokens,
            predicted_tps=15.0 if p_min == 0.8 else 10.0,
            prompt_tps=20.0,
            elapsed_seconds=1.0,
        )

    fake_autotune = SimpleNamespace(
        _compact_benchmark_request=lambda request: request,
        _env_int=lambda _name, default: default,
        _env_float=lambda _name, default: default,
        _free_port=lambda preferred: preferred,
        _start_server=lambda *_args: object(),
        _wait_ready=lambda _process, _port: "http://127.0.0.1:8910/v1",
        _probe_server=lambda _url, _request, *, max_tokens, variant: SimpleNamespace(ok=True),
        _stop_server=lambda _process: None,
        _BENCHMARK_OUTPUT_TOKENS=64,
        ProbeResult=autotune.ProbeResult,
    )
    fake_runtime = SimpleNamespace(_parallel_probe=aggregate_probe)
    selected = SpeedServerVariant(
        "mtp-2|pm0.8|p1", "draft-mtp", 2, ubatch=512, parallel=1, draft_p_min=0.8
    )
    raw_decode_probe = autotune.ProbeResult(
        variant=SpeedServerVariant("mtp-2|p1", "draft-mtp", 2, ubatch=512, parallel=1),
        ok=True,
        output_sha256="raw-short-only",
        predicted_tokens=64,
        predicted_tps=50.0,
        prompt_tps=20.0,
        elapsed_seconds=1.0,
    )
    winner, probes = _probe_p_min(
        fake_autotune,
        fake_runtime,
        "server",
        "model.gguf",
        SimpleNamespace(max_new_tokens=64),
        SimpleNamespace(messages=(), response_format="text"),
        selected,
        baseline_probe=raw_decode_probe,
    )
    assert calls == [(0.0, 1), (0.8, 1)]
    assert len(probes) == 2
    assert winner.parallel == 1
    assert winner.ubatch == 512
    assert winner.draft_p_min == 0.8


def test_kv_autotune_tries_all_supported_types_with_selected_type_first(monkeypatch) -> None:
    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "q8_0")
    monkeypatch.setenv("MMM_LLAMA_KV_CANDIDATES", "q4_0,q8_0,f16")
    assert _kv_candidates() == ("q8_0", "q4_0", "f16")


def test_kv_autotune_can_be_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_KV_AUTOTUNE", "0")
    monkeypatch.setenv("MMM_LLAMA_SERVER_AUTOTUNE", "1")
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)
    assert _kv_autotune_enabled(autotune) is False


def test_qwen35_structured_local_payload_keeps_json_validation_host_side_and_native_tools() -> None:
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            max_new_tokens=8192,
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        )
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    structured = SimpleNamespace(
        messages=({"role": "user", "content": "return json"},),
        response_format="json",
        response_schema=schema,
        tools=(),
    )
    payload = hardware_policy._server_payload(adapter, structured)
    assert structured.response_schema == schema
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload

    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    tool_request = SimpleNamespace(
        messages=({"role": "user", "content": "inspect then return json"},),
        response_format="json",
        response_schema=schema,
        tools=(tool,),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    tool_payload = hardware_policy._server_payload(adapter, tool_request)
    assert tool_payload["tools"] == [tool]
    assert tool_payload["tool_choice"] == "auto"
    assert tool_payload["parallel_tool_calls"] is True
    assert tool_payload["reasoning_effort"] == "none"
    assert tool_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in tool_payload
    assert "json_schema" not in tool_payload
    assert "grammar" not in tool_payload
    assert tool_request.response_schema == schema


def test_default_policy_searches_for_decode_speed_without_overfitting_exact_grid(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_MTP_P_MIN_CANDIDATES", raising=False)
    monkeypatch.delenv("MMM_LLAMA_KV_AUTOTUNE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_AUTOTUNE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)
    candidates = _mtp_p_min_candidates()
    assert candidates[0] == 0.0
    assert 0.8 in candidates
    assert all(0.0 <= value < 1.0 for value in candidates)
    assert _kv_autotune_enabled(autotune) is True


def test_kv_fingerprint_survives_fresh_runtime_path_and_mtime_changes(tmp_path, monkeypatch) -> None:
    first = tmp_path / "runtime-a" / "model.gguf"
    second = tmp_path / "runtime-b" / "model.gguf"
    first.parent.mkdir()
    second.parent.mkdir()
    payload = (b"GGUF" + bytes(range(256))) * 9000
    first.write_bytes(payload)
    second.write_bytes(payload)
    first.touch()
    second.touch()
    config = SimpleNamespace(
        model_id="example/model",
        max_context=32768,
        max_new_tokens=8192,
        extra={"gguf_filename": "model.gguf"},
    )
    monkeypatch.setattr(autotune, "_server_version", lambda binary: "llama-server-test")
    monkeypatch.setattr(autotune, "_hardware_identity", lambda: "gpu-test")
    candidates = ("q4_0", "q8_0", "f16")
    first_fp = _kv_fingerprint(autotune, config, "/bin/llama-server", str(first), candidates)
    second_fp = _kv_fingerprint(autotune, config, "/bin/llama-server", str(second), candidates)
    assert first_fp == second_fp
