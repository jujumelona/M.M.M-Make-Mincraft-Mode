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
    _mtp_p_min_candidates,
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


def test_default_objective_is_single_stream(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_PARALLEL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_CONCURRENT_REQUESTS", raising=False)
    assert _tuning_objective() == "single_stream"
    assert _explicit_parallel_requested() is False


def test_explicit_throughput_mode_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_TUNING_OBJECTIVE", "throughput")
    monkeypatch.setenv("MMM_LLAMA_CONCURRENT_REQUESTS", "4")
    assert _tuning_objective() == "throughput"
    assert _explicit_parallel_requested() is True


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


def test_kv_autotune_tries_all_supported_types_with_selected_type_first(monkeypatch) -> None:
    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "q8_0")
    monkeypatch.setenv("MMM_LLAMA_KV_CANDIDATES", "q4_0,q8_0,f16")
    assert _kv_candidates() == ("q8_0", "q4_0", "f16")


def test_kv_autotune_can_be_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_KV_AUTOTUNE", "0")
    monkeypatch.setenv("MMM_LLAMA_SERVER_AUTOTUNE", "1")
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)
    assert _kv_autotune_enabled(autotune) is False


def test_structured_local_payload_is_host_validated_without_server_json_grammar() -> None:
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=8192))
    request = SimpleNamespace(
        messages=({"role": "user", "content": "return json"},),
        response_format="json",
    )
    payload = hardware_policy._server_payload(adapter, request)
    assert "response_format" not in payload
    assert payload["reasoning_effort"] == "none"
    assert getattr(
        hardware_policy._server_payload,
        "_mmm_host_validated_json_no_gbnf",
        False,
    )

def test_default_policy_searches_for_maximum_decode_speed(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_MTP_P_MIN_CANDIDATES", raising=False)
    monkeypatch.delenv("MMM_LLAMA_KV_AUTOTUNE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_AUTOTUNE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)
    assert _mtp_p_min_candidates() == (0.0, 0.6, 0.8, 0.9)
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

