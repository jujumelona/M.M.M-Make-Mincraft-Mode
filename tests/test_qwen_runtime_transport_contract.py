from __future__ import annotations

import hashlib
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import qwen_runtime_transport_contract as contract


def _config(model_id: str, *, max_context: int = 32768) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        max_context=max_context,
        max_new_tokens=8192,
        extra={"gguf_filename": f"{model_id.split('/')[-1]}.gguf"},
    )


def test_qwen36_uses_registry_context_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    config = _config("unsloth/Qwen3.6-27B-MTP-GGUF")
    args = autotune._base_args("llama-server", "/tmp/model.gguf", config, 8910)
    assert args[args.index("--ctx-size") + 1] == "32768"
    assert getattr(autotune._base_args, "_mmm_qwen_context_contract_v1", False)


def test_qwen36_explicit_native_context_override_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "0")
    config = _config("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    args = autotune._base_args("llama-server", "/tmp/model.gguf", config, 8910)
    assert args[args.index("--ctx-size") + 1] == "0"


def test_qwen35_keeps_its_existing_mtp_context_policy(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    config = _config("unsloth/Qwen3.5-9B-MTP-GGUF", max_context=262144)
    assert contract._actual_qwen_context(config) == 0


def test_qwen_context_fingerprint_tracks_effective_context(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"qwen" * 2048)
    monkeypatch.setattr(autotune, "_server_version", lambda _binary: "server-v1")
    monkeypatch.setattr(autotune, "_hardware_identity", lambda: "gpu-v1")
    config = _config("unsloth/Qwen3.6-27B-MTP-GGUF")

    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "8192")
    small = autotune._fingerprint(config, "llama-server", str(model))
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "16384")
    large = autotune._fingerprint(config, "llama-server", str(model))

    assert small != large
    assert getattr(autotune._fingerprint, "_mmm_qwen_context_fingerprint_v1", False)


def _tool_response(*, call_id: str, arguments: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "mmm_transport_probe",
                                "arguments": arguments,
                            },
                        }
                    ]
                }
            }
        ]
    }


def test_tool_signature_is_semantic_not_call_id_or_json_format() -> None:
    first = contract._tool_call_signature(
        _tool_response(call_id="call-a", arguments='{"value":7}')
    )
    second = contract._tool_call_signature(
        _tool_response(call_id="call-b", arguments='{ "value" : 7 }')
    )
    assert first
    assert first == second
    assert first == hashlib.sha256(
        b'{"arguments":{"value":7},"name":"mmm_transport_probe"}'
    ).hexdigest()


def test_tool_signature_rejects_wrong_tool_arguments() -> None:
    assert not contract._tool_call_signature(
        _tool_response(call_id="call-a", arguments='{"value":8}')
    )


def test_speculative_benchmark_falls_back_when_tool_equivalence_fails(monkeypatch) -> None:
    baseline = autotune.ServerVariant("baseline")
    speculative = autotune.ServerVariant("mtp-2", "draft-mtp", 2)
    probes = (
        autotune.ProbeResult(
            variant=baseline,
            ok=True,
            output_sha256="same",
            predicted_tokens=96,
            predicted_tps=20.0,
            prompt_tps=100.0,
            elapsed_seconds=1.0,
        ),
        autotune.ProbeResult(
            variant=speculative,
            ok=True,
            output_sha256="same",
            predicted_tokens=96,
            predicted_tps=30.0,
            prompt_tps=100.0,
            elapsed_seconds=1.0,
        ),
    )
    original = autotune.AutotuneDecision(
        fingerprint="fingerprint",
        selected=speculative,
        baseline_tps=20.0,
        selected_tps=30.0,
        speedup=1.5,
        probes=probes,
    )
    monkeypatch.setattr(
        contract,
        "_verify_speculative_tool_equivalence",
        lambda *_args, **_kwargs: (False, "probe mismatch"),
    )

    class FakeAutotune:
        ServerVariant = autotune.ServerVariant
        ProbeResult = autotune.ProbeResult
        AutotuneDecision = autotune.AutotuneDecision

        @staticmethod
        def _benchmark(*_args, **_kwargs):
            return original

    fake = FakeAutotune()
    contract._install_tool_equivalence_policy(fake)
    result = fake._benchmark(
        "llama-server",
        "/tmp/model.gguf",
        _config("unsloth/Qwen3.5-9B-MTP-GGUF", max_context=262144),
        object(),
        "fingerprint",
    )
    assert result.selected.name == "baseline"
    assert result.speedup == 1.0
    assert getattr(fake._benchmark, "_mmm_qwen_speculative_tool_equivalence_v1", False)
