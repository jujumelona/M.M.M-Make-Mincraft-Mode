from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import colab_mtp_server


def _config(*, model_id: str, filename: str = "model.gguf") -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        max_context=32768,
        max_new_tokens=8192,
        extra={"gguf_filename": filename},
    )


def _request(response_format) -> SimpleNamespace:
    return SimpleNamespace(response_format=response_format)


@pytest.mark.parametrize(
    "response_format",
    [
        "json",
        "json_object",
        "json_schema",
        {"type": "json_object"},
        {"type": "json_schema"},
    ],
)
def test_structured_output_always_uses_baseline_even_for_mtp_model(
    response_format,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_POLICY", "auto")
    monkeypatch.setattr(colab_mtp_server, "_MTP_DISABLED_REASON", None)
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    assert colab_mtp_server.request_server_mode(config, _request(response_format)) == "baseline"


def test_free_text_can_use_mtp_only_for_mtp_capable_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_POLICY", "auto")
    monkeypatch.setattr(colab_mtp_server, "_MTP_DISABLED_REASON", None)
    mtp = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")
    gemma = _config(model_id="google/gemma-4-12B-it-qat-q4_0-gguf")

    assert colab_mtp_server.request_server_mode(mtp, _request("text")) == "mtp"
    assert colab_mtp_server.request_server_mode(gemma, _request("text")) == "baseline"


def test_runtime_mtp_disable_forces_future_text_requests_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_POLICY", "auto")
    monkeypatch.setattr(colab_mtp_server, "_MTP_DISABLED_REASON", "probe failed")
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    assert colab_mtp_server.request_server_mode(config, _request("text")) == "baseline"


def test_baseline_config_contains_no_speculative_draft_fields(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "server.json"
    monkeypatch.setattr(colab_mtp_server, "SERVER_CONFIG_PATH", config_path)
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    width = colab_mtp_server._write_config(config, "/tmp/model.gguf", mode="baseline")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert width == 0
    model = payload["model"]
    assert "draft_model" not in model
    assert "draft_model_num_pred_tokens" not in model
    assert model["n_ctx"] == colab_mtp_server.SERVER_CONTEXT_CAP


def test_mtp_config_is_explicit_and_bounded(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "server.json"
    monkeypatch.setattr(colab_mtp_server, "SERVER_CONFIG_PATH", config_path)
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTH", "3")
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    width = colab_mtp_server._write_config(config, "/tmp/model.gguf", mode="mtp")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert width == 3
    model = payload["model"]
    assert model["draft_model"] == "draft-mtp"
    assert model["draft_model_num_pred_tokens"] == 3
    assert model["n_seq_max"] == 1


def test_decode_log_summary_distinguishes_first_token_stall_from_server_progress() -> None:
    stalled = colab_mtp_server.decode_log_summary(
        "slot update: n_decoded = 1\nslot update: n_decoded = 1\n"
    )
    progressed = colab_mtp_server.decode_log_summary(
        "slot update: n_decoded = 1\nslot update: n_decoded = 42\n"
    )

    assert "stalled" in stalled
    assert "max=1" in stalled
    assert "advanced beyond the first token" in progressed
    assert "last=42" in progressed
