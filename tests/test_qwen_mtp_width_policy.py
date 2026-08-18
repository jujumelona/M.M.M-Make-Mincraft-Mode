from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai import qwen_runtime_transport_contract as contract


def _config(model_id: str, filename: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        extra={"gguf_filename": filename},
    )


def test_default_mtp_widths_follow_production_model_recommendations() -> None:
    q35 = _config(
        "unsloth/Qwen3.5-9B-MTP-GGUF",
        "Qwen3.5-9B-UD-Q4_K_XL.gguf",
    )
    q27 = _config(
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        "Qwen3.6-27B-UD-Q4_K_XL.gguf",
    )
    q36 = _config(
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    )

    assert contract._recommended_mtp_widths(q35) == "1,2,3,4,5,6"
    assert contract._recommended_mtp_widths(q27) == "1,2"
    assert contract._recommended_mtp_widths(q36) == "1,2"


def test_width_policy_scopes_default_and_restores_environment(monkeypatch) -> None:
    seen: list[str] = []

    def ensure(_config, _request) -> str:
        seen.append(os.environ.get("MMM_LLAMA_MTP_WIDTHS", ""))
        return "http://127.0.0.1:8910/v1"

    fake = SimpleNamespace(ensure_tuned_server=ensure)
    contract._install_mtp_width_policy(fake)
    monkeypatch.delenv("MMM_LLAMA_MTP_WIDTHS", raising=False)

    result = fake.ensure_tuned_server(
        _config(
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        ),
        object(),
    )

    assert result == "http://127.0.0.1:8910/v1"
    assert seen == ["1,2"]
    assert "MMM_LLAMA_MTP_WIDTHS" not in os.environ


def test_explicit_operator_widths_are_never_overridden(monkeypatch) -> None:
    seen: list[str] = []

    def ensure(_config, _request) -> str:
        seen.append(os.environ.get("MMM_LLAMA_MTP_WIDTHS", ""))
        return "http://127.0.0.1:8910/v1"

    fake = SimpleNamespace(ensure_tuned_server=ensure)
    contract._install_mtp_width_policy(fake)
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTHS", "2,4")

    fake.ensure_tuned_server(
        _config(
            "unsloth/Qwen3.5-9B-MTP-GGUF",
            "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        ),
        object(),
    )

    assert seen == ["2,4"]
    assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == "2,4"
