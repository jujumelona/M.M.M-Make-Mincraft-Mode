from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai import qwen_runtime_transport_contract as contract


def _config(widths=None) -> SimpleNamespace:
    extra = {"runtime_contract": "qwen"}
    if widths is not None:
        extra["mtp_widths"] = widths
    return SimpleNamespace(
        model_id="vendor/arbitrary-runtime-model",
        extra=extra,
    )


def test_default_mtp_widths_are_read_from_registry_metadata() -> None:
    assert contract._recommended_mtp_widths(_config("1,2,4")) == "1,2,4"
    assert contract._recommended_mtp_widths(_config([2, 4, 4])) == "2,4"
    assert contract._recommended_mtp_widths(_config()) is None
    assert contract._recommended_mtp_widths(_config("0,2")) is None
    assert contract._recommended_mtp_widths(_config("bad")) is None


def test_width_policy_scopes_registry_default_and_restores_environment(monkeypatch) -> None:
    seen: list[str] = []

    def ensure(_config, _request) -> str:
        seen.append(os.environ.get("MMM_LLAMA_MTP_WIDTHS", ""))
        return "http://127.0.0.1:8910/v1"

    fake = SimpleNamespace(ensure_tuned_server=ensure)
    contract._install_mtp_width_policy(fake)
    monkeypatch.delenv("MMM_LLAMA_MTP_WIDTHS", raising=False)

    result = fake.ensure_tuned_server(_config("1,3"), object())

    assert result == "http://127.0.0.1:8910/v1"
    assert seen == ["1,3"]
    assert "MMM_LLAMA_MTP_WIDTHS" not in os.environ


def test_explicit_operator_widths_are_never_overridden(monkeypatch) -> None:
    seen: list[str] = []

    def ensure(_config, _request) -> str:
        seen.append(os.environ.get("MMM_LLAMA_MTP_WIDTHS", ""))
        return "http://127.0.0.1:8910/v1"

    fake = SimpleNamespace(ensure_tuned_server=ensure)
    contract._install_mtp_width_policy(fake)
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTHS", "2,4")

    fake.ensure_tuned_server(_config("1,3"), object())

    assert seen == ["2,4"]
    assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == "2,4"
