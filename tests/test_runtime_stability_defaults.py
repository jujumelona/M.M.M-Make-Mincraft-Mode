from __future__ import annotations

import os

from minecraft_mod_ai import model_router
from minecraft_mod_ai.runtime_stability_defaults import install_runtime_stability_defaults


def _clear(monkeypatch) -> None:
    for name in (
        "MMM_AGENT_TOOL_ROUNDS",
        "MMM_SMALL_AGENT_CONTEXT_BYTES",
        "MMM_LLAMA_ENABLE_MTP",
        "MMM_LLAMA_MTP_WIDTHS",
        "MMM_LLAMA_MTP_CONFIDENCE_WIDTHS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_runtime_stability_defaults_bound_context_and_tool_rounds(monkeypatch) -> None:
    _clear(monkeypatch)

    install_runtime_stability_defaults()

    assert os.environ["MMM_AGENT_TOOL_ROUNDS"] == "12"
    assert os.environ["MMM_SMALL_AGENT_CONTEXT_BYTES"] == str(24 * 1024)
    assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == ""
    assert os.environ["MMM_LLAMA_MTP_CONFIDENCE_WIDTHS"] == ""
    assert model_router._agent_tool_round_limit() == 12


def test_explicit_context_and_round_overrides_are_preserved(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "7")
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", "20480")

    install_runtime_stability_defaults()

    assert os.environ["MMM_AGENT_TOOL_ROUNDS"] == "7"
    assert os.environ["MMM_SMALL_AGENT_CONTEXT_BYTES"] == "20480"


def test_mtp_requires_explicit_opt_in(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("MMM_LLAMA_ENABLE_MTP", "1")
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTHS", "1,2")
    monkeypatch.setenv("MMM_LLAMA_MTP_CONFIDENCE_WIDTHS", "4,8")

    install_runtime_stability_defaults()

    assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == "1,2"
    assert os.environ["MMM_LLAMA_MTP_CONFIDENCE_WIDTHS"] == "4,8"
