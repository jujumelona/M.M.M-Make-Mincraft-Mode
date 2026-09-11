from __future__ import annotations

import os

import minecraft_mod_ai
from minecraft_mod_ai import hardware_concurrency_installation as hardware


def test_package_does_not_expose_active_capacity_synthesizer() -> None:
    assert not hasattr(minecraft_mod_ai, "_configure_default_llama_parallelism")


def test_hardware_install_does_not_invent_active_llama_capacity(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.delenv("MMM_CENTRAL_AI_WORKERS", raising=False)

    result = hardware.install()

    assert "MMM_LLAMA_ACTIVE_PARALLEL" not in os.environ
    assert result["central_ai_workers"] == 1
    assert os.environ["MMM_CENTRAL_AI_WORKERS_EFFECTIVE"] == "1"


def test_validated_active_capacity_drives_central_workers(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    monkeypatch.delenv("MMM_CENTRAL_AI_WORKERS", raising=False)

    result = hardware.install()

    assert os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] == "3"
    assert result["central_ai_workers"] == 3
    assert os.environ["MMM_CENTRAL_AI_WORKERS_EFFECTIVE"] == "3"


def test_desired_server_width_never_becomes_active_receipt(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "8")
    monkeypatch.delenv("MMM_CENTRAL_AI_WORKERS", raising=False)

    result = hardware.install()

    assert "MMM_LLAMA_ACTIVE_PARALLEL" not in os.environ
    assert result["central_ai_workers"] == 1
