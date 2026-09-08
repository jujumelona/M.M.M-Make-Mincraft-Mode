from __future__ import annotations

import os

import minecraft_mod_ai


def test_default_llama_parallelism_uses_scheduler_maximum(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_PARALLEL", raising=False)

    minecraft_mod_ai._configure_default_llama_parallelism()

    assert os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] == "8"


def test_default_llama_parallelism_honors_explicit_active_override(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "8")

    minecraft_mod_ai._configure_default_llama_parallelism()

    assert os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] == "3"


def test_default_llama_parallelism_tracks_positive_server_override(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "4")

    minecraft_mod_ai._configure_default_llama_parallelism()

    assert os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] == "4"


def test_default_llama_parallelism_keeps_auto_server_mode_parallel(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "-1")

    minecraft_mod_ai._configure_default_llama_parallelism()

    assert os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] == "8"
