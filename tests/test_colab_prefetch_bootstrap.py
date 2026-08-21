from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.colab_prefetch_bootstrap as bootstrap
from minecraft_mod_ai.colab_prefetch_bootstrap import start


def test_colab_bootstrap_sets_worker_defaults_without_registry_lookup(monkeypatch) -> None:
    calls: list[str] = []

    class Registry:
        def __init__(self):
            calls.append("init")

        def load_profile(self, name: str):
            calls.append(name)
            return object()

    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", "receipt")
    monkeypatch.delenv("MMM_DISCOVERY_WORKERS", raising=False)
    monkeypatch.delenv("MMM_RESEARCH_WORKERS", raising=False)
    monkeypatch.setattr(bootstrap, "_colab_worker_defaults", lambda: (8, 4))
    start(SimpleNamespace(ModelRegistry=Registry))
    assert calls == []
    assert __import__("os").environ["MMM_DISCOVERY_WORKERS"] == "8"
    assert __import__("os").environ["MMM_RESEARCH_WORKERS"] == "4"


def test_colab_worker_defaults_separate_io_and_cpu_budgets(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        bootstrap.Path,
        "read_text",
        lambda self, **kwargs: "MemTotal: 13000000 kB\nMemAvailable: 8388608 kB\n",
    )

    assert bootstrap._colab_worker_defaults() == (8, 4)


def test_deleted_proposal_alias_migration_stays_retired() -> None:
    assert not hasattr(bootstrap, "_retarget_loaded_proposal_aliases")


def test_colab_bootstrap_preserves_explicit_worker_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", "receipt")
    monkeypatch.setenv("MMM_DISCOVERY_WORKERS", "17")
    monkeypatch.setenv("MMM_RESEARCH_WORKERS", "9")
    start(SimpleNamespace())
    assert __import__("os").environ["MMM_DISCOVERY_WORKERS"] == "17"
    assert __import__("os").environ["MMM_RESEARCH_WORKERS"] == "9"


def test_prefetch_bootstrap_is_inert_without_colab_setup(monkeypatch) -> None:
    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)
    monkeypatch.delenv("MMM_DISCOVERY_WORKERS", raising=False)
    monkeypatch.delenv("MMM_RESEARCH_WORKERS", raising=False)
    start(SimpleNamespace())
    assert "MMM_DISCOVERY_WORKERS" not in __import__("os").environ
    assert "MMM_RESEARCH_WORKERS" not in __import__("os").environ
