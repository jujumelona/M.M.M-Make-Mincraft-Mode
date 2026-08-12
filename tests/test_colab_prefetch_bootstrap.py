from __future__ import annotations

from types import SimpleNamespace

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
    start(SimpleNamespace(ModelRegistry=Registry))
    assert calls == []
    assert __import__("os").environ["MMM_DISCOVERY_WORKERS"] == "12"
    assert __import__("os").environ["MMM_RESEARCH_WORKERS"] == "8"


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
