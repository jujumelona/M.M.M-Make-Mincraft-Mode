from __future__ import annotations

import __main__
from types import SimpleNamespace

from minecraft_mod_ai.colab_prefetch_bootstrap import start


def test_selected_colab_profile_is_resolved_on_first_import(monkeypatch) -> None:
    loaded: list[str] = []

    class Registry:
        def load_profile(self, name: str):
            loaded.append(name)
            return object()

    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", "receipt")
    monkeypatch.setenv("MMM_DISCOVERY_WORKERS", "17")
    monkeypatch.setenv("MMM_RESEARCH_WORKERS", "9")
    monkeypatch.setattr(__main__, "MODEL_PROFILE", "Qwen3.5-9B_6GB", raising=False)
    start(SimpleNamespace(ModelRegistry=Registry))
    assert loaded == ["Qwen3.5-9B_6GB"]
    assert __import__("os").environ["MMM_DISCOVERY_WORKERS"] == "17"
    assert __import__("os").environ["MMM_RESEARCH_WORKERS"] == "9"


def test_prefetch_bootstrap_is_inert_without_colab_setup(monkeypatch) -> None:
    loaded: list[str] = []

    class Registry:
        def load_profile(self, name: str):
            loaded.append(name)
            return object()

    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)
    monkeypatch.setattr(__main__, "MODEL_PROFILE", "Qwen3.5-9B_6GB", raising=False)
    start(SimpleNamespace(ModelRegistry=Registry))
    assert loaded == []
