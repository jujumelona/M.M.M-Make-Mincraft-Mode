from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.colab_auto_platform_contract import install


def test_managed_colab_placeholder_is_hidden_from_platform_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", "sha256:test")

    class Planner:
        def __init__(self) -> None:
            self.router = SimpleNamespace(
                _mmm_requested_minecraft_version="1.20.1",
                _mmm_requested_loader="fabric",
            )

        def plan(self, prompt: str, *, media_paths=()):
            return (
                getattr(self.router, "_mmm_requested_minecraft_version", None),
                getattr(self.router, "_mmm_requested_loader", None),
            )

    module = SimpleNamespace(GameDesignPlanner=Planner)
    install(module)
    planner = Planner()

    observed = planner.plan("Make a simple mod")

    assert observed == (None, None)
    assert planner.router._mmm_requested_minecraft_version == "1.20.1"
    assert planner.router._mmm_requested_loader == "fabric"


def test_non_colab_explicit_target_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)

    class Planner:
        def __init__(self) -> None:
            self.router = SimpleNamespace(
                _mmm_requested_minecraft_version="1.20.1",
                _mmm_requested_loader="fabric",
            )

        def plan(self, prompt: str, *, media_paths=()):
            return self.router._mmm_requested_minecraft_version

    module = SimpleNamespace(GameDesignPlanner=Planner)
    install(module)

    assert Planner().plan("Make a simple mod") == "1.20.1"
