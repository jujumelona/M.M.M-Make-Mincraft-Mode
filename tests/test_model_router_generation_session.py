from __future__ import annotations

from contextlib import contextmanager

import pytest

import minecraft_mod_ai.model_router as router_module
from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig,
    ModelConfigurationError,
)
from minecraft_mod_ai.model_router import ModelRouter


class _Registry:
    def __init__(self) -> None:
        self.configs = {
            role: AdapterConfig(
                role=role,
                adapter="vllm",
                model_id=f"test/{role}",
            )
            for role in ("planner", "coder")
        }

    def load_profile(self, name):
        assert name == "test"
        return object()

    def role(self, profile, role):
        assert profile == "test"
        return self.configs[role]


class _SessionAdapter:
    instances: list["_SessionAdapter"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.events: list[str] = []
        self.closed = 0
        type(self).instances.append(self)

    @contextmanager
    def generation_session(self):
        self.events.append("enter")
        try:
            yield self
        finally:
            self.events.append("exit")
            self.close()

    def generate(self, request):
        self.events.append(request.messages[-1]["content"])
        return "ok"

    def close(self):
        self.closed += 1


def _router(monkeypatch: pytest.MonkeyPatch) -> ModelRouter:
    _SessionAdapter.instances = []
    monkeypatch.setattr(
        router_module,
        "VLLMAdapter",
        _SessionAdapter,
    )
    return ModelRouter(profile="test", registry=_Registry())


def test_router_reuses_exactly_one_adapter_inside_generation_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router(monkeypatch)

    with router.generation_session("planner"):
        assert router.generate_text(
            "planner",
            ({"role": "user", "content": "page one"},),
        ) == "ok"
        assert router.generate_text(
            "planner",
            ({"role": "user", "content": "page two"},),
        ) == "ok"

    assert len(_SessionAdapter.instances) == 1
    adapter = _SessionAdapter.instances[0]
    assert adapter.events == ["enter", "page one", "page two", "exit"]
    assert adapter.closed == 1


def test_router_generation_session_releases_adapter_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router(monkeypatch)

    with pytest.raises(RuntimeError, match="stop planning"):
        with router.generation_session("planner"):
            router.generate_text(
                "planner",
                ({"role": "user", "content": "page one"},),
            )
            raise RuntimeError("stop planning")

    assert len(_SessionAdapter.instances) == 1
    assert _SessionAdapter.instances[0].closed == 1


def test_router_session_rejects_a_different_generation_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router(monkeypatch)

    with router.generation_session("planner"):
        with pytest.raises(
            ModelConfigurationError,
            match="cannot serve role 'coder'",
        ):
            router.generate_text(
                "coder",
                ({"role": "user", "content": "do not load coder"},),
            )

    assert len(_SessionAdapter.instances) == 1
