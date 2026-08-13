from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai.agentic_search_efficiency_contract import _prime_native_slots
from minecraft_mod_ai.model_adapters.base import GenerationRequest


def test_every_native_server_start_path_includes_jinja() -> None:
    config = SimpleNamespace(max_context=32768)
    args = autotune._base_args("llama-server", "/tmp/model.gguf", config, 8910)
    assert "--jinja" in args
    assert args.count("--jinja") == 1


def test_planner_priming_uses_server_contract_that_already_owns_jinja(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    seen: dict[str, object] = {}

    config = SimpleNamespace(
        provider="local",
        adapter="llama_cpp",
        max_context=32768,
    )
    router = SimpleNamespace(
        registry=SimpleNamespace(role=lambda profile, role: config),
        profile="t4_local",
    )

    def ensure_tuned_server(received_config, request):
        seen["config"] = received_config
        seen["request"] = request
        args = autotune._base_args("llama-server", "/tmp/model.gguf", config, 8910)
        assert "--jinja" in args
        return "http://127.0.0.1:8910/v1"

    monkeypatch.setattr(autotune, "ensure_tuned_server", ensure_tuned_server)
    returned = _prime_native_slots(
        router,
        system_prompt="system",
        request={"plan": "test"},
        media_paths=(),
    )
    assert returned is config
    assert seen["config"] is config
    assert isinstance(seen["request"], GenerationRequest)


def test_hot_colab_reload_stops_old_managed_server_before_module_purge(
    monkeypatch, tmp_path
) -> None:
    import importlib.util
    from pathlib import Path

    setup_path = Path("tools/colab_runtime_setup.py").resolve()
    spec = importlib.util.spec_from_file_location("_mmm_test_colab_runtime_setup", setup_path)
    assert spec is not None and spec.loader is not None
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    calls: list[str] = []
    managed_url = "http://127.0.0.1:8910/v1"
    fake_autotune = SimpleNamespace(
        _MANAGED_URL=managed_url,
        _shutdown_managed_server=lambda: calls.append("shutdown"),
    )
    monkeypatch.setitem(
        sys.modules, "minecraft_mod_ai.llama_server_autotune", fake_autotune
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", managed_url)

    assert setup._shutdown_loaded_managed_llama_server() is True
    assert calls == ["shutdown"]
    assert "LLAMA_SERVER_URL" not in os.environ


def test_hot_colab_reload_preserves_unrelated_external_server_url(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    setup_path = Path("tools/colab_runtime_setup.py").resolve()
    spec = importlib.util.spec_from_file_location("_mmm_test_colab_runtime_setup_2", setup_path)
    assert spec is not None and spec.loader is not None
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    fake_autotune = SimpleNamespace(
        _MANAGED_URL="http://127.0.0.1:8910/v1",
        _shutdown_managed_server=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules, "minecraft_mod_ai.llama_server_autotune", fake_autotune
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", "https://example.invalid/v1")

    assert setup._shutdown_loaded_managed_llama_server() is True
    assert os.environ["LLAMA_SERVER_URL"] == "https://example.invalid/v1"
