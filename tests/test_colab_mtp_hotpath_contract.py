from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_hardware_policy
from minecraft_mod_ai.model_adapters import ModelBackendError


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "minecraft_mod_ai" / "llama_server_hardware_policy.py"


def test_colab_mtp_hot_path_skips_repeated_restart_and_local_fallback() -> None:
    text = POLICY.read_text(encoding="utf-8")
    assert "colab_mtp_server_running" in text
    assert 'os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL' in text
    assert "_mmm_explicit_server_strict" in text
    assert "return _strict_server_generate(self, request, explicit)" in text


def test_strict_server_generate_returns_openai_message(monkeypatch) -> None:
    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}]}

    import httpx

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response())
    adapter = SimpleNamespace(
        config=SimpleNamespace(role="planner", model_id="model", max_new_tokens=64)
    )
    adapter.__class__._reported_server_url = None
    request = SimpleNamespace(
        messages=({"role": "user", "content": "hello"},),
        response_format="text",
    )
    assert (
        llama_server_hardware_policy._strict_server_generate(
            adapter,
            request,
            "http://127.0.0.1:8910/v1",
        )
        == "ok"
    )


def test_strict_server_generate_surfaces_server_failure(monkeypatch) -> None:
    class Response:
        status_code = 500
        text = "server failed"

        @staticmethod
        def json():
            return {}

    import httpx

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response())
    adapter = SimpleNamespace(
        config=SimpleNamespace(role="planner", model_id="model", max_new_tokens=64)
    )
    request = SimpleNamespace(
        messages=({"role": "user", "content": "hello"},),
        response_format="text",
    )
    with pytest.raises(ModelBackendError):
        llama_server_hardware_policy._strict_server_generate(
            adapter,
            request,
            "http://127.0.0.1:8910/v1",
        )
