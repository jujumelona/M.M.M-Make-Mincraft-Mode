from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai import llama_server_hardware_policy
from minecraft_mod_ai.model_adapters import ModelBackendError


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "minecraft_mod_ai" / "llama_server_hardware_policy.py"


class _Adapter:
    _reported_server_url: str | None = None

    def __init__(self) -> None:
        class Config:
            role = "planner"
            model_id = "model"
            max_new_tokens = 64

        self.config = Config()


class _Request:
    messages = ({"role": "user", "content": "hello"},)
    response_format = "text"


class _StreamResponse:
    def __init__(self, *, status_code: int = 200, lines=(), text: str = "") -> None:
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return self.text.encode("utf-8")


def _sse(content: str) -> list[str]:
    chunk = {
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "delta": {"content": content},
            }
        ]
    }
    return ["data: " + json.dumps(chunk), "data: [DONE]"]


def test_colab_llama_hot_path_uses_explicit_managed_server_without_local_fallback() -> None:
    text = POLICY.read_text(encoding="utf-8")
    assert "colab_mtp_server_running" in text
    assert 'os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL' in text
    assert "_mmm_explicit_server_strict" in text
    assert "return _strict_server_generate(self, request, explicit)" in text


def test_strict_server_generate_consumes_openai_sse_stream(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(lines=_sse("ok")),
    )
    adapter = _Adapter()
    assert (
        llama_server_hardware_policy._strict_server_generate(
            adapter,
            _Request(),
            "http://127.0.0.1:8910/v1",
        )
        == "ok"
    )


def test_strict_server_generate_surfaces_stream_server_failure(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(
            status_code=500,
            text="server failed",
        ),
    )
    with pytest.raises(ModelBackendError):
        llama_server_hardware_policy._strict_server_generate(
            _Adapter(),
            _Request(),
            "http://127.0.0.1:8910/v1",
        )
