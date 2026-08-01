import base64
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from minecraft_mod_ai.model_adapters.base import AdapterConfig, ModelBackendError
from minecraft_mod_ai.model_adapters.openai_compatible import (
    OpenAICompatibleAdapter,
)


def _png_bytes(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", size, "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_openai_compatible_image_adapter_uses_reviewed_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict]] = []
    encoded = base64.b64encode(_png_bytes((512, 256))).decode("ascii")

    class _Response:
        content = b""

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"data": [{"b64_json": encoded}]}

    class _Client:
        def __init__(self, **kwargs) -> None:
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, url: str, **kwargs):
            calls.append((url, kwargs["json"]))
            assert kwargs["headers"]["Authorization"] == "Bearer secret"
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    config = AdapterConfig(
        role="image_generator",
        model_id="image-model",
        adapter="openai_compatible",
        provider="openai_compatible",
        base_url="https://models.example.test/v1",
        api_key="secret",
    )
    target = OpenAICompatibleAdapter(config).generate_image(
        prompt="Minecraft texture",
        output_path=tmp_path / "image.png",
        width=512,
        height=256,
        seed=123,
    )

    assert target.is_file()
    assert calls == [
        (
            "https://models.example.test/v1/images/generations",
            {
                "model": "image-model",
                "prompt": "Minecraft texture",
                "size": "512x256",
                "response_format": "b64_json",
                "n": 1,
            },
        )
    ]


def test_openai_compatible_image_adapter_rejects_remote_url_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"data": [{"url": "https://attacker.example/image.png"}]}

    class _Client:
        def __init__(self, **_kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, _url: str, **_kwargs):
            return _Response()

        def get(self, *_args, **_kwargs):
            raise AssertionError("URL image fallback must never be fetched")

    monkeypatch.setattr(httpx, "Client", _Client)
    config = AdapterConfig(
        role="image_generator",
        model_id="image-model",
        adapter="openai_compatible",
        provider="openai_compatible",
        base_url="https://models.example.test/v1",
        api_key="secret",
    )

    with pytest.raises(
        ModelBackendError,
        match="must contain inline b64_json bytes",
    ):
        OpenAICompatibleAdapter(config).generate_image(
            prompt="Minecraft texture",
            output_path=tmp_path / "image.png",
            width=64,
            height=64,
            seed=123,
        )
    assert not (tmp_path / "image.png").exists()
