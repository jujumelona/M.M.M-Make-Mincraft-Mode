import json
from pathlib import Path

import httpx

from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.openai_compatible import (
    OpenAICompatibleAdapter,
)
from minecraft_mod_ai.model_router import ModelRouter


def test_openai_compatible_speech_adapter_uses_bounded_multipart_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "request.wav"
    source.write_bytes(b"RIFF" + b"\0" * 64)
    calls: list[tuple[str, str, str, bytes]] = []

    class _Response:
        content = json.dumps({"text": "계절 작물을 추가해 줘"}).encode(
            "utf-8"
        )

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"text": "계절 작물을 추가해 줘"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, url: str, **kwargs):
            filename, stream, mime_type = kwargs["files"]["file"]
            calls.append(
                (
                    url,
                    filename,
                    mime_type,
                    stream.read(),
                )
            )
            assert kwargs["headers"] == {
                "Authorization": "Bearer secret"
            }
            assert kwargs["data"] == {"model": "speech-model"}
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    config = AdapterConfig(
        role="speech_recognition",
        model_id="speech-model",
        adapter="openai_compatible",
        provider="openai_compatible",
        base_url="https://models.example.test/v1",
        api_key="secret",
        extra={"max_audio_bytes": 1024},
    )

    text = OpenAICompatibleAdapter(config).transcribe(source)

    assert text == "계절 작물을 추가해 줘"
    assert calls == [
        (
            "https://models.example.test/v1/audio/transcriptions",
            "request.wav",
            "audio/wav",
            source.read_bytes(),
        )
    ]


def test_model_router_dispatches_remote_speech_role(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "request.ogg"
    source.write_bytes(b"OggS")
    config = AdapterConfig(
        role="speech_recognition",
        model_id="speech-model",
        adapter="openai_compatible",
        provider="openai_compatible",
        base_url="https://models.example.test/v1",
        api_key="secret",
    )

    class _Registry:
        def load_profile(self, profile: str) -> dict:
            assert profile == "remote_quality"
            return {}

        def role(self, profile: str, role: str) -> AdapterConfig:
            assert profile == "remote_quality"
            assert role == "speech_recognition"
            return config

    monkeypatch.setattr(
        OpenAICompatibleAdapter,
        "transcribe",
        lambda self, path: f"transcribed:{path.name}",
    )
    router = ModelRouter(
        profile="remote_quality",
        registry=_Registry(),  # type: ignore[arg-type]
    )

    assert router.transcribe(
        "speech_recognition",
        source,
    ) == "transcribed:request.ogg"
