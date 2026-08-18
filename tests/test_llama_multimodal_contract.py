from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal
from minecraft_mod_ai.model_adapters.base import GenerationRequest


def test_media_paths_become_openai_image_parts(tmp_path: Path) -> None:
    image = tmp_path / "review.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nMMM")
    request = GenerationRequest(
        messages=(
            {"role": "system", "content": "Inspect screenshots."},
            {"role": "user", "content": "Return the visual verdict."},
        ),
        media_paths=(image,),
        response_format="json",
    )

    messages = multimodal._messages_with_media(request)

    assert messages[0] == {"role": "system", "content": "Inspect screenshots."}
    parts = messages[1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1] == {"type": "text", "text": "Return the visual verdict."}


def test_multimodal_install_binds_projector_to_server_and_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"projector")
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nMMM")
    config = SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"mmproj_filename": "mmproj-F16.gguf"},
    )
    monkeypatch.setattr(
        multimodal,
        "_resolve_mmproj_path",
        lambda _config: str(projector),
    )
    autotune = SimpleNamespace(
        _base_args=lambda binary, model_path, _config, port: [
            binary,
            "-m",
            model_path,
            "--port",
            str(port),
        ],
        _fingerprint=lambda _config, _binary, _model_path: "base-fingerprint",
    )
    hardware = SimpleNamespace(
        _server_payload=lambda _adapter, request: {
            "messages": [dict(message) for message in request.messages]
        }
    )
    multimodal.install(autotune, hardware)

    args = autotune._base_args("llama-server", "model.gguf", config, 8910)
    assert args[-2:] == ["--mmproj", str(projector)]
    assert autotune._fingerprint(config, "llama-server", "model.gguf") != "base-fingerprint"

    request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect."},),
        media_paths=(image,),
    )
    payload = hardware._server_payload(SimpleNamespace(config=config), request)
    assert payload["messages"][0]["content"][0]["type"] == "image_url"
