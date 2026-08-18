from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal
from minecraft_mod_ai.model_adapters.base import GenerationRequest


class _RunningProcess:
    @staticmethod
    def poll():
        return None


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


def test_projector_is_loaded_only_when_media_upgrades_managed_server(
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
    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)

    autotune = SimpleNamespace()
    autotune._MANAGED_PROCESS = _RunningProcess()
    autotune._MANAGED_URL = "http://127.0.0.1:8910/v1"
    monkeypatch.setenv("LLAMA_SERVER_URL", autotune._MANAGED_URL)
    launched_args: list[str] = []
    shutdown_urls: list[str] = []

    def base_args(binary, model_path, _config, port):
        return [binary, "-m", model_path, "--port", str(port)]

    def shutdown():
        shutdown_urls.append(autotune._MANAGED_URL)
        autotune._MANAGED_PROCESS = None
        autotune._MANAGED_URL = None

    def ensure(_config, _request):
        if autotune._MANAGED_PROCESS is not None:
            return str(autotune._MANAGED_URL)
        launched_args.extend(
            autotune._base_args("llama-server", "model.gguf", config, 8920)
        )
        autotune._MANAGED_PROCESS = _RunningProcess()
        autotune._MANAGED_URL = "http://127.0.0.1:8920/v1"
        os.environ["LLAMA_SERVER_URL"] = autotune._MANAGED_URL
        return autotune._MANAGED_URL

    autotune._base_args = base_args
    autotune._shutdown_managed_server = shutdown
    autotune.ensure_tuned_server = ensure
    hardware = SimpleNamespace(
        _server_payload=lambda _adapter, request: {
            "messages": [dict(message) for message in request.messages]
        }
    )
    multimodal.install(autotune, hardware)

    # Text-only launch args stay lean and never resolve/load the projector.
    text_args = autotune._base_args("llama-server", "model.gguf", config, 8910)
    assert "--mmproj" not in text_args

    request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect."},),
        media_paths=(image,),
    )
    url = autotune.ensure_tuned_server(config, request)

    assert shutdown_urls == ["http://127.0.0.1:8910/v1"]
    assert url == "http://127.0.0.1:8920/v1"
    assert launched_args[-2:] == ["--mmproj", str(projector)]
    assert os.environ.get(multimodal._ACTIVE_MEDIA_ENV) is None

    payload = hardware._server_payload(SimpleNamespace(config=config), request)
    assert payload["messages"][0]["content"][0]["type"] == "image_url"