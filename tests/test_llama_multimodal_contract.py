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


def _variant(name: str, spec_type: str = "none", draft_n_max: int = 0) -> SimpleNamespace:
    return SimpleNamespace(name=name, spec_type=spec_type, draft_n_max=draft_n_max)


def _server_variant(name: str) -> SimpleNamespace:
    return _variant(name)


def _qwen_config(model_id: str, gguf_filename: str = "") -> SimpleNamespace:
    extra = {"mmproj_filename": "mmproj-F16.gguf"}
    if gguf_filename:
        extra["gguf_filename"] = gguf_filename
    return SimpleNamespace(model_id=model_id, extra=extra)


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


def test_projector_is_loaded_only_for_exact_managed_media_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"projector")
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nMMM")
    config = _qwen_config("unsloth/Qwen3.5-9B-MTP-GGUF")
    monkeypatch.setattr(
        multimodal,
        "_resolve_mmproj_path",
        lambda _config: str(projector),
    )
    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)

    autotune = SimpleNamespace()
    autotune._MANAGED_PROCESS = _RunningProcess()
    autotune._MANAGED_URL = "http://127.0.0.1:8910/v1"
    autotune.ServerVariant = _server_variant

    def dummy_launch(*_args):
        return "http://127.0.0.1:8920/v1"

    autotune._launch_selected = dummy_launch
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

    media_request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect."},),
        media_paths=(image,),
    )
    url = autotune.ensure_tuned_server(config, media_request)
    media_process = autotune._MANAGED_PROCESS

    assert shutdown_urls == ["http://127.0.0.1:8910/v1"]
    assert url == "http://127.0.0.1:8920/v1"
    assert launched_args[-2:] == ["--mmproj", str(projector)]
    assert os.environ.get(multimodal._ACTIVE_MEDIA_ENV) is None
    assert getattr(autotune, multimodal._MANAGED_MEDIA_PROCESS_ATTR) is media_process

    payload = hardware._server_payload(SimpleNamespace(config=config), media_request)
    assert payload["messages"][0]["content"][0]["type"] == "image_url"

    # Returning to text retires the exact media process and relaunches a lean server.
    # The fake allocator deliberately reuses the same URL, so identity rather than
    # URL equality must distinguish the new text process from the old media process.
    text_request = GenerationRequest(
        messages=({"role": "user", "content": "Continue coding."},),
    )
    assert autotune.ensure_tuned_server(config, text_request) == url
    text_process = autotune._MANAGED_PROCESS
    assert text_process is not media_process
    assert shutdown_urls[-1] == url
    assert not hasattr(autotune, multimodal._MANAGED_MEDIA_PROCESS_ATTR)

    # Same URL must not make the replacement text process look multimodal. A second
    # image request must retire it and launch a fresh projector-backed process.
    assert autotune.ensure_tuned_server(config, media_request) == url
    assert autotune._MANAGED_PROCESS is not text_process
    assert shutdown_urls[-1] == url


def test_media_baseline_requirement_covers_all_production_qwen_mtp_models() -> None:
    qwen35_9b = _qwen_config("unsloth/Qwen3.5-9B-MTP-GGUF")
    qwen36_35b = _qwen_config("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    qwen36_27b_q4 = _qwen_config(
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        "Qwen3.6-27B-UD-Q4_K_XL.gguf",
    )
    qwen36_27b_q3 = _qwen_config(
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        "Qwen3.6-27B-Q3_K_M.gguf",
    )

    assert multimodal._requires_media_baseline(qwen35_9b)
    assert multimodal._requires_media_baseline(qwen36_35b)
    assert multimodal._requires_media_baseline(qwen36_27b_q4)
    assert multimodal._requires_media_baseline(qwen36_27b_q3)


def test_media_launch_policy_preserves_text_mtp_and_disables_vision_mtp(monkeypatch) -> None:
    launched: list[tuple[str, str]] = []

    def launch_selected(_binary, _model_path, config, selected):
        launched.append((config.model_id, selected.spec_type))
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        _launch_selected=launch_selected,
        ServerVariant=_server_variant,
    )
    multimodal._install_launch_policy(autotune)
    speculative = _variant("mtp-2", "draft-mtp", 2)
    qwen35_9b = _qwen_config("unsloth/Qwen3.5-9B-MTP-GGUF")
    qwen36_35b = _qwen_config("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    qwen36_27b = _qwen_config(
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        "Qwen3.6-27B-UD-Q4_K_XL.gguf",
    )

    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)
    autotune._launch_selected("server", "model.gguf", qwen35_9b, speculative)
    assert launched[-1][1] == "draft-mtp"

    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")
    autotune._launch_selected("server", "model.gguf", qwen35_9b, speculative)
    autotune._launch_selected("server", "model.gguf", qwen36_35b, speculative)
    autotune._launch_selected("server", "model.gguf", qwen36_27b, speculative)

    assert [spec_type for _, spec_type in launched[-3:]] == [
        "none",
        "none",
        "none",
    ]
