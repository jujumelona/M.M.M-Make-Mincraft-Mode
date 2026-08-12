from __future__ import annotations

import minecraft_mod_ai.colab_mtp_server as colab_mtp_server


def test_colab_launcher_delegates_to_native_autotune(monkeypatch):
    config = object()
    seen = {}

    def fake_ensure_tuned_server(received_config, request):
        seen["config"] = received_config
        seen["messages"] = request.messages
        seen["response_format"] = request.response_format
        return "http://127.0.0.1:8910/v1"

    monkeypatch.setattr(colab_mtp_server, "ensure_tuned_server", fake_ensure_tuned_server)

    url = colab_mtp_server.start_colab_mtp_server(config)

    assert url == "http://127.0.0.1:8910/v1"
    assert seen == {
        "config": config,
        "messages": (),
        "response_format": "text",
    }


def test_colab_launcher_contains_no_forced_mtp_policy() -> None:
    assert not hasattr(colab_mtp_server, "_DEFAULT_MTP_WIDTH")
    assert not hasattr(colab_mtp_server, "_mtp_width")
