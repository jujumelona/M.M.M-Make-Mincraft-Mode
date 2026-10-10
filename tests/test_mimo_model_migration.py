from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.llama_multimodal_contract import _requires_media_baseline
from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup
from minecraft_mod_ai.llama_server_autotune import _assert_mimo_server_compatible


def test_default_t4_mimo_has_no_legacy_qwen_lora_or_mtp() -> None:
    for profile_name in ("t4_local", "t4_quality", "MiMo-V2.6-9B_6GB"):
        profile = ModelRegistry().load_profile(profile_name)
        for role in ("planner", "researcher", "coder", "coder_safe", "visual_critic"):
            config = profile.roles[role]
            assert config.model_id == "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF"
            assert config.extra["gguf_filename"] == "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"
            assert config.extra["mmproj_filename"] == "mmproj-MiMo-V2.6-Distill-Qwen-9B-f16.gguf"
            assert config.extra["runtime_contract"] == "mimo"
            assert config.extra["supports_mtp"] is False
            assert "decode_hotpath" not in config.extra
            assert "lora_adapters" not in config.extra
            assert "qwen_family" not in config.extra
            assert _requires_media_baseline(config)


def test_mimo_tool_argument_markup_can_use_parameters_or_json() -> None:
    for raw, expected in (
        ("<tool_call><function=write_file><parameter=path>src/A.java</parameter></function></tool_call>", "src/A.java"),
        ('<tool_call><function=write_file>{"path": "src/B.java"}</function></tool_call>', "src/B.java"),
    ):
        visible, calls = parse_qwen_tool_markup(raw)
        assert not visible
        assert len(calls) == 1
        assert calls[0].name == "write_file"
        assert calls[0].arguments["path"] == expected


def test_mimo_old_server_fails_closed(monkeypatch) -> None:
    from minecraft_mod_ai import llama_server_autotune
    config = SimpleNamespace(extra={"runtime_contract": "mimo"})
    monkeypatch.setattr(llama_server_autotune, "_server_version", lambda _: "version: 11101 (old)")
    with pytest.raises(RuntimeError, match="b11102"):
        _assert_mimo_server_compatible("llama-server", config)
    monkeypatch.setattr(llama_server_autotune, "_server_version", lambda _: "version: 11429 (new)")
    _assert_mimo_server_compatible("llama-server", config)
