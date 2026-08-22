from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal
from minecraft_mod_ai import llama_server_runtime_tuning as runtime_tuning
from minecraft_mod_ai import qwen_runtime_transport_contract as qwen_runtime


def _config(*, native_mtp: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model_id="vendor/arbitrary-runtime-model",
        extra={
            "runtime_contract": "qwen",
            "qwen_family": "qwen3.6",
            "qwen_tool_markup": "qwen3_coder_xml",
            "qwen_action_thinking_control": "enable_thinking_false",
            "qwen_preserve_thinking": True,
            "qwen_reasoning_effort": False,
            "qwen_assistant_prefill": True,
            "mmproj_filename": "projector.gguf",
            "native_mtp": native_mtp,
        },
    )


def test_registry_guarded_text_mtp_is_p1_but_media_baseline_keeps_parallel_policy(
    monkeypatch,
) -> None:
    seen: list[tuple[str, str, int]] = []

    def launch_selected(_binary, _model_path, _config, selected):
        seen.append(
            (
                os.environ.get("MMM_LLAMA_PARALLEL", ""),
                str(getattr(selected, "spec_type", "none")),
                int(getattr(selected, "parallel", 1)),
            )
        )
        return "http://127.0.0.1:8910/v1"

    def fingerprint(*_args):
        return "base"

    fake = SimpleNamespace(
        _launch_selected=launch_selected,
        _fingerprint=fingerprint,
        ServerVariant=runtime_tuning.ServerVariant,
    )
    multimodal._install_launch_policy(fake)
    qwen_runtime._install_mtp_single_slot_policy(fake)

    speculative = runtime_tuning.ServerVariant(
        "mtp-2|p4",
        "draft-mtp",
        2,
        parallel=4,
    )
    config = _config(native_mtp=False)

    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "4")
    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")
    fake._launch_selected("server", "model.gguf", config, speculative)
    assert seen[-1] == ("4", "none", 1)

    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)
    fake._launch_selected("server", "model.gguf", config, speculative)
    assert seen[-1] == ("1", "draft-mtp", 1)
    assert os.environ["MMM_LLAMA_PARALLEL"] == "4"


def test_native_mtp_metadata_keeps_multimodal_speculation() -> None:
    assert multimodal._requires_media_baseline(_config(native_mtp=False)) is True
    assert multimodal._requires_media_baseline(_config(native_mtp=True)) is False
