from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal
from minecraft_mod_ai import llama_server_runtime_tuning as runtime_tuning
from minecraft_mod_ai import qwen_runtime_transport_contract as qwen_runtime


def _config(model_id: str, filename: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        extra={
            "gguf_filename": filename,
            "mmproj_filename": "mmproj-F16.gguf",
        },
    )


def test_qwen_text_mtp_is_p1_but_media_baseline_keeps_parallel_policy(
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
    # Production order: multimodal is composed by the llama tuning pipeline first;
    # the Qwen runtime guard is installed later by the Qwen agent family contract.
    multimodal._install_launch_policy(fake)
    qwen_runtime._install_mtp_single_slot_policy(fake)

    speculative = runtime_tuning.ServerVariant(
        "mtp-2|p4",
        "draft-mtp",
        2,
        parallel=4,
    )
    configs = (
        _config(
            "unsloth/Qwen3.5-9B-MTP-GGUF",
            "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        ),
        _config(
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        ),
        _config(
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        ),
    )

    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "4")
    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")
    for config in configs:
        fake._launch_selected("server", "model.gguf", config, speculative)
        # Multimodal turns MTP into baseline, while the outer Qwen guard must not
        # temporarily overwrite the operator/runtime baseline parallel policy.
        assert seen[-1] == ("4", "none", 1)

    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)
    fake._launch_selected("server", "model.gguf", configs[1], speculative)
    assert seen[-1] == ("1", "draft-mtp", 1)
    assert os.environ["MMM_LLAMA_PARALLEL"] == "4"
