from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai.llama_server_runtime_tuning import ServerVariant
from minecraft_mod_ai.planner_single_stream_search_contract import (
    install as install_single_stream_plan_search,
)
from minecraft_mod_ai.qwen35_mtp_hotpath_contract import (
    _install_measured_fast_base_args,
    _is_qwen35_mtp,
    install as install_qwen35_hotpath,
)


def _qwen_config():
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
    )


def test_qwen35_mtp_detection_is_profile_specific() -> None:
    assert _is_qwen35_mtp(_qwen_config()) is True
    assert _is_qwen35_mtp(
        SimpleNamespace(
            model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
            extra={"gguf_filename": "Qwen3.6-27B-Q3_K_M.gguf"},
        )
    ) is False


def test_measured_fast_args_remove_generic_cache_experiments(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_HOTPATH", raising=False)

    def base_args(binary, model_path, config, port):
        del binary, model_path, config, port
        return [
            "llama-server",
            "--gpu-layers", "all",
            "--flash-attn", "on",
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
            "--load-mode", "auto",
            "--cache-prompt",
            "--ctx-size", "16384",
        ]

    autotune = SimpleNamespace(_base_args=base_args)
    _install_measured_fast_base_args(autotune)
    args = autotune._base_args("server", "model", _qwen_config(), 8910)

    assert args[args.index("--gpu-layers") + 1] == "all"
    assert args[args.index("--flash-attn") + 1] == "on"
    assert args[args.index("--batch-size") + 1] == "2048"
    assert args[args.index("--ubatch-size") + 1] == "512"
    assert "--cache-type-k" not in args
    assert "--cache-type-v" not in args
    assert "--load-mode" not in args
    assert "--cache-prompt" not in args
    assert "--metrics" in args
    # Context remains host-owned; speed restoration must not silently shrink it.
    assert args[args.index("--ctx-size") + 1] == "16384"


def test_qwen35_hotpath_selects_mtp3_single_stream(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_HOTPATH", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_WIDTH", raising=False)
    selected = []

    def env_int(name, default, minimum=1, maximum=None):
        del name
        value = max(minimum, int(default))
        return min(maximum, value) if maximum is not None else value

    def launch(binary, model_path, config, variant):
        del binary, model_path, config
        selected.append(variant)
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        ensure_tuned_server=lambda config, request: "fallback",
        _base_args=lambda binary, model_path, config, port: [
            binary, "-m", model_path, "--port", str(port)
        ],
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
        _AUTOTUNE_LOCK=threading.RLock(),
        _env_int=env_int,
        _server_binary=lambda: "llama-server",
        _resolve_model_path=lambda config: "/tmp/qwen35.gguf",
        _launch_selected=launch,
        ServerVariant=ServerVariant,
    )
    install_qwen35_hotpath(autotune)

    url = autotune.ensure_tuned_server(_qwen_config(), object())
    assert url == "http://127.0.0.1:8910/v1"
    assert len(selected) == 1
    variant = selected[0]
    assert variant.spec_type == "draft-mtp"
    assert variant.draft_n_max == 3
    assert variant.parallel == 1
    assert variant.ubatch == 512


def test_single_stream_auto_planner_search_collapses_to_one(monkeypatch) -> None:
    state = {"mode": "auto"}
    agentic = SimpleNamespace(
        _mode=lambda: state["mode"],
        _planner_candidate_count=lambda request, stage: 2,
    )
    install_single_stream_plan_search(agentic)

    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert agentic._planner_candidate_count({}, "outline") == 1

    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    assert agentic._planner_candidate_count({}, "outline") == 2

    state["mode"] = "on"
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert agentic._planner_candidate_count({}, "outline") == 2
