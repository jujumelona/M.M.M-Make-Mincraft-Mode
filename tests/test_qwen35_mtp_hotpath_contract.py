from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai.planner_single_stream_search_contract import (
    install as install_single_stream_plan_search,
)
from minecraft_mod_ai.qwen35_mtp_hotpath_contract import (
    _context_size,
    _install_measured_fast_base_args,
    _is_qwen35_mtp,
    install as install_qwen35_hotpath,
)


def _qwen_config():
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        max_context=32768,
    )


def _base_args(*_):
    return [
        "llama-server",
        "--gpu-layers",
        "auto",
        "--flash-attn",
        "off",
        "--batch-size",
        "1024",
        "--ubatch-size",
        "256",
        "--parallel",
        "2",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--load-mode",
        "auto",
        "--cache-prompt",
        "--ctx-size",
        "16384",
    ]


def test_qwen35_mtp_detection_is_profile_specific() -> None:
    assert _is_qwen35_mtp(_qwen_config()) is True
    assert _is_qwen35_mtp(
        SimpleNamespace(
            model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
            extra={"gguf_filename": "Qwen3.6-27B-Q3_K_M.gguf"},
        )
    ) is False


def test_measured_fast_args_preserve_kv_tuner_controls(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_HOTPATH", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)

    autotune = SimpleNamespace(_base_args=_base_args)
    _install_measured_fast_base_args(autotune)
    args = autotune._base_args("server", "model", _qwen_config(), 8910)

    assert args[args.index("--gpu-layers") + 1] == "all"
    assert args[args.index("--flash-attn") + 1] == "on"
    assert args[args.index("--batch-size") + 1] == "2048"
    assert args[args.index("--ubatch-size") + 1] == "512"
    assert args[args.index("--parallel") + 1] == "1"
    assert args[args.index("--ctx-size") + 1] == "32768"
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--cache-type-v") + 1] == "q8_0"
    assert "--load-mode" not in args
    assert "--cache-prompt" not in args
    assert "--metrics" in args


def test_qwen_context_defaults_to_profile_and_is_explicitly_overridable(monkeypatch) -> None:
    config = _qwen_config()
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    assert _context_size(config) == 32768
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "16384")
    assert _context_size(config) == 16384
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "999999")
    assert _context_size(config) == 999999
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "1024")
    assert _context_size(config) == 1024
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "0")
    assert _context_size(config) == 0
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "bad")
    try:
        _context_size(config)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid explicit Qwen context must fail")


def test_qwen35_hotpath_delegates_to_composed_measured_tuner(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_HOTPATH", raising=False)
    calls = []

    def measured_ensure(config, request):
        calls.append((config, request))
        os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = "draft-mtp"
        os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = "8"
        os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "1"
        os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] = "q8_0"
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        ensure_tuned_server=measured_ensure,
        _base_args=_base_args,
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
    )
    install_qwen35_hotpath(autotune)

    request = object()
    config = _qwen_config()
    url = autotune.ensure_tuned_server(config, request)

    assert url == "http://127.0.0.1:8910/v1"
    assert calls == [(config, request)]
    assert os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] == "8"
    assert os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] == "1"
    assert os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] == "q8_0"
    assert not hasattr(autotune, "_launch_selected")


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
