from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai.planner_single_stream_search_contract import (
    install as install_single_stream_plan_search,
)
from minecraft_mod_ai.qwen35_mtp_hotpath_contract import (
    _install_measured_fast_base_args,
    _is_qwen35_mtp,
    _qwen_speed_search_defaults,
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
    assert args[args.index("--ctx-size") + 1] == "16384"


def test_qwen35_hotpath_delegates_to_measured_adaptive_search(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_HOTPATH", raising=False)
    for name in (
        "MMM_LLAMA_MTP_WIDTHS",
        "MMM_LLAMA_MTP_CONFIDENCE_WIDTHS",
        "MMM_LLAMA_MTP_SEED_P_MIN",
        "MMM_LLAMA_MTP_P_MIN_CANDIDATES",
        "MMM_LLAMA_NGRAM_SPEC_TYPES",
        "MMM_LLAMA_KV_AUTOTUNE",
        "MMM_LLAMA_TUNING_OBJECTIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    observed = []

    def adaptive(config, request):
        del config, request
        observed.append(
            {
                "widths": os.environ.get("MMM_LLAMA_MTP_WIDTHS"),
                "confidence_widths": os.environ.get("MMM_LLAMA_MTP_CONFIDENCE_WIDTHS"),
                "p_min": os.environ.get("MMM_LLAMA_MTP_P_MIN_CANDIDATES"),
                "ngram": os.environ.get("MMM_LLAMA_NGRAM_SPEC_TYPES"),
                "kv_autotune": os.environ.get("MMM_LLAMA_KV_AUTOTUNE"),
                "objective": os.environ.get("MMM_LLAMA_TUNING_OBJECTIVE"),
            }
        )
        os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = "draft-mtp"
        os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = "6"
        os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] = "0.8"
        os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = "512"
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        ensure_tuned_server=adaptive,
        _base_args=lambda binary, model_path, config, port: [
            binary, "-m", model_path, "--port", str(port)
        ],
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
    )
    install_qwen35_hotpath(autotune)

    url = autotune.ensure_tuned_server(_qwen_config(), object())
    assert url == "http://127.0.0.1:8910/v1"
    assert observed == [
        {
            "widths": "2,3",
            "confidence_widths": "3,6,8",
            "p_min": "0,0.6,0.8,0.9",
            "ngram": "",
            "kv_autotune": "0",
            "objective": "single_stream",
        }
    ]
    assert os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] == "draft-mtp"
    assert os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] == "6"
    assert os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] == "0.8"
    assert os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] == "native-default"
    # Scoped defaults must not leak into later non-Qwen inference.
    assert "MMM_LLAMA_MTP_WIDTHS" not in os.environ
    assert "MMM_LLAMA_KV_AUTOTUNE" not in os.environ


def test_qwen_search_defaults_preserve_explicit_user_values(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTHS", "3,12")
    monkeypatch.setenv("MMM_LLAMA_MTP_P_MIN_CANDIDATES", "0,0.75")
    with _qwen_speed_search_defaults():
        assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == "3,12"
        assert os.environ["MMM_LLAMA_MTP_P_MIN_CANDIDATES"] == "0,0.75"
    assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == "3,12"
    assert os.environ["MMM_LLAMA_MTP_P_MIN_CANDIDATES"] == "0,0.75"


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
