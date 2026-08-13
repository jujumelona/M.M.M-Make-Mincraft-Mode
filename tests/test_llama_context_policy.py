from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai.llama_tuning_pipeline import NativeLlamaTuningPipeline
from minecraft_mod_ai.qwen35_mtp_hotpath_contract import (
    _install_measured_fast_base_args,
)


def _qwen_config():
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        max_context=32768,
    )


def test_generic_server_defaults_to_full_profile_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    args = autotune._base_args(
        "llama-server",
        "/tmp/model.gguf",
        SimpleNamespace(max_context=32768),
        8910,
    )
    assert args[args.index("--ctx-size") + 1] == "32768"


def test_generic_server_honors_only_explicit_context_override(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "24576")
    args = autotune._base_args(
        "llama-server",
        "/tmp/model.gguf",
        SimpleNamespace(max_context=32768),
        8910,
    )
    assert args[args.index("--ctx-size") + 1] == "24576"


def test_qwen_hotpath_defaults_to_full_profile_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_measured_fast_base_args(holder)
    args = holder._base_args("server", "model", _qwen_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "32768"


def test_profile_authority_preserves_explicit_qwen_context(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "16384")

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "16384"]

    holder = SimpleNamespace(_base_args=base)
    pipeline = NativeLlamaTuningPipeline(
        autotune=holder,
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    pipeline._install_profile_context_authority()
    args = holder._base_args("server", "model", _qwen_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "16384"


def test_profile_authority_restores_default_qwen_profile_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "8192"]

    holder = SimpleNamespace(_base_args=base)
    pipeline = NativeLlamaTuningPipeline(
        autotune=holder,
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    pipeline._install_profile_context_authority()
    args = holder._base_args("server", "model", _qwen_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "32768"
