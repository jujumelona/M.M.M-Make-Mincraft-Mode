from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai.llama_tuning_pipeline import NativeLlamaTuningPipeline
from minecraft_mod_ai.qwen35_mtp_hotpath_contract import (
    _context_size,
    _install_measured_fast_base_args,
)


def _qwen_config(max_context: int = 262144):
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        max_context=max_context,
    )


def _generic_config(max_context: int = 131072):
    return SimpleNamespace(
        model_id="local/generic-gguf",
        extra={"gguf_filename": "generic.gguf"},
        max_context=max_context,
    )


def _install_context_authority(holder: SimpleNamespace) -> None:
    pipeline = NativeLlamaTuningPipeline(
        autotune=holder,
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    pipeline._install_profile_context_authority()


def test_base_args_default_to_model_native_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    args = autotune._base_args(
        "llama-server",
        "/tmp/model.gguf",
        _generic_config(),
        8910,
    )
    assert args[args.index("--ctx-size") + 1] == "0"


def test_base_args_honor_explicit_context_override(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "24576")
    args = autotune._base_args(
        "llama-server",
        "/tmp/model.gguf",
        _generic_config(),
        8910,
    )
    assert args[args.index("--ctx-size") + 1] == "24576"


def test_profile_authority_defaults_generic_server_to_model_native_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _generic_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "0"


def test_profile_authority_honors_explicit_generic_context_override(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "24576")

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _generic_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "24576"


def test_qwen_hotpath_alone_defaults_to_model_native_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_measured_fast_base_args(holder)
    args = holder._base_args("server", "model", _qwen_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "0"


def test_qwen_hotpath_cannot_shrink_model_native_context(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_measured_fast_base_args(holder)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _qwen_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "0"


def test_qwen_context_helper_uses_only_explicit_override(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    assert _context_size(_qwen_config()) == 0
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "16384")
    assert _context_size(_qwen_config()) == 16384


def test_profile_authority_preserves_explicit_qwen_context(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "16384")

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _qwen_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "16384"


def test_qwen_registry_capacity_is_not_forced_into_server_ctx(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "8192"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _qwen_config(262144), 8910)
    assert args[args.index("--ctx-size") + 1] == "0"
