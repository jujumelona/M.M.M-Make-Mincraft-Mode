from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai.llama_tuning_pipeline import NativeLlamaTuningPipeline


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
    assert "--ctx-size" not in args
    assert "-c" not in args


def test_base_args_honor_explicit_context_override(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "24576")
    args = autotune._base_args(
        "llama-server",
        "/tmp/model.gguf",
        _generic_config(),
        8910,
    )
    assert args[args.index("--ctx-size") + 1] == "24576"


def test_profile_authority_removes_generic_inherited_context_without_override(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _generic_config(), 8910)
    assert "--ctx-size" not in args
    assert "-c" not in args


def test_profile_authority_honors_explicit_generic_context_override(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "24576")

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _generic_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "24576"


def _mimo_config(max_context: int = 32768):
    return SimpleNamespace(
        model_id="MiMo-V2.6-9B",
        extra={"gguf_filename": "MiMo-V2.6-9B.gguf"},
        max_context=max_context,
    )


def test_native_context_authority_does_not_inherit_stale_small_ctx_for_mimo(monkeypatch):
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _mimo_config(), 8910)
    assert "--ctx-size" not in args
    assert "-c" not in args


def test_mimo_profile_context_only_uses_explicit_valid_override(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "16384")

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "4096"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _mimo_config(), 8910)
    assert args[args.index("--ctx-size") + 1] == "16384"


def test_mimo_registry_capacity_is_not_forced_into_server_ctx(monkeypatch):
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)

    def base(binary, model, config, port):
        return [binary, "-m", model, "--port", str(port), "--ctx-size", "8192"]

    holder = SimpleNamespace(_base_args=base)
    _install_context_authority(holder)
    args = holder._base_args("server", "model", _mimo_config(32768), 8910)
    assert "--ctx-size" not in args
    assert "-c" not in args
