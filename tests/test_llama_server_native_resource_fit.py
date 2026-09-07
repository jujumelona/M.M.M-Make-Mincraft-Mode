from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.llama_server_autotune import ServerVariant, _base_args, _variant_args


_RESOURCE_ENV = (
    "MMM_LLAMA_PARALLEL",
    "MMM_LLAMA_SERVER_CTX",
    "MMM_LLAMA_BATCH",
    "MMM_LLAMA_UBATCH",
    "MMM_KV_CACHE_QUANT",
)


def _value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]


def test_default_launch_delegates_resource_sizing_to_llama_server(monkeypatch) -> None:
    for name in _RESOURCE_ENV:
        monkeypatch.delenv(name, raising=False)

    args = _base_args(
        "llama-server",
        "/tmp/model.gguf",
        SimpleNamespace(),
        8910,
    )

    assert _value(args, "--parallel") == "-1"
    assert _value(args, "--fit") == "on"
    for flag in (
        "--ctx-size",
        "--batch-size",
        "--ubatch-size",
        "--gpu-layers",
        "--flash-attn",
        "--cache-type-k",
        "--cache-type-v",
    ):
        assert flag not in args


def test_explicit_resource_overrides_are_forwarded_without_custom_clamping(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "3")
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "32768")
    monkeypatch.setenv("MMM_LLAMA_BATCH", "1024")
    monkeypatch.setenv("MMM_LLAMA_UBATCH", "1536")
    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "Q8_0")

    args = _base_args(
        "llama-server",
        "/tmp/model.gguf",
        SimpleNamespace(),
        8910,
    )

    assert _value(args, "--parallel") == "3"
    assert _value(args, "--ctx-size") == "32768"
    assert _value(args, "--batch-size") == "1024"
    assert _value(args, "--ubatch-size") == "1536"
    assert _value(args, "--cache-type-k") == "q8_0"
    assert _value(args, "--cache-type-v") == "q8_0"


def test_invalid_manual_overrides_fall_back_to_native_auto(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "invalid")
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "0")
    monkeypatch.setenv("MMM_LLAMA_BATCH", "-1")
    monkeypatch.setenv("MMM_LLAMA_UBATCH", "bad")

    args = _base_args(
        "llama-server",
        "/tmp/model.gguf",
        SimpleNamespace(),
        8910,
    )

    assert _value(args, "--parallel") == "-1"
    assert "--ctx-size" not in args
    assert "--batch-size" not in args
    assert "--ubatch-size" not in args


def test_mtp_variant_keeps_native_draft_gpu_placement() -> None:
    args = _variant_args(ServerVariant("mtp-2", "draft-mtp", 2))

    assert args == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    assert "--spec-draft-ngl" not in args
    assert "--spec-draft-n-min" not in args
